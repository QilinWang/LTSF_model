# AB SWD

## FRIREN

### huber


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
    mixing_strategy='convex', ### Here
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

### huber + 0.5 SWD


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
    mixing_strategy='convex', ### Here
    loss_backward_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
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

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 236.0685, mae: 10.9817, huber: 10.4934, swd: 13.4349, ept: 4.3301
    Epoch [1/50], Val Losses: mse: 154.7652, mae: 9.0832, huber: 8.5961, swd: 1.0262, ept: 6.9456
    Epoch [1/50], Test Losses: mse: 152.9187, mae: 9.0046, huber: 8.5178, swd: 0.9803, ept: 6.5923
      Epoch 1 composite train-obj: 17.210837
            Val objective improved inf → 9.1092, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 133.0894, mae: 7.8518, huber: 7.3676, swd: 0.9524, ept: 9.3484
    Epoch [2/50], Val Losses: mse: 128.5885, mae: 7.5441, huber: 7.0609, swd: 1.1576, ept: 10.7787
    Epoch [2/50], Test Losses: mse: 123.9290, mae: 7.3999, huber: 6.9177, swd: 1.0781, ept: 10.6087
      Epoch 2 composite train-obj: 7.843838
            Val objective improved 9.1092 → 7.6397, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 109.1708, mae: 6.6534, huber: 6.1740, swd: 0.8949, ept: 13.1976
    Epoch [3/50], Val Losses: mse: 109.3616, mae: 6.7131, huber: 6.2326, swd: 0.9199, ept: 14.5735
    Epoch [3/50], Test Losses: mse: 107.1287, mae: 6.5313, huber: 6.0536, swd: 0.8323, ept: 14.0611
      Epoch 3 composite train-obj: 6.621455
            Val objective improved 7.6397 → 6.6926, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 93.1014, mae: 5.9513, huber: 5.4757, swd: 0.8097, ept: 17.3878
    Epoch [4/50], Val Losses: mse: 96.8895, mae: 6.2388, huber: 5.7608, swd: 1.0868, ept: 18.3313
    Epoch [4/50], Test Losses: mse: 94.2081, mae: 5.9466, huber: 5.4720, swd: 0.9376, ept: 17.9931
      Epoch 4 composite train-obj: 5.880514
            Val objective improved 6.6926 → 6.3042, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 83.6082, mae: 5.5211, huber: 5.0493, swd: 0.7829, ept: 22.1682
    Epoch [5/50], Val Losses: mse: 90.9592, mae: 5.9633, huber: 5.4878, swd: 0.8479, ept: 22.1467
    Epoch [5/50], Test Losses: mse: 88.7788, mae: 5.6517, huber: 5.1792, swd: 0.7363, ept: 22.1197
      Epoch 5 composite train-obj: 5.440684
            Val objective improved 6.3042 → 5.9117, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 75.9356, mae: 5.1627, huber: 4.6950, swd: 0.7542, ept: 30.0793
    Epoch [6/50], Val Losses: mse: 86.0300, mae: 5.7116, huber: 5.2389, swd: 0.8543, ept: 24.7610
    Epoch [6/50], Test Losses: mse: 84.0814, mae: 5.4086, huber: 4.9384, swd: 0.7363, ept: 25.0117
      Epoch 6 composite train-obj: 5.072101
            Val objective improved 5.9117 → 5.6660, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 69.6081, mae: 4.8392, huber: 4.3758, swd: 0.7290, ept: 42.5260
    Epoch [7/50], Val Losses: mse: 79.4924, mae: 5.4439, huber: 4.9745, swd: 0.8832, ept: 33.4359
    Epoch [7/50], Test Losses: mse: 76.9647, mae: 5.1011, huber: 4.6347, swd: 0.7425, ept: 33.1062
      Epoch 7 composite train-obj: 4.740350
            Val objective improved 5.6660 → 5.4161, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 63.8951, mae: 4.5474, huber: 4.0883, swd: 0.6902, ept: 57.7636
    Epoch [8/50], Val Losses: mse: 76.5154, mae: 5.3371, huber: 4.8688, swd: 0.8514, ept: 42.8335
    Epoch [8/50], Test Losses: mse: 72.7155, mae: 4.9207, huber: 4.4554, swd: 0.6981, ept: 44.1609
      Epoch 8 composite train-obj: 4.433366
            Val objective improved 5.4161 → 5.2945, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 60.0294, mae: 4.3686, huber: 3.9115, swd: 0.7201, ept: 67.8401
    Epoch [9/50], Val Losses: mse: 76.2166, mae: 5.2957, huber: 4.8292, swd: 0.8971, ept: 51.7939
    Epoch [9/50], Test Losses: mse: 70.9240, mae: 4.8410, huber: 4.3777, swd: 0.7282, ept: 51.4555
      Epoch 9 composite train-obj: 4.271617
            Val objective improved 5.2945 → 5.2778, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.7414, mae: 4.1570, huber: 3.7031, swd: 0.6606, ept: 78.8288
    Epoch [10/50], Val Losses: mse: 74.8282, mae: 5.2598, huber: 4.7945, swd: 0.9250, ept: 54.6103
    Epoch [10/50], Test Losses: mse: 68.2299, mae: 4.7489, huber: 4.2874, swd: 0.7502, ept: 55.1330
      Epoch 10 composite train-obj: 4.033337
            Val objective improved 5.2778 → 5.2570, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 52.4265, mae: 3.9955, huber: 3.5441, swd: 0.6734, ept: 85.7879
    Epoch [11/50], Val Losses: mse: 71.8059, mae: 5.1719, huber: 4.7070, swd: 0.9684, ept: 61.6900
    Epoch [11/50], Test Losses: mse: 65.7534, mae: 4.6716, huber: 4.2096, swd: 0.7930, ept: 58.8815
      Epoch 11 composite train-obj: 3.880776
            Val objective improved 5.2570 → 5.1912, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 49.2517, mae: 3.8496, huber: 3.3999, swd: 0.6672, ept: 91.0284
    Epoch [12/50], Val Losses: mse: 71.7992, mae: 5.1781, huber: 4.7148, swd: 1.0299, ept: 62.9338
    Epoch [12/50], Test Losses: mse: 63.8729, mae: 4.5736, huber: 4.1154, swd: 0.8049, ept: 60.1961
      Epoch 12 composite train-obj: 3.733533
            No improvement (5.2297), counter 1/5
    Epoch [13/50], Train Losses: mse: 45.9749, mae: 3.6900, huber: 3.2432, swd: 0.6401, ept: 95.7407
    Epoch [13/50], Val Losses: mse: 68.8611, mae: 5.0375, huber: 4.5763, swd: 1.0238, ept: 68.4846
    Epoch [13/50], Test Losses: mse: 61.5170, mae: 4.5019, huber: 4.0443, swd: 0.7804, ept: 66.4652
      Epoch 13 composite train-obj: 3.563291
            Val objective improved 5.1912 → 5.0882, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 43.0880, mae: 3.5629, huber: 3.1184, swd: 0.6498, ept: 100.4674
    Epoch [14/50], Val Losses: mse: 68.9589, mae: 5.1177, huber: 4.6559, swd: 1.0352, ept: 71.1381
    Epoch [14/50], Test Losses: mse: 62.0102, mae: 4.5409, huber: 4.0843, swd: 0.8034, ept: 63.4368
      Epoch 14 composite train-obj: 3.443322
            No improvement (5.1735), counter 1/5
    Epoch [15/50], Train Losses: mse: 40.2625, mae: 3.4451, huber: 3.0024, swd: 0.6445, ept: 103.5615
    Epoch [15/50], Val Losses: mse: 66.4631, mae: 4.9999, huber: 4.5391, swd: 1.0879, ept: 73.9113
    Epoch [15/50], Test Losses: mse: 58.1310, mae: 4.4112, huber: 3.9549, swd: 0.8626, ept: 68.8388
      Epoch 15 composite train-obj: 3.324660
            Val objective improved 5.0882 → 5.0831, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 38.0580, mae: 3.3420, huber: 2.9014, swd: 0.6216, ept: 107.8420
    Epoch [16/50], Val Losses: mse: 69.6776, mae: 5.1618, huber: 4.7013, swd: 1.3065, ept: 75.3799
    Epoch [16/50], Test Losses: mse: 58.0858, mae: 4.4053, huber: 3.9506, swd: 0.9713, ept: 71.3020
      Epoch 16 composite train-obj: 3.212229
            No improvement (5.3545), counter 1/5
    Epoch [17/50], Train Losses: mse: 36.4843, mae: 3.2809, huber: 2.8410, swd: 0.6238, ept: 110.9006
    Epoch [17/50], Val Losses: mse: 67.9605, mae: 5.1179, huber: 4.6561, swd: 1.0998, ept: 75.1748
    Epoch [17/50], Test Losses: mse: 58.3565, mae: 4.4384, huber: 3.9815, swd: 0.8397, ept: 71.5525
      Epoch 17 composite train-obj: 3.152911
            No improvement (5.2060), counter 2/5
    Epoch [18/50], Train Losses: mse: 34.2104, mae: 3.1622, huber: 2.7256, swd: 0.5952, ept: 115.1496
    Epoch [18/50], Val Losses: mse: 64.9134, mae: 4.9927, huber: 4.5336, swd: 1.3157, ept: 79.7682
    Epoch [18/50], Test Losses: mse: 54.9419, mae: 4.3120, huber: 3.8595, swd: 1.0024, ept: 74.7310
      Epoch 18 composite train-obj: 3.023216
            No improvement (5.1914), counter 3/5
    Epoch [19/50], Train Losses: mse: 32.3548, mae: 3.0935, huber: 2.6578, swd: 0.6010, ept: 117.9707
    Epoch [19/50], Val Losses: mse: 67.4657, mae: 5.0953, huber: 4.6365, swd: 1.2536, ept: 82.8552
    Epoch [19/50], Test Losses: mse: 56.1665, mae: 4.3658, huber: 3.9123, swd: 0.8534, ept: 74.5030
      Epoch 19 composite train-obj: 2.958329
            No improvement (5.2633), counter 4/5
    Epoch [20/50], Train Losses: mse: 29.9014, mae: 2.9588, huber: 2.5266, swd: 0.5821, ept: 124.5912
    Epoch [20/50], Val Losses: mse: 63.9036, mae: 4.9256, huber: 4.4696, swd: 1.1617, ept: 93.9364
    Epoch [20/50], Test Losses: mse: 53.4229, mae: 4.2063, huber: 3.7571, swd: 0.8243, ept: 84.9588
      Epoch 20 composite train-obj: 2.817690
            Val objective improved 5.0831 → 5.0504, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 28.3907, mae: 2.8969, huber: 2.4657, swd: 0.5812, ept: 129.6789
    Epoch [21/50], Val Losses: mse: 64.5214, mae: 4.9905, huber: 4.5326, swd: 1.1795, ept: 94.4440
    Epoch [21/50], Test Losses: mse: 53.9090, mae: 4.3010, huber: 3.8490, swd: 0.8903, ept: 84.2868
      Epoch 21 composite train-obj: 2.756268
            No improvement (5.1223), counter 1/5
    Epoch [22/50], Train Losses: mse: 27.3072, mae: 2.8461, huber: 2.4157, swd: 0.5838, ept: 134.5644
    Epoch [22/50], Val Losses: mse: 62.3506, mae: 4.9108, huber: 4.4548, swd: 1.4756, ept: 101.1176
    Epoch [22/50], Test Losses: mse: 51.1550, mae: 4.1517, huber: 3.7026, swd: 0.9356, ept: 91.1777
      Epoch 22 composite train-obj: 2.707624
            No improvement (5.1926), counter 2/5
    Epoch [23/50], Train Losses: mse: 25.8158, mae: 2.7667, huber: 2.3383, swd: 0.5656, ept: 141.1637
    Epoch [23/50], Val Losses: mse: 64.2683, mae: 5.0320, huber: 4.5737, swd: 1.3750, ept: 96.5657
    Epoch [23/50], Test Losses: mse: 54.0049, mae: 4.3378, huber: 3.8861, swd: 0.9680, ept: 88.7392
      Epoch 23 composite train-obj: 2.621067
            No improvement (5.2612), counter 3/5
    Epoch [24/50], Train Losses: mse: 24.9356, mae: 2.7179, huber: 2.2907, swd: 0.5626, ept: 146.8269
    Epoch [24/50], Val Losses: mse: 62.6696, mae: 4.8720, huber: 4.4175, swd: 1.3209, ept: 107.7556
    Epoch [24/50], Test Losses: mse: 50.5322, mae: 4.0756, huber: 3.6288, swd: 0.9294, ept: 100.6162
      Epoch 24 composite train-obj: 2.571996
            No improvement (5.0780), counter 4/5
    Epoch [25/50], Train Losses: mse: 23.4920, mae: 2.6235, huber: 2.1988, swd: 0.5615, ept: 155.8650
    Epoch [25/50], Val Losses: mse: 64.1366, mae: 4.9399, huber: 4.4848, swd: 1.3836, ept: 109.8147
    Epoch [25/50], Test Losses: mse: 52.1371, mae: 4.1720, huber: 3.7251, swd: 0.9704, ept: 100.8672
      Epoch 25 composite train-obj: 2.479605
    Epoch [25/50], Test Losses: mse: 53.4206, mae: 4.2063, huber: 3.7571, swd: 0.8242, ept: 84.9828
    Best round's Test MSE: 53.4229, MAE: 4.2063, SWD: 0.8243
    Best round's Validation MSE: 63.9036, MAE: 4.9256, SWD: 1.1617
    Best round's Test verification MSE : 53.4206, MAE: 4.2063, SWD: 0.8242
    Time taken: 116.73 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 242.6285, mae: 10.5423, huber: 10.0546, swd: 11.7825, ept: 4.1723
    Epoch [1/50], Val Losses: mse: 168.6110, mae: 8.4865, huber: 8.0010, swd: 1.2568, ept: 10.0845
    Epoch [1/50], Test Losses: mse: 165.3585, mae: 8.5065, huber: 8.0212, swd: 1.2365, ept: 8.3414
      Epoch 1 composite train-obj: 15.945873
            Val objective improved inf → 8.6294, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 142.7381, mae: 7.5111, huber: 7.0278, swd: 0.9618, ept: 13.7132
    Epoch [2/50], Val Losses: mse: 129.9025, mae: 7.2702, huber: 6.7878, swd: 1.3864, ept: 16.5125
    Epoch [2/50], Test Losses: mse: 126.8542, mae: 7.0989, huber: 6.6181, swd: 1.2638, ept: 15.4160
      Epoch 2 composite train-obj: 7.508703
            Val objective improved 8.6294 → 7.4810, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 107.8847, mae: 6.4151, huber: 5.9366, swd: 0.9344, ept: 24.1030
    Epoch [3/50], Val Losses: mse: 107.2363, mae: 6.5712, huber: 6.0911, swd: 1.0080, ept: 21.7722
    Epoch [3/50], Test Losses: mse: 104.3290, mae: 6.2770, huber: 5.8005, swd: 0.8201, ept: 20.1682
      Epoch 3 composite train-obj: 6.403791
            Val objective improved 7.4810 → 6.5951, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 89.8554, mae: 5.7717, huber: 5.2970, swd: 0.8588, ept: 41.2255
    Epoch [4/50], Val Losses: mse: 96.4928, mae: 6.0794, huber: 5.6020, swd: 0.9580, ept: 32.8625
    Epoch [4/50], Test Losses: mse: 93.6441, mae: 5.7765, huber: 5.3028, swd: 0.7840, ept: 28.2544
      Epoch 4 composite train-obj: 5.726414
            Val objective improved 6.5951 → 6.0810, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 80.8764, mae: 5.3196, huber: 4.8491, swd: 0.8188, ept: 57.5182
    Epoch [5/50], Val Losses: mse: 86.9292, mae: 5.7678, huber: 5.2934, swd: 1.4285, ept: 46.9103
    Epoch [5/50], Test Losses: mse: 85.8805, mae: 5.4398, huber: 4.9695, swd: 1.2281, ept: 39.7726
      Epoch 5 composite train-obj: 5.258521
            Val objective improved 6.0810 → 6.0077, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 73.9814, mae: 4.9564, huber: 4.4908, swd: 0.7678, ept: 77.4511
    Epoch [6/50], Val Losses: mse: 82.3513, mae: 5.4715, huber: 4.9998, swd: 0.9458, ept: 70.8578
    Epoch [6/50], Test Losses: mse: 79.6843, mae: 5.1511, huber: 4.6819, swd: 0.8626, ept: 66.5430
      Epoch 6 composite train-obj: 4.874713
            Val objective improved 6.0077 → 5.4727, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 68.5587, mae: 4.6897, huber: 4.2276, swd: 0.7542, ept: 103.5146
    Epoch [7/50], Val Losses: mse: 78.9806, mae: 5.3185, huber: 4.8497, swd: 1.0399, ept: 85.6289
    Epoch [7/50], Test Losses: mse: 75.6290, mae: 4.9496, huber: 4.4843, swd: 0.8411, ept: 85.9184
      Epoch 7 composite train-obj: 4.604697
            Val objective improved 5.4727 → 5.3697, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 63.6355, mae: 4.4517, huber: 3.9934, swd: 0.7062, ept: 121.4790
    Epoch [8/50], Val Losses: mse: 75.1268, mae: 5.1813, huber: 4.7143, swd: 0.8999, ept: 98.7731
    Epoch [8/50], Test Losses: mse: 72.7051, mae: 4.8048, huber: 4.3410, swd: 0.7132, ept: 96.5871
      Epoch 8 composite train-obj: 4.346513
            Val objective improved 5.3697 → 5.1643, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 59.5794, mae: 4.2527, huber: 3.7978, swd: 0.6928, ept: 133.8260
    Epoch [9/50], Val Losses: mse: 73.8794, mae: 5.1306, huber: 4.6647, swd: 0.8852, ept: 106.6258
    Epoch [9/50], Test Losses: mse: 68.6352, mae: 4.6380, huber: 4.1764, swd: 0.6824, ept: 105.7660
      Epoch 9 composite train-obj: 4.144200
            Val objective improved 5.1643 → 5.1073, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 55.9056, mae: 4.0797, huber: 3.6274, swd: 0.6789, ept: 144.9459
    Epoch [10/50], Val Losses: mse: 72.4418, mae: 5.1216, huber: 4.6553, swd: 0.9417, ept: 108.4117
    Epoch [10/50], Test Losses: mse: 65.9110, mae: 4.5859, huber: 4.1237, swd: 0.7186, ept: 112.9054
      Epoch 10 composite train-obj: 3.966861
            No improvement (5.1262), counter 1/5
    Epoch [11/50], Train Losses: mse: 52.7687, mae: 3.9185, huber: 3.4694, swd: 0.6753, ept: 154.9463
    Epoch [11/50], Val Losses: mse: 70.2277, mae: 5.0155, huber: 4.5529, swd: 0.9195, ept: 115.1844
    Epoch [11/50], Test Losses: mse: 64.2228, mae: 4.4648, huber: 4.0063, swd: 0.6862, ept: 123.3550
      Epoch 11 composite train-obj: 3.807079
            Val objective improved 5.1073 → 5.0126, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 50.1346, mae: 3.8057, huber: 3.3584, swd: 0.6510, ept: 160.7492
    Epoch [12/50], Val Losses: mse: 68.6143, mae: 4.9495, huber: 4.4875, swd: 0.9158, ept: 116.3517
    Epoch [12/50], Test Losses: mse: 62.6087, mae: 4.4083, huber: 3.9505, swd: 0.6554, ept: 125.5239
      Epoch 12 composite train-obj: 3.683852
            Val objective improved 5.0126 → 4.9454, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 47.2987, mae: 3.6697, huber: 3.2254, swd: 0.6570, ept: 166.7241
    Epoch [13/50], Val Losses: mse: 66.7955, mae: 4.9161, huber: 4.4550, swd: 0.9542, ept: 120.4479
    Epoch [13/50], Test Losses: mse: 60.3046, mae: 4.3777, huber: 3.9196, swd: 0.7440, ept: 128.4383
      Epoch 13 composite train-obj: 3.553876
            Val objective improved 4.9454 → 4.9321, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 45.0072, mae: 3.5628, huber: 3.1204, swd: 0.6347, ept: 171.5173
    Epoch [14/50], Val Losses: mse: 65.5377, mae: 4.8388, huber: 4.3799, swd: 0.9799, ept: 125.5393
    Epoch [14/50], Test Losses: mse: 58.8453, mae: 4.2809, huber: 3.8269, swd: 0.7596, ept: 135.3304
      Epoch 14 composite train-obj: 3.437740
            Val objective improved 4.9321 → 4.8698, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 42.1037, mae: 3.4234, huber: 2.9838, swd: 0.6204, ept: 176.9095
    Epoch [15/50], Val Losses: mse: 66.2361, mae: 4.9049, huber: 4.4448, swd: 0.8960, ept: 124.9670
    Epoch [15/50], Test Losses: mse: 58.2206, mae: 4.2945, huber: 3.8396, swd: 0.7177, ept: 135.2711
      Epoch 15 composite train-obj: 3.294000
            No improvement (4.8928), counter 1/5
    Epoch [16/50], Train Losses: mse: 40.2433, mae: 3.3624, huber: 2.9233, swd: 0.6251, ept: 178.9931
    Epoch [16/50], Val Losses: mse: 63.0020, mae: 4.8221, huber: 4.3627, swd: 1.0267, ept: 124.1202
    Epoch [16/50], Test Losses: mse: 56.4298, mae: 4.2605, huber: 3.8067, swd: 0.8235, ept: 132.8902
      Epoch 16 composite train-obj: 3.235847
            No improvement (4.8760), counter 2/5
    Epoch [17/50], Train Losses: mse: 38.0602, mae: 3.2645, huber: 2.8269, swd: 0.6412, ept: 182.9127
    Epoch [17/50], Val Losses: mse: 62.7860, mae: 4.7680, huber: 4.3099, swd: 0.9533, ept: 130.7263
    Epoch [17/50], Test Losses: mse: 55.8518, mae: 4.2148, huber: 3.7615, swd: 0.7779, ept: 142.2066
      Epoch 17 composite train-obj: 3.147499
            Val objective improved 4.8698 → 4.7866, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 35.5898, mae: 3.1389, huber: 2.7046, swd: 0.5897, ept: 188.6536
    Epoch [18/50], Val Losses: mse: 64.2807, mae: 4.8613, huber: 4.4046, swd: 1.0303, ept: 133.6011
    Epoch [18/50], Test Losses: mse: 53.6002, mae: 4.1260, huber: 3.6751, swd: 0.7880, ept: 144.5506
      Epoch 18 composite train-obj: 2.999410
            No improvement (4.9197), counter 1/5
    Epoch [19/50], Train Losses: mse: 34.2364, mae: 3.1161, huber: 2.6807, swd: 0.6065, ept: 189.9175
    Epoch [19/50], Val Losses: mse: 60.9158, mae: 4.7381, huber: 4.2821, swd: 1.0495, ept: 139.5507
    Epoch [19/50], Test Losses: mse: 51.6113, mae: 4.0704, huber: 3.6207, swd: 0.8139, ept: 152.1301
      Epoch 19 composite train-obj: 2.983978
            No improvement (4.8068), counter 2/5
    Epoch [20/50], Train Losses: mse: 31.8243, mae: 3.0047, huber: 2.5719, swd: 0.5998, ept: 194.8571
    Epoch [20/50], Val Losses: mse: 59.5152, mae: 4.6787, huber: 4.2229, swd: 1.0871, ept: 135.6235
    Epoch [20/50], Test Losses: mse: 50.8156, mae: 4.0544, huber: 3.6049, swd: 0.8398, ept: 156.0523
      Epoch 20 composite train-obj: 2.871795
            Val objective improved 4.7866 → 4.7664, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 29.7260, mae: 2.9135, huber: 2.4825, swd: 0.5917, ept: 199.4727
    Epoch [21/50], Val Losses: mse: 58.2820, mae: 4.6218, huber: 4.1677, swd: 1.2163, ept: 141.1649
    Epoch [21/50], Test Losses: mse: 49.6756, mae: 4.0352, huber: 3.5866, swd: 0.9362, ept: 157.2399
      Epoch 21 composite train-obj: 2.778299
            No improvement (4.7758), counter 1/5
    Epoch [22/50], Train Losses: mse: 27.5800, mae: 2.8037, huber: 2.3752, swd: 0.5755, ept: 204.7206
    Epoch [22/50], Val Losses: mse: 61.7154, mae: 4.8041, huber: 4.3485, swd: 1.0495, ept: 134.9894
    Epoch [22/50], Test Losses: mse: 50.0973, mae: 4.0745, huber: 3.6251, swd: 0.8319, ept: 151.3546
      Epoch 22 composite train-obj: 2.662947
            No improvement (4.8732), counter 2/5
    Epoch [23/50], Train Losses: mse: 25.9771, mae: 2.7421, huber: 2.3148, swd: 0.5742, ept: 207.9636
    Epoch [23/50], Val Losses: mse: 57.8299, mae: 4.6309, huber: 4.1771, swd: 1.2841, ept: 143.2093
    Epoch [23/50], Test Losses: mse: 47.5368, mae: 3.9811, huber: 3.5331, swd: 0.9227, ept: 164.2104
      Epoch 23 composite train-obj: 2.601923
            No improvement (4.8191), counter 3/5
    Epoch [24/50], Train Losses: mse: 24.8196, mae: 2.6936, huber: 2.2671, swd: 0.5670, ept: 211.2248
    Epoch [24/50], Val Losses: mse: 60.2629, mae: 4.7522, huber: 4.2980, swd: 1.1314, ept: 144.1534
    Epoch [24/50], Test Losses: mse: 48.9643, mae: 4.0209, huber: 3.5741, swd: 0.8189, ept: 165.6785
      Epoch 24 composite train-obj: 2.550608
            No improvement (4.8637), counter 4/5
    Epoch [25/50], Train Losses: mse: 24.4229, mae: 2.7033, huber: 2.2752, swd: 0.5868, ept: 211.8761
    Epoch [25/50], Val Losses: mse: 56.8229, mae: 4.5431, huber: 4.0928, swd: 1.3660, ept: 153.2840
    Epoch [25/50], Test Losses: mse: 44.0442, mae: 3.8183, huber: 3.3744, swd: 0.9659, ept: 167.7484
      Epoch 25 composite train-obj: 2.568608
    Epoch [25/50], Test Losses: mse: 50.8240, mae: 4.0547, huber: 3.6052, swd: 0.8396, ept: 156.0278
    Best round's Test MSE: 50.8156, MAE: 4.0544, SWD: 0.8398
    Best round's Validation MSE: 59.5152, MAE: 4.6787, SWD: 1.0871
    Best round's Test verification MSE : 50.8240, MAE: 4.0547, SWD: 0.8396
    Time taken: 114.68 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 239.6302, mae: 10.5411, huber: 10.0533, swd: 14.1173, ept: 4.0781
    Epoch [1/50], Val Losses: mse: 170.6099, mae: 8.4407, huber: 7.9551, swd: 1.1190, ept: 8.0352
    Epoch [1/50], Test Losses: mse: 166.7617, mae: 8.3774, huber: 7.8923, swd: 1.0424, ept: 7.2482
      Epoch 1 composite train-obj: 17.111966
            Val objective improved inf → 8.5146, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 144.4483, mae: 7.4775, huber: 6.9950, swd: 1.0355, ept: 12.9247
    Epoch [2/50], Val Losses: mse: 132.6160, mae: 7.3068, huber: 6.8239, swd: 0.9919, ept: 14.2835
    Epoch [2/50], Test Losses: mse: 130.9303, mae: 7.1467, huber: 6.6655, swd: 0.8323, ept: 12.8519
      Epoch 2 composite train-obj: 7.512750
            Val objective improved 8.5146 → 7.3199, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 107.1296, mae: 6.3592, huber: 5.8811, swd: 0.9172, ept: 29.6505
    Epoch [3/50], Val Losses: mse: 106.7124, mae: 6.4602, huber: 5.9814, swd: 1.0310, ept: 20.9139
    Epoch [3/50], Test Losses: mse: 104.8243, mae: 6.2347, huber: 5.7583, swd: 0.8483, ept: 18.1845
      Epoch 3 composite train-obj: 6.339704
            Val objective improved 7.3199 → 6.4969, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 91.7771, mae: 5.7642, huber: 5.2894, swd: 0.8572, ept: 36.7122
    Epoch [4/50], Val Losses: mse: 99.1197, mae: 6.1035, huber: 5.6265, swd: 0.9802, ept: 28.4009
    Epoch [4/50], Test Losses: mse: 97.8483, mae: 5.8291, huber: 5.3548, swd: 0.7724, ept: 23.3940
      Epoch 4 composite train-obj: 5.718058
            Val objective improved 6.4969 → 6.1166, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 84.2388, mae: 5.3589, huber: 4.8882, swd: 0.8122, ept: 45.5831
    Epoch [5/50], Val Losses: mse: 93.4648, mae: 5.8513, huber: 5.3770, swd: 1.0080, ept: 37.6771
    Epoch [5/50], Test Losses: mse: 91.6829, mae: 5.4956, huber: 5.0248, swd: 0.7747, ept: 29.7555
      Epoch 5 composite train-obj: 5.294295
            Val objective improved 6.1166 → 5.8810, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 78.0125, mae: 5.0300, huber: 4.5635, swd: 0.7760, ept: 58.7077
    Epoch [6/50], Val Losses: mse: 89.4615, mae: 5.7329, huber: 5.2601, swd: 0.9895, ept: 49.6660
    Epoch [6/50], Test Losses: mse: 87.0114, mae: 5.3122, huber: 4.8427, swd: 0.7808, ept: 42.5582
      Epoch 6 composite train-obj: 4.951458
            Val objective improved 5.8810 → 5.7549, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 72.3737, mae: 4.7440, huber: 4.2813, swd: 0.7810, ept: 76.7541
    Epoch [7/50], Val Losses: mse: 85.4087, mae: 5.5346, huber: 5.0635, swd: 1.0153, ept: 67.0210
    Epoch [7/50], Test Losses: mse: 83.0629, mae: 5.1208, huber: 4.6529, swd: 0.7835, ept: 64.7683
      Epoch 7 composite train-obj: 4.671841
            Val objective improved 5.7549 → 5.5712, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 67.1624, mae: 4.4945, huber: 4.0352, swd: 0.7483, ept: 101.9862
    Epoch [8/50], Val Losses: mse: 81.8022, mae: 5.3966, huber: 4.9279, swd: 1.0858, ept: 76.1615
    Epoch [8/50], Test Losses: mse: 79.1296, mae: 4.9796, huber: 4.5141, swd: 0.8690, ept: 74.4072
      Epoch 8 composite train-obj: 4.409331
            Val objective improved 5.5712 → 5.4708, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 62.4268, mae: 4.2789, huber: 3.8224, swd: 0.7406, ept: 117.9718
    Epoch [9/50], Val Losses: mse: 80.3823, mae: 5.3430, huber: 4.8755, swd: 1.0250, ept: 92.3529
    Epoch [9/50], Test Losses: mse: 76.1174, mae: 4.8332, huber: 4.3703, swd: 0.8132, ept: 91.0883
      Epoch 9 composite train-obj: 4.192725
            Val objective improved 5.4708 → 5.3880, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 58.1366, mae: 4.0734, huber: 3.6204, swd: 0.7299, ept: 132.1318
    Epoch [10/50], Val Losses: mse: 77.4252, mae: 5.2505, huber: 4.7849, swd: 1.0881, ept: 99.6584
    Epoch [10/50], Test Losses: mse: 72.1575, mae: 4.7223, huber: 4.2604, swd: 0.8184, ept: 98.6794
      Epoch 10 composite train-obj: 3.985393
            Val objective improved 5.3880 → 5.3290, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 53.9898, mae: 3.8915, huber: 3.4414, swd: 0.7141, ept: 144.2636
    Epoch [11/50], Val Losses: mse: 79.2083, mae: 5.3428, huber: 4.8770, swd: 1.1439, ept: 101.3119
    Epoch [11/50], Test Losses: mse: 71.8625, mae: 4.7250, huber: 4.2639, swd: 0.8867, ept: 99.7854
      Epoch 11 composite train-obj: 3.798469
            No improvement (5.4490), counter 1/5
    Epoch [12/50], Train Losses: mse: 50.8444, mae: 3.7565, huber: 3.3084, swd: 0.7085, ept: 152.8569
    Epoch [12/50], Val Losses: mse: 72.8334, mae: 5.1323, huber: 4.6684, swd: 1.1854, ept: 111.8948
    Epoch [12/50], Test Losses: mse: 66.2204, mae: 4.5690, huber: 4.1084, swd: 0.8934, ept: 114.0923
      Epoch 12 composite train-obj: 3.662689
            Val objective improved 5.3290 → 5.2610, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 47.5262, mae: 3.6160, huber: 3.1701, swd: 0.7080, ept: 160.3117
    Epoch [13/50], Val Losses: mse: 71.7520, mae: 5.0219, huber: 4.5609, swd: 1.1464, ept: 114.6842
    Epoch [13/50], Test Losses: mse: 66.1203, mae: 4.5148, huber: 4.0570, swd: 0.8382, ept: 117.4396
      Epoch 13 composite train-obj: 3.524121
            Val objective improved 5.2610 → 5.1341, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 45.1842, mae: 3.5025, huber: 3.0592, swd: 0.6723, ept: 167.1329
    Epoch [14/50], Val Losses: mse: 73.6581, mae: 5.1934, huber: 4.7292, swd: 1.1842, ept: 115.7806
    Epoch [14/50], Test Losses: mse: 64.2777, mae: 4.5203, huber: 4.0613, swd: 0.8942, ept: 119.6676
      Epoch 14 composite train-obj: 3.395387
            No improvement (5.3213), counter 1/5
    Epoch [15/50], Train Losses: mse: 42.4908, mae: 3.3899, huber: 2.9484, swd: 0.6742, ept: 172.5908
    Epoch [15/50], Val Losses: mse: 71.3403, mae: 5.0547, huber: 4.5940, swd: 1.1903, ept: 122.6436
    Epoch [15/50], Test Losses: mse: 61.7384, mae: 4.3618, huber: 3.9065, swd: 0.9162, ept: 125.8875
      Epoch 15 composite train-obj: 3.285536
            No improvement (5.1891), counter 2/5
    Epoch [16/50], Train Losses: mse: 39.8330, mae: 3.2664, huber: 2.8279, swd: 0.6600, ept: 178.3434
    Epoch [16/50], Val Losses: mse: 70.9194, mae: 5.0071, huber: 4.5484, swd: 1.2233, ept: 130.2992
    Epoch [16/50], Test Losses: mse: 61.0421, mae: 4.3032, huber: 3.8497, swd: 0.8838, ept: 132.8014
      Epoch 16 composite train-obj: 3.157830
            No improvement (5.1600), counter 3/5
    Epoch [17/50], Train Losses: mse: 37.8122, mae: 3.1835, huber: 2.7458, swd: 0.6627, ept: 183.3732
    Epoch [17/50], Val Losses: mse: 69.5268, mae: 4.9496, huber: 4.4929, swd: 1.2479, ept: 135.1841
    Epoch [17/50], Test Losses: mse: 59.9268, mae: 4.2802, huber: 3.8288, swd: 0.9230, ept: 140.9726
      Epoch 17 composite train-obj: 3.077101
            Val objective improved 5.1341 → 5.1169, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 35.1623, mae: 3.0658, huber: 2.6309, swd: 0.6476, ept: 189.5481
    Epoch [18/50], Val Losses: mse: 67.3169, mae: 4.9485, huber: 4.4896, swd: 1.2945, ept: 132.6986
    Epoch [18/50], Test Losses: mse: 58.1620, mae: 4.2996, huber: 3.8455, swd: 0.9678, ept: 139.6904
      Epoch 18 composite train-obj: 2.954704
            No improvement (5.1369), counter 1/5
    Epoch [19/50], Train Losses: mse: 33.3709, mae: 2.9927, huber: 2.5592, swd: 0.6496, ept: 194.2380
    Epoch [19/50], Val Losses: mse: 67.2986, mae: 4.9488, huber: 4.4909, swd: 1.3505, ept: 134.1531
    Epoch [19/50], Test Losses: mse: 56.7802, mae: 4.2424, huber: 3.7896, swd: 0.9480, ept: 147.3467
      Epoch 19 composite train-obj: 2.884033
            No improvement (5.1662), counter 2/5
    Epoch [20/50], Train Losses: mse: 32.1373, mae: 2.9538, huber: 2.5205, swd: 0.6535, ept: 196.7460
    Epoch [20/50], Val Losses: mse: 67.0136, mae: 4.9446, huber: 4.4875, swd: 1.3772, ept: 136.5776
    Epoch [20/50], Test Losses: mse: 55.3128, mae: 4.2176, huber: 3.7664, swd: 1.0620, ept: 145.3491
      Epoch 20 composite train-obj: 2.847274
            No improvement (5.1761), counter 3/5
    Epoch [21/50], Train Losses: mse: 30.6721, mae: 2.9073, huber: 2.4744, swd: 0.6575, ept: 199.4457
    Epoch [21/50], Val Losses: mse: 66.0646, mae: 4.8928, huber: 4.4376, swd: 1.3786, ept: 137.5384
    Epoch [21/50], Test Losses: mse: 53.7080, mae: 4.1202, huber: 3.6715, swd: 1.0098, ept: 150.5743
      Epoch 21 composite train-obj: 2.803150
            No improvement (5.1269), counter 4/5
    Epoch [22/50], Train Losses: mse: 27.7745, mae: 2.7334, huber: 2.3059, swd: 0.6058, ept: 208.4270
    Epoch [22/50], Val Losses: mse: 65.6218, mae: 4.8863, huber: 4.4321, swd: 1.4050, ept: 144.4733
    Epoch [22/50], Test Losses: mse: 51.4241, mae: 4.0384, huber: 3.5908, swd: 1.0106, ept: 157.4682
      Epoch 22 composite train-obj: 2.608836
    Epoch [22/50], Test Losses: mse: 59.9202, mae: 4.2800, huber: 3.8287, swd: 0.9231, ept: 141.0177
    Best round's Test MSE: 59.9268, MAE: 4.2802, SWD: 0.9230
    Best round's Validation MSE: 69.5268, MAE: 4.9496, SWD: 1.2479
    Best round's Test verification MSE : 59.9202, MAE: 4.2800, SWD: 0.9231
    Time taken: 99.99 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_2141)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 54.7218 ± 3.8313
      mae: 4.1803 ± 0.0940
      huber: 3.7303 ± 0.0934
      swd: 0.8624 ± 0.0434
      ept: 127.3279 ± 30.5854
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 64.3152 ± 4.0976
      mae: 4.8513 ± 0.1224
      huber: 4.3951 ± 0.1222
      swd: 1.1656 ± 0.0657
      ept: 121.5813 ± 19.5487
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 331.50 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_2141
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE


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
    mixing_strategy='convex', ### Here
    loss_backward_weights = [1.0, 0.0, 0.0, 0.0, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.0, 0.0],
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

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.2142, mae: 6.7053, huber: 6.2235, swd: 44.6189, ept: 27.1548
    Epoch [1/50], Val Losses: mse: 61.2947, mae: 6.0596, huber: 5.5797, swd: 31.3272, ept: 29.1181
    Epoch [1/50], Test Losses: mse: 57.7297, mae: 5.7331, huber: 5.2563, swd: 31.5414, ept: 29.0298
      Epoch 1 composite train-obj: 76.214202
            Val objective improved inf → 61.2947, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.4411, mae: 5.1071, huber: 4.6340, swd: 18.5370, ept: 75.7770
    Epoch [2/50], Val Losses: mse: 49.5617, mae: 5.1823, huber: 4.7095, swd: 15.1372, ept: 74.3711
    Epoch [2/50], Test Losses: mse: 45.8884, mae: 4.9089, huber: 4.4379, swd: 15.8691, ept: 76.0302
      Epoch 2 composite train-obj: 48.441116
            Val objective improved 61.2947 → 49.5617, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.8704, mae: 4.3689, huber: 3.9036, swd: 9.4058, ept: 125.2380
    Epoch [3/50], Val Losses: mse: 46.9676, mae: 4.8914, huber: 4.4227, swd: 10.5067, ept: 96.9904
    Epoch [3/50], Test Losses: mse: 40.0464, mae: 4.5096, huber: 4.0418, swd: 10.1229, ept: 102.0774
      Epoch 3 composite train-obj: 38.870397
            Val objective improved 49.5617 → 46.9676, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.0387, mae: 3.9833, huber: 3.5229, swd: 6.4750, ept: 152.0327
    Epoch [4/50], Val Losses: mse: 45.6485, mae: 4.7006, huber: 4.2350, swd: 8.5291, ept: 115.3081
    Epoch [4/50], Test Losses: mse: 38.1809, mae: 4.2888, huber: 3.8241, swd: 8.1005, ept: 116.9749
      Epoch 4 composite train-obj: 34.038691
            Val objective improved 46.9676 → 45.6485, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.1765, mae: 3.6815, huber: 3.2247, swd: 4.9420, ept: 168.8580
    Epoch [5/50], Val Losses: mse: 46.6544, mae: 4.6889, huber: 4.2234, swd: 6.1852, ept: 119.3921
    Epoch [5/50], Test Losses: mse: 37.6156, mae: 4.1798, huber: 3.7178, swd: 5.5884, ept: 129.6609
      Epoch 5 composite train-obj: 30.176508
            No improvement (46.6544), counter 1/5
    Epoch [6/50], Train Losses: mse: 27.4032, mae: 3.4679, huber: 3.0137, swd: 4.0240, ept: 179.4772
    Epoch [6/50], Val Losses: mse: 46.4081, mae: 4.6526, huber: 4.1883, swd: 5.9784, ept: 133.0380
    Epoch [6/50], Test Losses: mse: 37.7977, mae: 4.1449, huber: 3.6837, swd: 4.8822, ept: 144.2130
      Epoch 6 composite train-obj: 27.403235
            No improvement (46.4081), counter 2/5
    Epoch [7/50], Train Losses: mse: 25.3322, mae: 3.3126, huber: 2.8604, swd: 3.5514, ept: 185.5417
    Epoch [7/50], Val Losses: mse: 44.3827, mae: 4.4555, huber: 3.9928, swd: 4.8057, ept: 134.3112
    Epoch [7/50], Test Losses: mse: 35.2933, mae: 3.9287, huber: 3.4697, swd: 4.0312, ept: 147.1782
      Epoch 7 composite train-obj: 25.332218
            Val objective improved 45.6485 → 44.3827, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 23.0871, mae: 3.1285, huber: 2.6792, swd: 2.9643, ept: 195.2362
    Epoch [8/50], Val Losses: mse: 45.0380, mae: 4.4598, huber: 3.9979, swd: 4.9973, ept: 140.2786
    Epoch [8/50], Test Losses: mse: 36.7443, mae: 3.9775, huber: 3.5189, swd: 4.0321, ept: 150.8209
      Epoch 8 composite train-obj: 23.087090
            No improvement (45.0380), counter 1/5
    Epoch [9/50], Train Losses: mse: 21.1895, mae: 2.9771, huber: 2.5303, swd: 2.5225, ept: 202.5540
    Epoch [9/50], Val Losses: mse: 43.5528, mae: 4.2979, huber: 3.8396, swd: 4.8322, ept: 150.7202
    Epoch [9/50], Test Losses: mse: 34.0978, mae: 3.7231, huber: 3.2690, swd: 3.4984, ept: 165.2340
      Epoch 9 composite train-obj: 21.189523
            Val objective improved 44.3827 → 43.5528, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 19.4161, mae: 2.8223, huber: 2.3785, swd: 2.1835, ept: 212.2987
    Epoch [10/50], Val Losses: mse: 51.9912, mae: 4.6669, huber: 4.2068, swd: 3.4625, ept: 145.9485
    Epoch [10/50], Test Losses: mse: 36.6551, mae: 3.8546, huber: 3.3989, swd: 2.5473, ept: 161.3962
      Epoch 10 composite train-obj: 19.416149
            No improvement (51.9912), counter 1/5
    Epoch [11/50], Train Losses: mse: 18.8180, mae: 2.7648, huber: 2.3223, swd: 2.0872, ept: 214.6863
    Epoch [11/50], Val Losses: mse: 48.5753, mae: 4.4642, huber: 4.0061, swd: 3.6040, ept: 152.5100
    Epoch [11/50], Test Losses: mse: 35.8287, mae: 3.7717, huber: 3.3179, swd: 2.6724, ept: 166.5419
      Epoch 11 composite train-obj: 18.818007
            No improvement (48.5753), counter 2/5
    Epoch [12/50], Train Losses: mse: 16.9837, mae: 2.6043, huber: 2.1652, swd: 1.7993, ept: 223.8631
    Epoch [12/50], Val Losses: mse: 43.4645, mae: 4.2256, huber: 3.7684, swd: 3.5779, ept: 158.2252
    Epoch [12/50], Test Losses: mse: 35.7468, mae: 3.7267, huber: 3.2739, swd: 2.3779, ept: 176.1261
      Epoch 12 composite train-obj: 16.983741
            Val objective improved 43.5528 → 43.4645, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 16.9834, mae: 2.5938, huber: 2.1552, swd: 1.8691, ept: 225.1490
    Epoch [13/50], Val Losses: mse: 42.2932, mae: 4.1241, huber: 3.6689, swd: 3.0001, ept: 163.1594
    Epoch [13/50], Test Losses: mse: 32.8866, mae: 3.5733, huber: 3.1221, swd: 2.1119, ept: 177.5932
      Epoch 13 composite train-obj: 16.983394
            Val objective improved 43.4645 → 42.2932, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 15.2614, mae: 2.4411, huber: 2.0061, swd: 1.5633, ept: 234.1064
    Epoch [14/50], Val Losses: mse: 42.0044, mae: 4.0591, huber: 3.6061, swd: 3.0590, ept: 160.0764
    Epoch [14/50], Test Losses: mse: 33.1971, mae: 3.5391, huber: 3.0906, swd: 2.5199, ept: 180.2070
      Epoch 14 composite train-obj: 15.261414
            Val objective improved 42.2932 → 42.0044, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 15.3621, mae: 2.4399, huber: 2.0054, swd: 1.6071, ept: 234.2420
    Epoch [15/50], Val Losses: mse: 40.7723, mae: 3.9322, huber: 3.4821, swd: 2.9424, ept: 168.0249
    Epoch [15/50], Test Losses: mse: 32.0293, mae: 3.4052, huber: 2.9607, swd: 2.1072, ept: 190.8351
      Epoch 15 composite train-obj: 15.362121
            Val objective improved 42.0044 → 40.7723, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 14.8617, mae: 2.3905, huber: 1.9575, swd: 1.5347, ept: 237.3587
    Epoch [16/50], Val Losses: mse: 41.4000, mae: 3.9791, huber: 3.5282, swd: 3.5127, ept: 164.3429
    Epoch [16/50], Test Losses: mse: 33.3032, mae: 3.4708, huber: 3.0244, swd: 2.7246, ept: 187.6371
      Epoch 16 composite train-obj: 14.861673
            No improvement (41.4000), counter 1/5
    Epoch [17/50], Train Losses: mse: 13.6369, mae: 2.2753, huber: 1.8458, swd: 1.3975, ept: 244.1551
    Epoch [17/50], Val Losses: mse: 41.0091, mae: 3.9043, huber: 3.4558, swd: 2.9431, ept: 169.9911
    Epoch [17/50], Test Losses: mse: 30.9020, mae: 3.3180, huber: 2.8754, swd: 1.9978, ept: 197.4560
      Epoch 17 composite train-obj: 13.636934
            No improvement (41.0091), counter 2/5
    Epoch [18/50], Train Losses: mse: 13.6375, mae: 2.2751, huber: 1.8451, swd: 1.3749, ept: 244.5666
    Epoch [18/50], Val Losses: mse: 40.0423, mae: 3.8524, huber: 3.4046, swd: 2.7487, ept: 176.0332
    Epoch [18/50], Test Losses: mse: 31.7826, mae: 3.3386, huber: 2.8961, swd: 1.9022, ept: 197.1425
      Epoch 18 composite train-obj: 13.637523
            Val objective improved 40.7723 → 40.0423, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 12.3376, mae: 2.1446, huber: 1.7193, swd: 1.1830, ept: 252.3001
    Epoch [19/50], Val Losses: mse: 43.0030, mae: 4.0214, huber: 3.5722, swd: 2.9152, ept: 170.1089
    Epoch [19/50], Test Losses: mse: 33.9297, mae: 3.4491, huber: 3.0053, swd: 1.9302, ept: 194.2277
      Epoch 19 composite train-obj: 12.337616
            No improvement (43.0030), counter 1/5
    Epoch [20/50], Train Losses: mse: 13.3110, mae: 2.2241, huber: 1.7966, swd: 1.3514, ept: 248.1911
    Epoch [20/50], Val Losses: mse: 47.6426, mae: 4.2990, huber: 3.8462, swd: 2.5945, ept: 163.2008
    Epoch [20/50], Test Losses: mse: 34.6760, mae: 3.5795, huber: 3.1323, swd: 2.1703, ept: 182.6561
      Epoch 20 composite train-obj: 13.310980
            No improvement (47.6426), counter 2/5
    Epoch [21/50], Train Losses: mse: 12.4537, mae: 2.1489, huber: 1.7236, swd: 1.2433, ept: 253.0080
    Epoch [21/50], Val Losses: mse: 43.6050, mae: 4.0667, huber: 3.6156, swd: 3.2276, ept: 168.5072
    Epoch [21/50], Test Losses: mse: 34.8274, mae: 3.5320, huber: 3.0866, swd: 2.3406, ept: 187.1910
      Epoch 21 composite train-obj: 12.453682
            No improvement (43.6050), counter 3/5
    Epoch [22/50], Train Losses: mse: 11.9722, mae: 2.0997, huber: 1.6757, swd: 1.1840, ept: 255.8623
    Epoch [22/50], Val Losses: mse: 40.1346, mae: 3.8008, huber: 3.3556, swd: 2.1179, ept: 181.5791
    Epoch [22/50], Test Losses: mse: 30.5646, mae: 3.2178, huber: 2.7797, swd: 1.6476, ept: 208.9172
      Epoch 22 composite train-obj: 11.972170
            No improvement (40.1346), counter 4/5
    Epoch [23/50], Train Losses: mse: 11.0650, mae: 2.0054, huber: 1.5850, swd: 1.0397, ept: 261.5217
    Epoch [23/50], Val Losses: mse: 39.8176, mae: 3.8523, huber: 3.4033, swd: 2.7736, ept: 173.8763
    Epoch [23/50], Test Losses: mse: 32.4688, mae: 3.4076, huber: 2.9641, swd: 2.5008, ept: 192.6465
      Epoch 23 composite train-obj: 11.065002
            Val objective improved 40.0423 → 39.8176, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 11.2575, mae: 2.0241, huber: 1.6030, swd: 1.0830, ept: 260.3265
    Epoch [24/50], Val Losses: mse: 39.8935, mae: 3.7929, huber: 3.3486, swd: 2.8971, ept: 181.0127
    Epoch [24/50], Test Losses: mse: 29.4395, mae: 3.1228, huber: 2.6867, swd: 1.8148, ept: 212.3238
      Epoch 24 composite train-obj: 11.257518
            No improvement (39.8935), counter 1/5
    Epoch [25/50], Train Losses: mse: 11.0197, mae: 1.9998, huber: 1.5796, swd: 1.0544, ept: 262.3675
    Epoch [25/50], Val Losses: mse: 39.6006, mae: 3.8031, huber: 3.3578, swd: 2.3966, ept: 181.5820
    Epoch [25/50], Test Losses: mse: 31.1428, mae: 3.2084, huber: 2.7714, swd: 1.6837, ept: 211.8266
      Epoch 25 composite train-obj: 11.019691
            Val objective improved 39.8176 → 39.6006, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 10.8349, mae: 1.9702, huber: 1.5512, swd: 1.0134, ept: 263.1232
    Epoch [26/50], Val Losses: mse: 40.1203, mae: 3.8697, huber: 3.4203, swd: 2.6726, ept: 177.7516
    Epoch [26/50], Test Losses: mse: 33.8891, mae: 3.4578, huber: 3.0145, swd: 2.0460, ept: 193.9234
      Epoch 26 composite train-obj: 10.834892
            No improvement (40.1203), counter 1/5
    Epoch [27/50], Train Losses: mse: 11.2950, mae: 2.0236, huber: 1.6024, swd: 1.0909, ept: 261.1597
    Epoch [27/50], Val Losses: mse: 40.9544, mae: 3.7905, huber: 3.3467, swd: 2.1156, ept: 188.7249
    Epoch [27/50], Test Losses: mse: 29.9358, mae: 3.1395, huber: 2.7032, swd: 1.7110, ept: 212.5145
      Epoch 27 composite train-obj: 11.295040
            No improvement (40.9544), counter 2/5
    Epoch [28/50], Train Losses: mse: 11.3358, mae: 2.0136, huber: 1.5935, swd: 1.1263, ept: 261.9646
    Epoch [28/50], Val Losses: mse: 39.9059, mae: 3.8509, huber: 3.4048, swd: 2.7852, ept: 177.3206
    Epoch [28/50], Test Losses: mse: 31.6638, mae: 3.2937, huber: 2.8551, swd: 1.8489, ept: 200.8995
      Epoch 28 composite train-obj: 11.335774
            No improvement (39.9059), counter 3/5
    Epoch [29/50], Train Losses: mse: 10.7500, mae: 1.9566, huber: 1.5383, swd: 1.0522, ept: 265.5198
    Epoch [29/50], Val Losses: mse: 40.3077, mae: 3.7871, huber: 3.3433, swd: 2.4077, ept: 182.3063
    Epoch [29/50], Test Losses: mse: 33.2938, mae: 3.2721, huber: 2.8366, swd: 1.8781, ept: 213.0302
      Epoch 29 composite train-obj: 10.750009
            No improvement (40.3077), counter 4/5
    Epoch [30/50], Train Losses: mse: 9.5240, mae: 1.8312, huber: 1.4182, swd: 0.9401, ept: 272.0253
    Epoch [30/50], Val Losses: mse: 41.3784, mae: 3.8363, huber: 3.3921, swd: 2.1824, ept: 184.0280
    Epoch [30/50], Test Losses: mse: 31.8634, mae: 3.2268, huber: 2.7910, swd: 1.7286, ept: 208.8743
      Epoch 30 composite train-obj: 9.524027
    Epoch [30/50], Test Losses: mse: 31.1466, mae: 3.2085, huber: 2.7715, swd: 1.6824, ept: 211.7492
    Best round's Test MSE: 31.1428, MAE: 3.2084, SWD: 1.6837
    Best round's Validation MSE: 39.6006, MAE: 3.8031, SWD: 2.3966
    Best round's Test verification MSE : 31.1466, MAE: 3.2085, SWD: 1.6824
    Time taken: 135.43 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.8731, mae: 6.5469, huber: 6.0658, swd: 41.1594, ept: 34.0374
    Epoch [1/50], Val Losses: mse: 56.9989, mae: 5.7504, huber: 5.2726, swd: 26.4673, ept: 41.7919
    Epoch [1/50], Test Losses: mse: 53.9519, mae: 5.4951, huber: 5.0195, swd: 26.7282, ept: 38.0256
      Epoch 1 composite train-obj: 73.873144
            Val objective improved inf → 56.9989, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 45.7086, mae: 4.9219, huber: 4.4504, swd: 14.6318, ept: 85.7160
    Epoch [2/50], Val Losses: mse: 46.5662, mae: 4.9542, huber: 4.4833, swd: 12.0070, ept: 83.4899
    Epoch [2/50], Test Losses: mse: 43.1379, mae: 4.7157, huber: 4.2462, swd: 12.6572, ept: 83.2672
      Epoch 2 composite train-obj: 45.708578
            Val objective improved 56.9989 → 46.5662, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 37.6453, mae: 4.2928, huber: 3.8279, swd: 8.3835, ept: 124.5933
    Epoch [3/50], Val Losses: mse: 44.0964, mae: 4.7118, huber: 4.2439, swd: 9.2476, ept: 101.8287
    Epoch [3/50], Test Losses: mse: 39.4344, mae: 4.4428, huber: 3.9761, swd: 8.8965, ept: 100.6914
      Epoch 3 composite train-obj: 37.645313
            Val objective improved 46.5662 → 44.0964, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 33.1521, mae: 3.9333, huber: 3.4727, swd: 6.0721, ept: 148.5274
    Epoch [4/50], Val Losses: mse: 41.4929, mae: 4.4822, huber: 4.0166, swd: 7.1522, ept: 119.7906
    Epoch [4/50], Test Losses: mse: 38.4029, mae: 4.3034, huber: 3.8391, swd: 7.9604, ept: 117.6789
      Epoch 4 composite train-obj: 33.152118
            Val objective improved 44.0964 → 41.4929, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.6620, mae: 3.7375, huber: 3.2794, swd: 5.1808, ept: 160.7034
    Epoch [5/50], Val Losses: mse: 44.4185, mae: 4.5580, huber: 4.0925, swd: 6.2004, ept: 126.9582
    Epoch [5/50], Test Losses: mse: 35.7568, mae: 4.0700, huber: 3.6077, swd: 5.7129, ept: 140.9283
      Epoch 5 composite train-obj: 30.662000
            No improvement (44.4185), counter 1/5
    Epoch [6/50], Train Losses: mse: 27.3385, mae: 3.4811, huber: 3.0263, swd: 4.0653, ept: 174.3246
    Epoch [6/50], Val Losses: mse: 43.9945, mae: 4.5180, huber: 4.0537, swd: 6.0748, ept: 123.9614
    Epoch [6/50], Test Losses: mse: 35.1614, mae: 4.0212, huber: 3.5599, swd: 5.1441, ept: 133.7666
      Epoch 6 composite train-obj: 27.338484
            No improvement (43.9945), counter 2/5
    Epoch [7/50], Train Losses: mse: 25.2384, mae: 3.3223, huber: 2.8694, swd: 3.4254, ept: 181.9329
    Epoch [7/50], Val Losses: mse: 47.2870, mae: 4.6391, huber: 4.1756, swd: 6.0131, ept: 129.6390
    Epoch [7/50], Test Losses: mse: 35.0389, mae: 3.9569, huber: 3.4971, swd: 4.7966, ept: 143.6263
      Epoch 7 composite train-obj: 25.238413
            No improvement (47.2870), counter 3/5
    Epoch [8/50], Train Losses: mse: 22.9969, mae: 3.1452, huber: 2.6947, swd: 2.9208, ept: 190.3893
    Epoch [8/50], Val Losses: mse: 48.1568, mae: 4.5869, huber: 4.1251, swd: 5.2460, ept: 130.8145
    Epoch [8/50], Test Losses: mse: 38.2430, mae: 4.0434, huber: 3.5842, swd: 3.6731, ept: 138.9490
      Epoch 8 composite train-obj: 22.996902
            No improvement (48.1568), counter 4/5
    Epoch [9/50], Train Losses: mse: 21.2094, mae: 2.9949, huber: 2.5471, swd: 2.5328, ept: 198.1066
    Epoch [9/50], Val Losses: mse: 43.7066, mae: 4.3610, huber: 3.9010, swd: 5.4493, ept: 136.2341
    Epoch [9/50], Test Losses: mse: 34.2083, mae: 3.8041, huber: 3.3480, swd: 4.1903, ept: 150.1972
      Epoch 9 composite train-obj: 21.209409
    Epoch [9/50], Test Losses: mse: 38.4019, mae: 4.3034, huber: 3.8391, swd: 7.9584, ept: 117.6298
    Best round's Test MSE: 38.4029, MAE: 4.3034, SWD: 7.9604
    Best round's Validation MSE: 41.4929, MAE: 4.4822, SWD: 7.1522
    Best round's Test verification MSE : 38.4019, MAE: 4.3034, SWD: 7.9584
    Time taken: 41.74 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.6711, mae: 6.6494, huber: 6.1678, swd: 44.5825, ept: 29.4808
    Epoch [1/50], Val Losses: mse: 59.3146, mae: 5.9102, huber: 5.4311, swd: 30.4514, ept: 34.5727
    Epoch [1/50], Test Losses: mse: 55.7561, mae: 5.6036, huber: 5.1277, swd: 31.3415, ept: 32.1911
      Epoch 1 composite train-obj: 75.671089
            Val objective improved inf → 59.3146, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.4783, mae: 5.0350, huber: 4.5626, swd: 18.9008, ept: 81.1051
    Epoch [2/50], Val Losses: mse: 52.2814, mae: 5.2495, huber: 4.7771, swd: 14.1357, ept: 83.7769
    Epoch [2/50], Test Losses: mse: 45.1586, mae: 4.8221, huber: 4.3520, swd: 15.2847, ept: 83.8496
      Epoch 2 composite train-obj: 47.478256
            Val objective improved 59.3146 → 52.2814, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.0165, mae: 4.3078, huber: 3.8427, swd: 9.8241, ept: 127.2108
    Epoch [3/50], Val Losses: mse: 42.7327, mae: 4.6260, huber: 4.1590, swd: 11.1343, ept: 99.8630
    Epoch [3/50], Test Losses: mse: 37.8789, mae: 4.3328, huber: 3.8670, swd: 11.1091, ept: 101.6109
      Epoch 3 composite train-obj: 38.016514
            Val objective improved 52.2814 → 42.7327, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 33.4895, mae: 3.9414, huber: 3.4812, swd: 6.9737, ept: 150.2733
    Epoch [4/50], Val Losses: mse: 39.2855, mae: 4.3464, huber: 3.8819, swd: 8.1425, ept: 113.8563
    Epoch [4/50], Test Losses: mse: 35.6590, mae: 4.1505, huber: 3.6865, swd: 8.9512, ept: 117.3437
      Epoch 4 composite train-obj: 33.489467
            Val objective improved 42.7327 → 39.2855, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 30.4027, mae: 3.7077, huber: 3.2502, swd: 5.6094, ept: 163.4221
    Epoch [5/50], Val Losses: mse: 41.2508, mae: 4.4112, huber: 3.9470, swd: 6.6228, ept: 124.5331
    Epoch [5/50], Test Losses: mse: 33.4731, mae: 3.9719, huber: 3.5102, swd: 6.2768, ept: 136.4762
      Epoch 5 composite train-obj: 30.402732
            No improvement (41.2508), counter 1/5
    Epoch [6/50], Train Losses: mse: 28.0853, mae: 3.5337, huber: 3.0781, swd: 4.7062, ept: 171.4705
    Epoch [6/50], Val Losses: mse: 38.4331, mae: 4.2462, huber: 3.7835, swd: 6.0238, ept: 126.6455
    Epoch [6/50], Test Losses: mse: 32.8389, mae: 3.9251, huber: 3.4650, swd: 6.2725, ept: 132.3281
      Epoch 6 composite train-obj: 28.085331
            Val objective improved 39.2855 → 38.4331, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 25.7539, mae: 3.3630, huber: 2.9095, swd: 3.9413, ept: 178.2169
    Epoch [7/50], Val Losses: mse: 37.8769, mae: 4.1086, huber: 3.6493, swd: 5.8933, ept: 136.5334
    Epoch [7/50], Test Losses: mse: 30.9372, mae: 3.7308, huber: 3.2737, swd: 5.9828, ept: 146.2020
      Epoch 7 composite train-obj: 25.753900
            Val objective improved 38.4331 → 37.8769, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 23.2703, mae: 3.1646, huber: 2.7140, swd: 3.2809, ept: 188.4000
    Epoch [8/50], Val Losses: mse: 37.7709, mae: 4.0201, huber: 3.5623, swd: 4.0989, ept: 149.3857
    Epoch [8/50], Test Losses: mse: 30.3852, mae: 3.6475, huber: 3.1913, swd: 4.0112, ept: 157.1736
      Epoch 8 composite train-obj: 23.270309
            Val objective improved 37.8769 → 37.7709, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 21.5828, mae: 3.0309, huber: 2.5823, swd: 2.7962, ept: 194.2177
    Epoch [9/50], Val Losses: mse: 36.8768, mae: 3.9368, huber: 3.4804, swd: 4.1571, ept: 149.5529
    Epoch [9/50], Test Losses: mse: 30.1867, mae: 3.5921, huber: 3.1376, swd: 3.8267, ept: 157.4091
      Epoch 9 composite train-obj: 21.582838
            Val objective improved 37.7709 → 36.8768, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 19.9302, mae: 2.8895, huber: 2.4436, swd: 2.4274, ept: 201.1813
    Epoch [10/50], Val Losses: mse: 36.5742, mae: 3.8964, huber: 3.4405, swd: 3.7053, ept: 151.0959
    Epoch [10/50], Test Losses: mse: 28.9950, mae: 3.4871, huber: 3.0342, swd: 3.5489, ept: 165.5652
      Epoch 10 composite train-obj: 19.930199
            Val objective improved 36.8768 → 36.5742, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 18.9702, mae: 2.8019, huber: 2.3578, swd: 2.2362, ept: 206.4195
    Epoch [11/50], Val Losses: mse: 38.2550, mae: 3.9545, huber: 3.5005, swd: 3.7864, ept: 164.3421
    Epoch [11/50], Test Losses: mse: 28.7539, mae: 3.4224, huber: 2.9716, swd: 3.2095, ept: 176.8305
      Epoch 11 composite train-obj: 18.970210
            No improvement (38.2550), counter 1/5
    Epoch [12/50], Train Losses: mse: 17.9929, mae: 2.7150, huber: 2.2727, swd: 2.0815, ept: 212.1723
    Epoch [12/50], Val Losses: mse: 35.9427, mae: 3.7886, huber: 3.3348, swd: 3.0954, ept: 168.2026
    Epoch [12/50], Test Losses: mse: 28.3854, mae: 3.4112, huber: 2.9596, swd: 2.9134, ept: 174.1469
      Epoch 12 composite train-obj: 17.992934
            Val objective improved 36.5742 → 35.9427, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 17.5620, mae: 2.6751, huber: 2.2337, swd: 2.0394, ept: 214.1348
    Epoch [13/50], Val Losses: mse: 38.4646, mae: 3.9169, huber: 3.4624, swd: 3.3407, ept: 160.5378
    Epoch [13/50], Test Losses: mse: 29.7128, mae: 3.4311, huber: 2.9805, swd: 3.0295, ept: 179.8393
      Epoch 13 composite train-obj: 17.561952
            No improvement (38.4646), counter 1/5
    Epoch [14/50], Train Losses: mse: 16.5111, mae: 2.5781, huber: 2.1391, swd: 1.8502, ept: 220.1328
    Epoch [14/50], Val Losses: mse: 35.4755, mae: 3.7434, huber: 3.2919, swd: 3.2774, ept: 162.3638
    Epoch [14/50], Test Losses: mse: 30.6237, mae: 3.4451, huber: 2.9965, swd: 2.8065, ept: 176.5071
      Epoch 14 composite train-obj: 16.511130
            Val objective improved 35.9427 → 35.4755, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 15.7850, mae: 2.5110, huber: 2.0737, swd: 1.6931, ept: 224.8200
    Epoch [15/50], Val Losses: mse: 40.3120, mae: 3.9843, huber: 3.5311, swd: 3.8205, ept: 161.4232
    Epoch [15/50], Test Losses: mse: 28.9501, mae: 3.4016, huber: 2.9507, swd: 3.1059, ept: 171.7415
      Epoch 15 composite train-obj: 15.785039
            No improvement (40.3120), counter 1/5
    Epoch [16/50], Train Losses: mse: 15.4535, mae: 2.4734, huber: 2.0374, swd: 1.6790, ept: 227.9675
    Epoch [16/50], Val Losses: mse: 37.0847, mae: 3.7902, huber: 3.3384, swd: 3.1562, ept: 170.2113
    Epoch [16/50], Test Losses: mse: 28.9113, mae: 3.3564, huber: 2.9080, swd: 2.9841, ept: 179.8202
      Epoch 16 composite train-obj: 15.453481
            No improvement (37.0847), counter 2/5
    Epoch [17/50], Train Losses: mse: 14.6003, mae: 2.3864, huber: 1.9530, swd: 1.5446, ept: 234.2910
    Epoch [17/50], Val Losses: mse: 36.3608, mae: 3.7488, huber: 3.2985, swd: 3.2101, ept: 169.3458
    Epoch [17/50], Test Losses: mse: 29.1665, mae: 3.3455, huber: 2.8985, swd: 2.8875, ept: 179.5770
      Epoch 17 composite train-obj: 14.600251
            No improvement (36.3608), counter 3/5
    Epoch [18/50], Train Losses: mse: 13.7645, mae: 2.3119, huber: 1.8803, swd: 1.4424, ept: 238.7460
    Epoch [18/50], Val Losses: mse: 37.4155, mae: 3.7752, huber: 3.3245, swd: 2.6893, ept: 175.8144
    Epoch [18/50], Test Losses: mse: 27.2676, mae: 3.2351, huber: 2.7889, swd: 2.6672, ept: 190.0901
      Epoch 18 composite train-obj: 13.764466
            No improvement (37.4155), counter 4/5
    Epoch [19/50], Train Losses: mse: 13.6925, mae: 2.3003, huber: 1.8693, swd: 1.3902, ept: 239.7850
    Epoch [19/50], Val Losses: mse: 36.7815, mae: 3.6905, huber: 3.2421, swd: 2.6449, ept: 183.5872
    Epoch [19/50], Test Losses: mse: 26.2307, mae: 3.1507, huber: 2.7064, swd: 2.1262, ept: 195.2868
      Epoch 19 composite train-obj: 13.692515
    Epoch [19/50], Test Losses: mse: 30.6231, mae: 3.4451, huber: 2.9966, swd: 2.8069, ept: 176.5460
    Best round's Test MSE: 30.6237, MAE: 3.4451, SWD: 2.8065
    Best round's Validation MSE: 35.4755, MAE: 3.7434, SWD: 3.2774
    Best round's Test verification MSE : 30.6231, MAE: 3.4451, SWD: 2.8069
    Time taken: 87.69 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_2146)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 33.3898 ± 3.5511
      mae: 3.6523 ± 0.4704
      huber: 3.2023 ± 0.4595
      swd: 4.1502 ± 2.7329
      ept: 168.6709 ± 38.8330
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 38.8563 ± 2.5123
      mae: 4.0096 ± 0.3351
      huber: 3.5554 ± 0.3272
      swd: 4.2754 ± 2.0658
      ept: 154.5788 ± 25.8199
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 264.95 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_2146
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE + 0.5 SWD


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
    mixing_strategy='convex', ### Here
    loss_backward_weights = [1.0, 0.0, 0.0, 0.5, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.5, 0.0],
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

    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 82.0629, mae: 7.0258, huber: 6.5421, swd: 23.9779, ept: 17.9927
    Epoch [1/50], Val Losses: mse: 70.0927, mae: 6.6505, huber: 6.1668, swd: 10.5825, ept: 21.2040
    Epoch [1/50], Test Losses: mse: 65.4715, mae: 6.2840, huber: 5.8027, swd: 9.3775, ept: 19.1075
      Epoch 1 composite train-obj: 94.051870
            Val objective improved inf → 75.3840, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.9255, mae: 5.5346, huber: 5.0569, swd: 6.6395, ept: 50.3219
    Epoch [2/50], Val Losses: mse: 59.4504, mae: 5.8301, huber: 5.3519, swd: 6.0080, ept: 46.8398
    Epoch [2/50], Test Losses: mse: 52.1712, mae: 5.3712, huber: 4.8952, swd: 5.5661, ept: 44.2951
      Epoch 2 composite train-obj: 57.245252
            Val objective improved 75.3840 → 62.4545, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 44.6574, mae: 4.8493, huber: 4.3771, swd: 4.3430, ept: 87.9642
    Epoch [3/50], Val Losses: mse: 55.8254, mae: 5.4569, huber: 4.9825, swd: 5.9632, ept: 73.1491
    Epoch [3/50], Test Losses: mse: 46.9820, mae: 4.9578, huber: 4.4852, swd: 6.3018, ept: 75.6332
      Epoch 3 composite train-obj: 46.828942
            Val objective improved 62.4545 → 58.8070, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.5911, mae: 4.4649, huber: 3.9964, swd: 3.4467, ept: 111.7521
    Epoch [4/50], Val Losses: mse: 51.4541, mae: 5.1598, huber: 4.6873, swd: 3.3881, ept: 83.5109
    Epoch [4/50], Test Losses: mse: 42.4385, mae: 4.6595, huber: 4.1889, swd: 3.8070, ept: 88.2284
      Epoch 4 composite train-obj: 41.314442
            Val objective improved 58.8070 → 53.1482, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 36.0335, mae: 4.1863, huber: 3.7206, swd: 2.9153, ept: 130.8198
    Epoch [5/50], Val Losses: mse: 52.6429, mae: 5.1063, huber: 4.6358, swd: 3.5923, ept: 96.6749
    Epoch [5/50], Test Losses: mse: 42.4818, mae: 4.5749, huber: 4.1060, swd: 3.8236, ept: 104.6364
      Epoch 5 composite train-obj: 37.491169
            No improvement (54.4391), counter 1/5
    Epoch [6/50], Train Losses: mse: 32.9190, mae: 3.9450, huber: 3.4820, swd: 2.5014, ept: 146.4093
    Epoch [6/50], Val Losses: mse: 52.0356, mae: 5.0056, huber: 4.5365, swd: 3.0647, ept: 110.9329
    Epoch [6/50], Test Losses: mse: 40.4381, mae: 4.3934, huber: 3.9263, swd: 3.0274, ept: 119.1569
      Epoch 6 composite train-obj: 34.169723
            No improvement (53.5680), counter 2/5
    Epoch [7/50], Train Losses: mse: 30.4940, mae: 3.7619, huber: 3.3010, swd: 2.2197, ept: 156.2184
    Epoch [7/50], Val Losses: mse: 52.8839, mae: 5.0101, huber: 4.5413, swd: 2.8798, ept: 113.0829
    Epoch [7/50], Test Losses: mse: 40.4759, mae: 4.3504, huber: 3.8843, swd: 2.8115, ept: 126.6992
      Epoch 7 composite train-obj: 31.603911
            No improvement (54.3238), counter 3/5
    Epoch [8/50], Train Losses: mse: 29.0008, mae: 3.6385, huber: 3.1792, swd: 2.1014, ept: 164.2882
    Epoch [8/50], Val Losses: mse: 57.0585, mae: 5.1300, huber: 4.6618, swd: 2.9112, ept: 110.5672
    Epoch [8/50], Test Losses: mse: 40.0137, mae: 4.2895, huber: 3.8245, swd: 2.9291, ept: 124.4256
      Epoch 8 composite train-obj: 30.051534
            No improvement (58.5141), counter 4/5
    Epoch [9/50], Train Losses: mse: 27.0395, mae: 3.4807, huber: 3.0237, swd: 1.8791, ept: 172.8551
    Epoch [9/50], Val Losses: mse: 54.9041, mae: 5.0200, huber: 4.5524, swd: 2.7178, ept: 122.0398
    Epoch [9/50], Test Losses: mse: 39.8452, mae: 4.2015, huber: 3.7378, swd: 2.3182, ept: 137.7653
      Epoch 9 composite train-obj: 27.979074
    Epoch [9/50], Test Losses: mse: 42.4317, mae: 4.6593, huber: 4.1887, swd: 3.8048, ept: 88.2746
    Best round's Test MSE: 42.4385, MAE: 4.6595, SWD: 3.8070
    Best round's Validation MSE: 51.4541, MAE: 5.1598, SWD: 3.3881
    Best round's Test verification MSE : 42.4317, MAE: 4.6593, SWD: 3.8048
    Time taken: 42.40 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 80.0274, mae: 6.9279, huber: 6.4444, swd: 22.8584, ept: 18.7380
    Epoch [1/50], Val Losses: mse: 64.8544, mae: 6.3173, huber: 5.8352, swd: 8.4684, ept: 22.5468
    Epoch [1/50], Test Losses: mse: 60.1405, mae: 5.9537, huber: 5.4737, swd: 8.7855, ept: 22.1426
      Epoch 1 composite train-obj: 91.456609
            Val objective improved inf → 69.0886, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.7279, mae: 5.2616, huber: 4.7857, swd: 5.4349, ept: 63.2753
    Epoch [2/50], Val Losses: mse: 52.0466, mae: 5.3587, huber: 4.8833, swd: 4.7352, ept: 65.6931
    Epoch [2/50], Test Losses: mse: 47.1126, mae: 5.0377, huber: 4.5637, swd: 4.7937, ept: 65.1097
      Epoch 2 composite train-obj: 52.445368
            Val objective improved 69.0886 → 54.4142, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.4195, mae: 4.6208, huber: 4.1508, swd: 3.6727, ept: 105.3719
    Epoch [3/50], Val Losses: mse: 51.4095, mae: 5.2656, huber: 4.7913, swd: 4.0632, ept: 80.1907
    Epoch [3/50], Test Losses: mse: 45.2000, mae: 4.8865, huber: 4.4142, swd: 4.1193, ept: 85.8379
      Epoch 3 composite train-obj: 43.255823
            Val objective improved 54.4142 → 53.4411, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.1752, mae: 4.2032, huber: 3.7377, swd: 2.8840, ept: 133.9582
    Epoch [4/50], Val Losses: mse: 46.9559, mae: 4.8750, huber: 4.4044, swd: 3.3181, ept: 106.1448
    Epoch [4/50], Test Losses: mse: 42.1181, mae: 4.6051, huber: 4.1355, swd: 3.2085, ept: 101.5899
      Epoch 4 composite train-obj: 37.617231
            Val objective improved 53.4411 → 48.6150, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.5706, mae: 3.9144, huber: 3.4523, swd: 2.3570, ept: 153.2209
    Epoch [5/50], Val Losses: mse: 46.5059, mae: 4.7246, huber: 4.2568, swd: 3.4193, ept: 118.6174
    Epoch [5/50], Test Losses: mse: 38.3309, mae: 4.2694, huber: 3.8040, swd: 2.9873, ept: 127.1859
      Epoch 5 composite train-obj: 33.749079
            Val objective improved 48.6150 → 48.2156, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.6898, mae: 3.6800, huber: 3.2210, swd: 2.0957, ept: 168.9859
    Epoch [6/50], Val Losses: mse: 45.9568, mae: 4.6850, huber: 4.2177, swd: 2.7215, ept: 123.4600
    Epoch [6/50], Test Losses: mse: 38.8551, mae: 4.2764, huber: 3.8111, swd: 2.6810, ept: 130.7801
      Epoch 6 composite train-obj: 30.737647
            Val objective improved 48.2156 → 47.3175, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 26.7362, mae: 3.4573, huber: 3.0009, swd: 1.7502, ept: 181.2609
    Epoch [7/50], Val Losses: mse: 47.0018, mae: 4.6304, huber: 4.1651, swd: 2.5220, ept: 133.9371
    Epoch [7/50], Test Losses: mse: 39.1393, mae: 4.2154, huber: 3.7521, swd: 2.5061, ept: 143.0304
      Epoch 7 composite train-obj: 27.611318
            No improvement (48.2628), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.7274, mae: 3.3649, huber: 2.9100, swd: 1.6768, ept: 186.0080
    Epoch [8/50], Val Losses: mse: 53.2811, mae: 4.8682, huber: 4.4020, swd: 2.5747, ept: 143.4119
    Epoch [8/50], Test Losses: mse: 39.4978, mae: 4.1547, huber: 3.6918, swd: 2.3270, ept: 157.1579
      Epoch 8 composite train-obj: 26.565774
            No improvement (54.5685), counter 2/5
    Epoch [9/50], Train Losses: mse: 23.2938, mae: 3.1675, huber: 2.7158, swd: 1.4913, ept: 197.2829
    Epoch [9/50], Val Losses: mse: 47.8258, mae: 4.5765, huber: 4.1133, swd: 2.2742, ept: 139.2451
    Epoch [9/50], Test Losses: mse: 37.3956, mae: 4.0234, huber: 3.5629, swd: 2.0727, ept: 152.1337
      Epoch 9 composite train-obj: 24.039485
            No improvement (48.9629), counter 3/5
    Epoch [10/50], Train Losses: mse: 21.8085, mae: 3.0417, huber: 2.5922, swd: 1.4028, ept: 205.1783
    Epoch [10/50], Val Losses: mse: 46.7525, mae: 4.4731, huber: 4.0109, swd: 2.4715, ept: 141.2460
    Epoch [10/50], Test Losses: mse: 39.4736, mae: 4.0756, huber: 3.6154, swd: 1.8828, ept: 149.7095
      Epoch 10 composite train-obj: 22.509858
            No improvement (47.9883), counter 4/5
    Epoch [11/50], Train Losses: mse: 21.1072, mae: 2.9698, huber: 2.5218, swd: 1.3825, ept: 208.9192
    Epoch [11/50], Val Losses: mse: 52.2057, mae: 4.6725, huber: 4.2116, swd: 2.9440, ept: 144.0695
    Epoch [11/50], Test Losses: mse: 39.0402, mae: 4.0070, huber: 3.5491, swd: 2.3392, ept: 156.8990
      Epoch 11 composite train-obj: 21.798464
    Epoch [11/50], Test Losses: mse: 38.8569, mae: 4.2766, huber: 3.8112, swd: 2.6809, ept: 130.6972
    Best round's Test MSE: 38.8551, MAE: 4.2764, SWD: 2.6810
    Best round's Validation MSE: 45.9568, MAE: 4.6850, SWD: 2.7215
    Best round's Test verification MSE : 38.8569, MAE: 4.2766, SWD: 2.6809
    Time taken: 50.77 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 81.3662, mae: 6.9491, huber: 6.4658, swd: 24.1312, ept: 20.3027
    Epoch [1/50], Val Losses: mse: 65.2440, mae: 6.2824, huber: 5.8012, swd: 8.2632, ept: 28.1685
    Epoch [1/50], Test Losses: mse: 61.7257, mae: 5.9836, huber: 5.5042, swd: 8.7799, ept: 26.5506
      Epoch 1 composite train-obj: 93.431770
            Val objective improved inf → 69.3756, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 51.1463, mae: 5.3351, huber: 4.8587, swd: 6.0820, ept: 60.8442
    Epoch [2/50], Val Losses: mse: 55.6795, mae: 5.5700, huber: 5.0935, swd: 7.0228, ept: 62.5156
    Epoch [2/50], Test Losses: mse: 50.4560, mae: 5.2044, huber: 4.7302, swd: 6.5435, ept: 58.7628
      Epoch 2 composite train-obj: 54.187361
            Val objective improved 69.3756 → 59.1909, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.5304, mae: 4.6963, huber: 4.2252, swd: 4.3117, ept: 96.5592
    Epoch [3/50], Val Losses: mse: 50.2648, mae: 5.1613, huber: 4.6881, swd: 5.2538, ept: 82.9186
    Epoch [3/50], Test Losses: mse: 44.5902, mae: 4.7917, huber: 4.3202, swd: 4.6260, ept: 81.1109
      Epoch 3 composite train-obj: 44.686283
            Val objective improved 59.1909 → 52.8917, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.3397, mae: 4.2977, huber: 3.8306, swd: 3.3249, ept: 125.5371
    Epoch [4/50], Val Losses: mse: 49.1654, mae: 4.9964, huber: 4.5248, swd: 3.9010, ept: 90.7134
    Epoch [4/50], Test Losses: mse: 42.6440, mae: 4.6285, huber: 4.1581, swd: 3.4994, ept: 93.3117
      Epoch 4 composite train-obj: 39.002152
            Val objective improved 52.8917 → 51.1159, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.6390, mae: 4.0158, huber: 3.5517, swd: 2.7428, ept: 143.7349
    Epoch [5/50], Val Losses: mse: 46.9369, mae: 4.7748, huber: 4.3061, swd: 3.4216, ept: 116.1821
    Epoch [5/50], Test Losses: mse: 40.9482, mae: 4.4408, huber: 3.9732, swd: 3.2697, ept: 122.2403
      Epoch 5 composite train-obj: 35.010352
            Val objective improved 51.1159 → 48.6477, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.8034, mae: 3.7930, huber: 3.3314, swd: 2.3667, ept: 157.2426
    Epoch [6/50], Val Losses: mse: 50.3135, mae: 5.0042, huber: 4.5330, swd: 3.3436, ept: 104.0807
    Epoch [6/50], Test Losses: mse: 42.0013, mae: 4.5404, huber: 4.0713, swd: 3.1788, ept: 114.3383
      Epoch 6 composite train-obj: 31.986777
            No improvement (51.9853), counter 1/5
    Epoch [7/50], Train Losses: mse: 28.2845, mae: 3.6004, huber: 3.1412, swd: 2.0653, ept: 168.9471
    Epoch [7/50], Val Losses: mse: 47.9677, mae: 4.7086, huber: 4.2416, swd: 2.8014, ept: 125.6926
    Epoch [7/50], Test Losses: mse: 38.7538, mae: 4.2191, huber: 3.7539, swd: 2.4919, ept: 135.9868
      Epoch 7 composite train-obj: 29.317178
            No improvement (49.3684), counter 2/5
    Epoch [8/50], Train Losses: mse: 26.0739, mae: 3.4184, huber: 2.9617, swd: 1.8037, ept: 179.8238
    Epoch [8/50], Val Losses: mse: 48.3898, mae: 4.6969, huber: 4.2306, swd: 2.9801, ept: 132.8668
    Epoch [8/50], Test Losses: mse: 39.6955, mae: 4.2121, huber: 3.7478, swd: 2.4420, ept: 141.0829
      Epoch 8 composite train-obj: 26.975743
            No improvement (49.8799), counter 3/5
    Epoch [9/50], Train Losses: mse: 23.7838, mae: 3.2312, huber: 2.7774, swd: 1.5954, ept: 190.0891
    Epoch [9/50], Val Losses: mse: 49.5354, mae: 4.7264, huber: 4.2604, swd: 2.5294, ept: 132.0282
    Epoch [9/50], Test Losses: mse: 38.2334, mae: 4.0915, huber: 3.6295, swd: 2.2672, ept: 146.6010
      Epoch 9 composite train-obj: 24.581457
            No improvement (50.8001), counter 4/5
    Epoch [10/50], Train Losses: mse: 22.8045, mae: 3.1413, huber: 2.6892, swd: 1.5288, ept: 194.9989
    Epoch [10/50], Val Losses: mse: 46.8205, mae: 4.5276, huber: 4.0637, swd: 2.6429, ept: 141.6756
    Epoch [10/50], Test Losses: mse: 36.6990, mae: 3.9767, huber: 3.5157, swd: 2.1455, ept: 156.6342
      Epoch 10 composite train-obj: 23.568946
            Val objective improved 48.6477 → 48.1420, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.9364, mae: 3.0628, huber: 2.6122, swd: 1.4925, ept: 199.3533
    Epoch [11/50], Val Losses: mse: 46.6015, mae: 4.5281, huber: 4.0642, swd: 2.4543, ept: 136.9072
    Epoch [11/50], Test Losses: mse: 38.3160, mae: 4.0229, huber: 3.5627, swd: 2.0994, ept: 152.8731
      Epoch 11 composite train-obj: 22.682682
            Val objective improved 48.1420 → 47.8286, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.7591, mae: 2.9694, huber: 2.5205, swd: 1.3862, ept: 205.6357
    Epoch [12/50], Val Losses: mse: 46.6360, mae: 4.4597, huber: 3.9975, swd: 2.1277, ept: 144.9096
    Epoch [12/50], Test Losses: mse: 35.4426, mae: 3.8296, huber: 3.3713, swd: 1.7611, ept: 163.0471
      Epoch 12 composite train-obj: 21.452159
            Val objective improved 47.8286 → 47.6999, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.9367, mae: 2.8080, huber: 2.3623, swd: 1.2213, ept: 216.0302
    Epoch [13/50], Val Losses: mse: 45.8944, mae: 4.3757, huber: 3.9161, swd: 2.3887, ept: 153.0185
    Epoch [13/50], Test Losses: mse: 36.4977, mae: 3.8510, huber: 3.3947, swd: 1.9868, ept: 166.2079
      Epoch 13 composite train-obj: 19.547365
            Val objective improved 47.6999 → 47.0887, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.5656, mae: 2.7686, huber: 2.3240, swd: 1.2076, ept: 217.9658
    Epoch [14/50], Val Losses: mse: 46.8423, mae: 4.4729, huber: 4.0110, swd: 3.0801, ept: 142.6194
    Epoch [14/50], Test Losses: mse: 40.7859, mae: 4.0354, huber: 3.5777, swd: 2.3516, ept: 159.4056
      Epoch 14 composite train-obj: 19.169364
            No improvement (48.3823), counter 1/5
    Epoch [15/50], Train Losses: mse: 18.1787, mae: 2.7321, huber: 2.2883, swd: 1.1809, ept: 220.2823
    Epoch [15/50], Val Losses: mse: 46.4771, mae: 4.4106, huber: 3.9509, swd: 2.4205, ept: 150.6994
    Epoch [15/50], Test Losses: mse: 35.2464, mae: 3.7589, huber: 3.3038, swd: 1.7977, ept: 169.7195
      Epoch 15 composite train-obj: 18.769129
            No improvement (47.6873), counter 2/5
    Epoch [16/50], Train Losses: mse: 17.2277, mae: 2.6417, huber: 2.1999, swd: 1.1469, ept: 226.1598
    Epoch [16/50], Val Losses: mse: 46.9261, mae: 4.3505, huber: 3.8931, swd: 2.2504, ept: 156.5843
    Epoch [16/50], Test Losses: mse: 35.1813, mae: 3.6894, huber: 3.2366, swd: 1.8225, ept: 181.2450
      Epoch 16 composite train-obj: 17.801165
            No improvement (48.0513), counter 3/5
    Epoch [17/50], Train Losses: mse: 16.3660, mae: 2.5633, huber: 2.1236, swd: 1.0491, ept: 231.3796
    Epoch [17/50], Val Losses: mse: 44.7021, mae: 4.2537, huber: 3.7960, swd: 1.9435, ept: 152.2571
    Epoch [17/50], Test Losses: mse: 35.5567, mae: 3.7271, huber: 3.2733, swd: 1.6083, ept: 174.4190
      Epoch 17 composite train-obj: 16.890530
            Val objective improved 47.0887 → 45.6738, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.7046, mae: 2.5080, huber: 2.0693, swd: 1.0046, ept: 234.2073
    Epoch [18/50], Val Losses: mse: 47.9293, mae: 4.3689, huber: 3.9109, swd: 1.9600, ept: 156.7074
    Epoch [18/50], Test Losses: mse: 33.7912, mae: 3.6066, huber: 3.1541, swd: 1.6597, ept: 183.7390
      Epoch 18 composite train-obj: 16.206871
            No improvement (48.9093), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.9208, mae: 2.4258, huber: 1.9894, swd: 0.9665, ept: 239.6925
    Epoch [19/50], Val Losses: mse: 49.6802, mae: 4.4394, huber: 3.9813, swd: 2.3125, ept: 160.2813
    Epoch [19/50], Test Losses: mse: 37.1464, mae: 3.7550, huber: 3.3023, swd: 1.6842, ept: 186.2279
      Epoch 19 composite train-obj: 15.404079
            No improvement (50.8364), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.5913, mae: 2.3847, huber: 1.9500, swd: 0.9459, ept: 242.0826
    Epoch [20/50], Val Losses: mse: 45.0060, mae: 4.2204, huber: 3.7649, swd: 2.1681, ept: 163.2001
    Epoch [20/50], Test Losses: mse: 35.2736, mae: 3.6224, huber: 3.1724, swd: 1.7914, ept: 191.8176
      Epoch 20 composite train-obj: 15.064256
            No improvement (46.0900), counter 3/5
    Epoch [21/50], Train Losses: mse: 13.8114, mae: 2.3124, huber: 1.8799, swd: 0.8963, ept: 246.3647
    Epoch [21/50], Val Losses: mse: 48.5636, mae: 4.3259, huber: 3.8715, swd: 2.2496, ept: 156.3761
    Epoch [21/50], Test Losses: mse: 35.9879, mae: 3.6536, huber: 3.2035, swd: 1.6859, ept: 181.3681
      Epoch 21 composite train-obj: 14.259552
            No improvement (49.6884), counter 4/5
    Epoch [22/50], Train Losses: mse: 13.9844, mae: 2.3274, huber: 1.8944, swd: 0.9292, ept: 245.4744
    Epoch [22/50], Val Losses: mse: 45.6983, mae: 4.2080, huber: 3.7542, swd: 2.0846, ept: 162.3357
    Epoch [22/50], Test Losses: mse: 34.2903, mae: 3.5224, huber: 3.0757, swd: 1.7019, ept: 190.1389
      Epoch 22 composite train-obj: 14.448954
    Epoch [22/50], Test Losses: mse: 35.5581, mae: 3.7271, huber: 3.2734, swd: 1.6092, ept: 174.3969
    Best round's Test MSE: 35.5567, MAE: 3.7271, SWD: 1.6083
    Best round's Validation MSE: 44.7021, MAE: 4.2537, SWD: 1.9435
    Best round's Test verification MSE : 35.5581, MAE: 3.7271, SWD: 1.6092
    Time taken: 101.35 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_2151)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 38.9501 ± 2.8103
      mae: 4.2210 ± 0.3827
      huber: 3.7578 ± 0.3757
      swd: 2.6988 ± 0.8977
      ept: 131.1425 ± 35.1881
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 47.3710 ± 2.9323
      mae: 4.6995 ± 0.3701
      huber: 4.2337 ± 0.3641
      swd: 2.6844 ± 0.5904
      ept: 119.7427 ± 28.1883
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 194.61 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_2151
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

## TimeMixer

### huber


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
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 68.5378, mae: 6.0891, huber: 5.6115, swd: 14.7714, ept: 56.2274
    Epoch [1/50], Val Losses: mse: 73.1194, mae: 6.2198, huber: 5.7449, swd: 14.1269, ept: 60.3634
    Epoch [1/50], Test Losses: mse: 58.9262, mae: 5.4717, huber: 5.0004, swd: 17.0263, ept: 63.0101
      Epoch 1 composite train-obj: 5.611540
            Val objective improved inf → 5.7449, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.4282, mae: 5.0499, huber: 4.5818, swd: 12.2971, ept: 100.9991
    Epoch [2/50], Val Losses: mse: 65.5457, mae: 5.7642, huber: 5.2928, swd: 12.9402, ept: 88.0973
    Epoch [2/50], Test Losses: mse: 51.7283, mae: 5.0151, huber: 4.5482, swd: 14.6379, ept: 90.5404
      Epoch 2 composite train-obj: 4.581813
            Val objective improved 5.7449 → 5.2928, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 45.6724, mae: 4.4777, huber: 4.0175, swd: 9.9764, ept: 130.2276
    Epoch [3/50], Val Losses: mse: 66.0191, mae: 5.7064, huber: 5.2362, swd: 10.9218, ept: 96.7578
    Epoch [3/50], Test Losses: mse: 51.4347, mae: 4.8607, huber: 4.3968, swd: 11.2445, ept: 107.5080
      Epoch 3 composite train-obj: 4.017476
            Val objective improved 5.2928 → 5.2362, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 41.3791, mae: 4.1269, huber: 3.6730, swd: 8.3041, ept: 148.0584
    Epoch [4/50], Val Losses: mse: 66.5513, mae: 5.6588, huber: 5.1915, swd: 9.6230, ept: 106.8421
    Epoch [4/50], Test Losses: mse: 51.0945, mae: 4.7393, huber: 4.2792, swd: 9.3530, ept: 121.0844
      Epoch 4 composite train-obj: 3.673047
            Val objective improved 5.2362 → 5.1915, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 38.2673, mae: 3.8798, huber: 3.4307, swd: 7.1731, ept: 160.7728
    Epoch [5/50], Val Losses: mse: 67.0135, mae: 5.6052, huber: 5.1402, swd: 8.4978, ept: 112.5655
    Epoch [5/50], Test Losses: mse: 51.4815, mae: 4.6703, huber: 4.2134, swd: 7.6319, ept: 129.6157
      Epoch 5 composite train-obj: 3.430746
            Val objective improved 5.1915 → 5.1402, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.5597, mae: 3.6758, huber: 3.2305, swd: 6.3410, ept: 170.9143
    Epoch [6/50], Val Losses: mse: 68.9296, mae: 5.6169, huber: 5.1531, swd: 6.7565, ept: 119.3750
    Epoch [6/50], Test Losses: mse: 52.4488, mae: 4.6567, huber: 4.2008, swd: 6.4317, ept: 135.5595
      Epoch 6 composite train-obj: 3.230533
            No improvement (5.1531), counter 1/5
    Epoch [7/50], Train Losses: mse: 33.2395, mae: 3.5045, huber: 3.0622, swd: 5.5949, ept: 180.8960
    Epoch [7/50], Val Losses: mse: 69.4638, mae: 5.6276, huber: 5.1641, swd: 6.2062, ept: 123.2577
    Epoch [7/50], Test Losses: mse: 52.9615, mae: 4.6390, huber: 4.1832, swd: 5.8717, ept: 147.9039
      Epoch 7 composite train-obj: 3.062170
            No improvement (5.1641), counter 2/5
    Epoch [8/50], Train Losses: mse: 31.1648, mae: 3.3523, huber: 2.9128, swd: 5.0339, ept: 189.2759
    Epoch [8/50], Val Losses: mse: 69.3536, mae: 5.6052, huber: 5.1427, swd: 5.8313, ept: 127.7430
    Epoch [8/50], Test Losses: mse: 52.7047, mae: 4.5744, huber: 4.1201, swd: 5.1873, ept: 153.4098
      Epoch 8 composite train-obj: 2.912780
            No improvement (5.1427), counter 3/5
    Epoch [9/50], Train Losses: mse: 29.4112, mae: 3.2204, huber: 2.7833, swd: 4.5338, ept: 197.1414
    Epoch [9/50], Val Losses: mse: 69.8656, mae: 5.6125, huber: 5.1507, swd: 5.4717, ept: 129.8107
    Epoch [9/50], Test Losses: mse: 52.8895, mae: 4.5259, huber: 4.0746, swd: 4.7175, ept: 158.1075
      Epoch 9 composite train-obj: 2.783329
            No improvement (5.1507), counter 4/5
    Epoch [10/50], Train Losses: mse: 27.8984, mae: 3.1011, huber: 2.6665, swd: 4.0891, ept: 204.5713
    Epoch [10/50], Val Losses: mse: 69.4506, mae: 5.5768, huber: 5.1159, swd: 5.2940, ept: 134.3273
    Epoch [10/50], Test Losses: mse: 52.1017, mae: 4.4896, huber: 4.0388, swd: 4.4494, ept: 160.3037
      Epoch 10 composite train-obj: 2.666508
            Val objective improved 5.1402 → 5.1159, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 26.4267, mae: 2.9947, huber: 2.5621, swd: 3.7972, ept: 211.3101
    Epoch [11/50], Val Losses: mse: 71.5221, mae: 5.6152, huber: 5.1551, swd: 4.7951, ept: 135.2819
    Epoch [11/50], Test Losses: mse: 53.2466, mae: 4.4702, huber: 4.0216, swd: 4.0049, ept: 166.1812
      Epoch 11 composite train-obj: 2.562142
            No improvement (5.1551), counter 1/5
    Epoch [12/50], Train Losses: mse: 25.3442, mae: 2.9070, huber: 2.4762, swd: 3.4806, ept: 216.5581
    Epoch [12/50], Val Losses: mse: 70.8213, mae: 5.5579, huber: 5.0991, swd: 4.8395, ept: 138.5439
    Epoch [12/50], Test Losses: mse: 51.9099, mae: 4.4110, huber: 3.9622, swd: 3.9175, ept: 169.9807
      Epoch 12 composite train-obj: 2.476199
            Val objective improved 5.1159 → 5.0991, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 24.2532, mae: 2.8273, huber: 2.3980, swd: 3.2475, ept: 222.0945
    Epoch [13/50], Val Losses: mse: 71.9376, mae: 5.6021, huber: 5.1420, swd: 4.5477, ept: 138.9212
    Epoch [13/50], Test Losses: mse: 52.6387, mae: 4.4229, huber: 3.9743, swd: 3.5542, ept: 169.3813
      Epoch 13 composite train-obj: 2.398030
            No improvement (5.1420), counter 1/5
    Epoch [14/50], Train Losses: mse: 23.1031, mae: 2.7370, huber: 2.3099, swd: 3.0177, ept: 226.6207
    Epoch [14/50], Val Losses: mse: 69.6160, mae: 5.4899, huber: 5.0317, swd: 4.7376, ept: 141.3652
    Epoch [14/50], Test Losses: mse: 51.5406, mae: 4.3535, huber: 3.9069, swd: 3.4119, ept: 174.3920
      Epoch 14 composite train-obj: 2.309897
            Val objective improved 5.0991 → 5.0317, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 22.1091, mae: 2.6593, huber: 2.2340, swd: 2.8176, ept: 231.4901
    Epoch [15/50], Val Losses: mse: 70.8478, mae: 5.4925, huber: 5.0354, swd: 4.3557, ept: 142.4399
    Epoch [15/50], Test Losses: mse: 52.6226, mae: 4.3366, huber: 3.8919, swd: 3.0239, ept: 180.7857
      Epoch 15 composite train-obj: 2.234023
            No improvement (5.0354), counter 1/5
    Epoch [16/50], Train Losses: mse: 21.3921, mae: 2.6075, huber: 2.1830, swd: 2.6752, ept: 234.6650
    Epoch [16/50], Val Losses: mse: 69.2722, mae: 5.4372, huber: 4.9809, swd: 4.5710, ept: 144.3065
    Epoch [16/50], Test Losses: mse: 51.4159, mae: 4.2877, huber: 3.8428, swd: 3.1281, ept: 183.5967
      Epoch 16 composite train-obj: 2.183032
            Val objective improved 5.0317 → 4.9809, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 20.5493, mae: 2.5475, huber: 2.1241, swd: 2.5490, ept: 238.6387
    Epoch [17/50], Val Losses: mse: 70.5208, mae: 5.4856, huber: 5.0287, swd: 4.2153, ept: 143.5659
    Epoch [17/50], Test Losses: mse: 51.9009, mae: 4.3082, huber: 3.8627, swd: 2.7272, ept: 185.6295
      Epoch 17 composite train-obj: 2.124052
            No improvement (5.0287), counter 1/5
    Epoch [18/50], Train Losses: mse: 19.8355, mae: 2.4833, huber: 2.0618, swd: 2.3940, ept: 242.3491
    Epoch [18/50], Val Losses: mse: 70.3259, mae: 5.4750, huber: 5.0183, swd: 4.0548, ept: 144.0603
    Epoch [18/50], Test Losses: mse: 52.2374, mae: 4.3071, huber: 3.8619, swd: 2.5247, ept: 183.4194
      Epoch 18 composite train-obj: 2.061806
            No improvement (5.0183), counter 2/5
    Epoch [19/50], Train Losses: mse: 19.1758, mae: 2.4348, huber: 2.0144, swd: 2.2893, ept: 245.5466
    Epoch [19/50], Val Losses: mse: 69.6347, mae: 5.4095, huber: 4.9557, swd: 4.2613, ept: 145.3145
    Epoch [19/50], Test Losses: mse: 51.7234, mae: 4.2419, huber: 3.8000, swd: 2.5364, ept: 186.3497
      Epoch 19 composite train-obj: 2.014389
            Val objective improved 4.9809 → 4.9557, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 18.5982, mae: 2.3871, huber: 1.9680, swd: 2.1965, ept: 248.6576
    Epoch [20/50], Val Losses: mse: 70.0699, mae: 5.4305, huber: 4.9747, swd: 4.0200, ept: 147.1466
    Epoch [20/50], Test Losses: mse: 52.0284, mae: 4.2633, huber: 3.8201, swd: 2.5141, ept: 188.8017
      Epoch 20 composite train-obj: 1.968038
            No improvement (4.9747), counter 1/5
    Epoch [21/50], Train Losses: mse: 18.1523, mae: 2.3498, huber: 1.9316, swd: 2.1108, ept: 250.9758
    Epoch [21/50], Val Losses: mse: 69.2608, mae: 5.3883, huber: 4.9344, swd: 4.2809, ept: 149.4451
    Epoch [21/50], Test Losses: mse: 50.9117, mae: 4.1911, huber: 3.7498, swd: 2.3364, ept: 188.8412
      Epoch 21 composite train-obj: 1.931581
            Val objective improved 4.9557 → 4.9344, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 17.5868, mae: 2.3039, huber: 1.8872, swd: 2.0439, ept: 253.5553
    Epoch [22/50], Val Losses: mse: 67.9559, mae: 5.3405, huber: 4.8873, swd: 4.4265, ept: 149.4735
    Epoch [22/50], Test Losses: mse: 50.3843, mae: 4.1969, huber: 3.7552, swd: 2.4580, ept: 189.6465
      Epoch 22 composite train-obj: 1.887200
            Val objective improved 4.9344 → 4.8873, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 17.0589, mae: 2.2598, huber: 1.8443, swd: 1.9597, ept: 256.0477
    Epoch [23/50], Val Losses: mse: 67.7357, mae: 5.3209, huber: 4.8669, swd: 4.2507, ept: 149.8344
    Epoch [23/50], Test Losses: mse: 51.1750, mae: 4.2197, huber: 3.7787, swd: 2.4771, ept: 189.9685
      Epoch 23 composite train-obj: 1.844321
            Val objective improved 4.8873 → 4.8669, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 16.6597, mae: 2.2262, huber: 1.8118, swd: 1.8727, ept: 258.2243
    Epoch [24/50], Val Losses: mse: 68.5593, mae: 5.3513, huber: 4.8970, swd: 4.2474, ept: 150.3705
    Epoch [24/50], Test Losses: mse: 50.4412, mae: 4.1819, huber: 3.7410, swd: 2.3270, ept: 193.1499
      Epoch 24 composite train-obj: 1.811760
            No improvement (4.8970), counter 1/5
    Epoch [25/50], Train Losses: mse: 16.3595, mae: 2.2041, huber: 1.7901, swd: 1.8531, ept: 259.7297
    Epoch [25/50], Val Losses: mse: 68.2922, mae: 5.3339, huber: 4.8800, swd: 4.2608, ept: 149.3203
    Epoch [25/50], Test Losses: mse: 50.8198, mae: 4.1625, huber: 3.7220, swd: 2.2722, ept: 194.6660
      Epoch 25 composite train-obj: 1.790107
            No improvement (4.8800), counter 2/5
    Epoch [26/50], Train Losses: mse: 15.9817, mae: 2.1735, huber: 1.7603, swd: 1.7998, ept: 261.1192
    Epoch [26/50], Val Losses: mse: 67.7558, mae: 5.2997, huber: 4.8474, swd: 4.1527, ept: 151.7281
    Epoch [26/50], Test Losses: mse: 50.8512, mae: 4.1424, huber: 3.7041, swd: 2.3043, ept: 195.1483
      Epoch 26 composite train-obj: 1.760331
            Val objective improved 4.8669 → 4.8474, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 15.5450, mae: 2.1309, huber: 1.7195, swd: 1.7277, ept: 263.3707
    Epoch [27/50], Val Losses: mse: 68.1339, mae: 5.3051, huber: 4.8529, swd: 4.2248, ept: 151.8217
    Epoch [27/50], Test Losses: mse: 50.1986, mae: 4.1163, huber: 3.6783, swd: 2.1696, ept: 194.8104
      Epoch 27 composite train-obj: 1.719531
            No improvement (4.8529), counter 1/5
    Epoch [28/50], Train Losses: mse: 15.2824, mae: 2.1128, huber: 1.7017, swd: 1.6990, ept: 264.2923
    Epoch [28/50], Val Losses: mse: 66.8521, mae: 5.2910, huber: 4.8397, swd: 4.5526, ept: 152.0506
    Epoch [28/50], Test Losses: mse: 49.5492, mae: 4.0928, huber: 3.6555, swd: 2.4009, ept: 195.7368
      Epoch 28 composite train-obj: 1.701747
            Val objective improved 4.8474 → 4.8397, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 14.9246, mae: 2.0840, huber: 1.6739, swd: 1.6465, ept: 265.9526
    Epoch [29/50], Val Losses: mse: 67.0360, mae: 5.2765, huber: 4.8245, swd: 4.4566, ept: 151.4991
    Epoch [29/50], Test Losses: mse: 49.9112, mae: 4.1134, huber: 3.6750, swd: 2.2669, ept: 198.1958
      Epoch 29 composite train-obj: 1.673907
            Val objective improved 4.8397 → 4.8245, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 14.7303, mae: 2.0645, huber: 1.6553, swd: 1.6198, ept: 266.9467
    Epoch [30/50], Val Losses: mse: 68.4398, mae: 5.3085, huber: 4.8576, swd: 4.2102, ept: 152.0176
    Epoch [30/50], Test Losses: mse: 51.5613, mae: 4.1556, huber: 3.7187, swd: 2.2097, ept: 198.7415
      Epoch 30 composite train-obj: 1.655315
            No improvement (4.8576), counter 1/5
    Epoch [31/50], Train Losses: mse: 14.4597, mae: 2.0481, huber: 1.6390, swd: 1.5911, ept: 268.1572
    Epoch [31/50], Val Losses: mse: 68.4289, mae: 5.3266, huber: 4.8746, swd: 4.2322, ept: 151.6646
    Epoch [31/50], Test Losses: mse: 50.0007, mae: 4.0869, huber: 3.6500, swd: 2.1770, ept: 199.6748
      Epoch 31 composite train-obj: 1.639018
            No improvement (4.8746), counter 2/5
    Epoch [32/50], Train Losses: mse: 14.2617, mae: 2.0240, huber: 1.6162, swd: 1.5513, ept: 269.4901
    Epoch [32/50], Val Losses: mse: 67.7900, mae: 5.3027, huber: 4.8506, swd: 4.4609, ept: 149.8217
    Epoch [32/50], Test Losses: mse: 50.4037, mae: 4.1005, huber: 3.6630, swd: 2.1900, ept: 198.8976
      Epoch 32 composite train-obj: 1.616190
            No improvement (4.8506), counter 3/5
    Epoch [33/50], Train Losses: mse: 13.8695, mae: 1.9908, huber: 1.5842, swd: 1.4963, ept: 270.7393
    Epoch [33/50], Val Losses: mse: 67.9478, mae: 5.2947, huber: 4.8434, swd: 4.2118, ept: 152.8919
    Epoch [33/50], Test Losses: mse: 49.8036, mae: 4.0847, huber: 3.6480, swd: 2.1617, ept: 197.9071
      Epoch 33 composite train-obj: 1.584152
            No improvement (4.8434), counter 4/5
    Epoch [34/50], Train Losses: mse: 13.6321, mae: 1.9762, huber: 1.5696, swd: 1.4822, ept: 271.8441
    Epoch [34/50], Val Losses: mse: 67.7360, mae: 5.2897, huber: 4.8386, swd: 4.2845, ept: 152.8622
    Epoch [34/50], Test Losses: mse: 50.0071, mae: 4.0858, huber: 3.6498, swd: 2.2001, ept: 199.8569
      Epoch 34 composite train-obj: 1.569563
    Epoch [34/50], Test Losses: mse: 49.9112, mae: 4.1134, huber: 3.6750, swd: 2.2669, ept: 198.1958
    Best round's Test MSE: 49.9112, MAE: 4.1134, SWD: 2.2669
    Best round's Validation MSE: 67.0360, MAE: 5.2765, SWD: 4.4566
    Best round's Test verification MSE : 49.9112, MAE: 4.1134, SWD: 2.2669
    Time taken: 148.32 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.0794, mae: 6.2341, huber: 5.7555, swd: 13.9389, ept: 50.6529
    Epoch [1/50], Val Losses: mse: 72.5505, mae: 6.2010, huber: 5.7263, swd: 14.2392, ept: 60.7823
    Epoch [1/50], Test Losses: mse: 60.0182, mae: 5.4984, huber: 5.0271, swd: 17.6846, ept: 63.7604
      Epoch 1 composite train-obj: 5.755451
            Val objective improved inf → 5.7263, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 53.8372, mae: 5.1189, huber: 4.6498, swd: 13.1590, ept: 96.0946
    Epoch [2/50], Val Losses: mse: 64.9851, mae: 5.8005, huber: 5.3270, swd: 13.8333, ept: 82.8138
    Epoch [2/50], Test Losses: mse: 53.3856, mae: 5.0652, huber: 4.5972, swd: 14.0840, ept: 93.3494
      Epoch 2 composite train-obj: 4.649757
            Val objective improved 5.7263 → 5.3270, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 45.6358, mae: 4.5094, huber: 4.0482, swd: 10.4211, ept: 127.8166
    Epoch [3/50], Val Losses: mse: 64.6444, mae: 5.6816, huber: 5.2095, swd: 11.2608, ept: 99.3711
    Epoch [3/50], Test Losses: mse: 51.9937, mae: 4.8902, huber: 4.4243, swd: 11.3835, ept: 108.4254
      Epoch 3 composite train-obj: 4.048173
            Val objective improved 5.3270 → 5.2095, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 41.2230, mae: 4.1460, huber: 3.6908, swd: 8.6296, ept: 146.1179
    Epoch [4/50], Val Losses: mse: 65.3118, mae: 5.6202, huber: 5.1518, swd: 9.7180, ept: 109.1520
    Epoch [4/50], Test Losses: mse: 51.8251, mae: 4.7969, huber: 4.3339, swd: 9.3048, ept: 122.6519
      Epoch 4 composite train-obj: 3.690833
            Val objective improved 5.2095 → 5.1518, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 37.8613, mae: 3.8817, huber: 3.4313, swd: 7.3804, ept: 158.6832
    Epoch [5/50], Val Losses: mse: 65.4336, mae: 5.5347, huber: 5.0694, swd: 8.7981, ept: 112.4875
    Epoch [5/50], Test Losses: mse: 52.6164, mae: 4.7401, huber: 4.2817, swd: 8.0420, ept: 129.1314
      Epoch 5 composite train-obj: 3.431305
            Val objective improved 5.1518 → 5.0694, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 35.2491, mae: 3.6740, huber: 3.2277, swd: 6.3682, ept: 170.1020
    Epoch [6/50], Val Losses: mse: 66.8358, mae: 5.5636, huber: 5.0993, swd: 7.7088, ept: 119.1134
    Epoch [6/50], Test Losses: mse: 53.1928, mae: 4.7157, huber: 4.2582, swd: 6.8574, ept: 138.8072
      Epoch 6 composite train-obj: 3.227658
            No improvement (5.0993), counter 1/5
    Epoch [7/50], Train Losses: mse: 32.8274, mae: 3.4947, huber: 3.0516, swd: 5.6057, ept: 179.7614
    Epoch [7/50], Val Losses: mse: 67.2738, mae: 5.5610, huber: 5.0976, swd: 7.0955, ept: 124.0644
    Epoch [7/50], Test Losses: mse: 52.7165, mae: 4.6326, huber: 4.1762, swd: 5.6814, ept: 146.3617
      Epoch 7 composite train-obj: 3.051636
            No improvement (5.0976), counter 2/5
    Epoch [8/50], Train Losses: mse: 30.8993, mae: 3.3473, huber: 2.9068, swd: 4.9508, ept: 188.7250
    Epoch [8/50], Val Losses: mse: 69.4334, mae: 5.6041, huber: 5.1418, swd: 5.9232, ept: 127.5850
    Epoch [8/50], Test Losses: mse: 54.3875, mae: 4.6616, huber: 4.2083, swd: 5.1639, ept: 152.1712
      Epoch 8 composite train-obj: 2.906832
            No improvement (5.1418), counter 3/5
    Epoch [9/50], Train Losses: mse: 29.0916, mae: 3.2082, huber: 2.7703, swd: 4.4176, ept: 196.4057
    Epoch [9/50], Val Losses: mse: 70.1408, mae: 5.5978, huber: 5.1351, swd: 5.4359, ept: 131.8571
    Epoch [9/50], Test Losses: mse: 54.0036, mae: 4.5843, huber: 4.1315, swd: 4.2848, ept: 157.8827
      Epoch 9 composite train-obj: 2.770326
            No improvement (5.1351), counter 4/5
    Epoch [10/50], Train Losses: mse: 27.6115, mae: 3.0851, huber: 2.6500, swd: 3.9540, ept: 203.8833
    Epoch [10/50], Val Losses: mse: 70.1528, mae: 5.5683, huber: 5.1076, swd: 5.1927, ept: 132.2911
    Epoch [10/50], Test Losses: mse: 53.9045, mae: 4.5376, huber: 4.0858, swd: 3.9616, ept: 163.1025
      Epoch 10 composite train-obj: 2.649962
    Epoch [10/50], Test Losses: mse: 52.6164, mae: 4.7401, huber: 4.2817, swd: 8.0420, ept: 129.1314
    Best round's Test MSE: 52.6164, MAE: 4.7401, SWD: 8.0420
    Best round's Validation MSE: 65.4336, MAE: 5.5347, SWD: 8.7981
    Best round's Test verification MSE : 52.6164, MAE: 4.7401, SWD: 8.0420
    Time taken: 49.69 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.1027, mae: 5.9612, huber: 5.4844, swd: 15.0038, ept: 59.1109
    Epoch [1/50], Val Losses: mse: 67.2019, mae: 5.9385, huber: 5.4648, swd: 14.8839, ept: 70.2909
    Epoch [1/50], Test Losses: mse: 56.0579, mae: 5.2431, huber: 4.7734, swd: 16.1188, ept: 74.8450
      Epoch 1 composite train-obj: 5.484437
            Val objective improved inf → 5.4648, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.4613, mae: 4.7786, huber: 4.3141, swd: 11.9166, ept: 113.9408
    Epoch [2/50], Val Losses: mse: 62.3009, mae: 5.6047, huber: 5.1347, swd: 13.4873, ept: 93.1741
    Epoch [2/50], Test Losses: mse: 50.7001, mae: 4.8538, huber: 4.3898, swd: 13.4808, ept: 104.2611
      Epoch 2 composite train-obj: 4.314147
            Val objective improved 5.4648 → 5.1347, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 43.6532, mae: 4.3165, huber: 3.8594, swd: 9.7264, ept: 139.0612
    Epoch [3/50], Val Losses: mse: 65.6515, mae: 5.6662, huber: 5.1985, swd: 10.6797, ept: 100.9705
    Epoch [3/50], Test Losses: mse: 51.4095, mae: 4.7635, huber: 4.3031, swd: 10.2843, ept: 113.0305
      Epoch 3 composite train-obj: 3.859418
            No improvement (5.1985), counter 1/5
    Epoch [4/50], Train Losses: mse: 40.0913, mae: 4.0168, huber: 3.5654, swd: 8.1157, ept: 154.2070
    Epoch [4/50], Val Losses: mse: 65.2201, mae: 5.5720, huber: 5.1063, swd: 9.6402, ept: 110.7124
    Epoch [4/50], Test Losses: mse: 51.4184, mae: 4.7155, huber: 4.2570, swd: 9.0216, ept: 126.4672
      Epoch 4 composite train-obj: 3.565411
            Val objective improved 5.1347 → 5.1063, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 37.1021, mae: 3.7880, huber: 3.3409, swd: 7.1186, ept: 165.6944
    Epoch [5/50], Val Losses: mse: 65.4481, mae: 5.5545, huber: 5.0906, swd: 8.8352, ept: 116.7531
    Epoch [5/50], Test Losses: mse: 50.2109, mae: 4.6112, huber: 4.1555, swd: 8.1986, ept: 131.2043
      Epoch 5 composite train-obj: 3.340878
            Val objective improved 5.1063 → 5.0906, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 34.5598, mae: 3.6004, huber: 3.1566, swd: 6.3341, ept: 175.3393
    Epoch [6/50], Val Losses: mse: 65.6166, mae: 5.4836, huber: 5.0210, swd: 8.1068, ept: 124.2113
    Epoch [6/50], Test Losses: mse: 51.0124, mae: 4.5698, huber: 4.1154, swd: 6.8810, ept: 142.5613
      Epoch 6 composite train-obj: 3.156612
            Val objective improved 5.0906 → 5.0210, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 32.5027, mae: 3.4466, huber: 3.0058, swd: 5.6748, ept: 184.0202
    Epoch [7/50], Val Losses: mse: 66.1306, mae: 5.5062, huber: 5.0443, swd: 7.6626, ept: 127.9605
    Epoch [7/50], Test Losses: mse: 50.2740, mae: 4.4881, huber: 4.0360, swd: 5.8435, ept: 150.8048
      Epoch 7 composite train-obj: 3.005794
            No improvement (5.0443), counter 1/5
    Epoch [8/50], Train Losses: mse: 30.6106, mae: 3.3112, huber: 2.8730, swd: 5.1714, ept: 191.2775
    Epoch [8/50], Val Losses: mse: 68.7909, mae: 5.5635, huber: 5.1009, swd: 6.4726, ept: 130.5172
    Epoch [8/50], Test Losses: mse: 51.4041, mae: 4.4947, huber: 4.0415, swd: 4.8941, ept: 157.2807
      Epoch 8 composite train-obj: 2.872962
            No improvement (5.1009), counter 2/5
    Epoch [9/50], Train Losses: mse: 29.1175, mae: 3.1985, huber: 2.7623, swd: 4.6829, ept: 198.2135
    Epoch [9/50], Val Losses: mse: 68.3580, mae: 5.5570, huber: 5.0946, swd: 6.5154, ept: 132.9620
    Epoch [9/50], Test Losses: mse: 51.1133, mae: 4.4682, huber: 4.0169, swd: 4.7830, ept: 161.1509
      Epoch 9 composite train-obj: 2.762318
            No improvement (5.0946), counter 3/5
    Epoch [10/50], Train Losses: mse: 27.7506, mae: 3.0960, huber: 2.6617, swd: 4.2865, ept: 205.2825
    Epoch [10/50], Val Losses: mse: 69.4037, mae: 5.5754, huber: 5.1147, swd: 5.9237, ept: 136.1133
    Epoch [10/50], Test Losses: mse: 50.9728, mae: 4.4364, huber: 3.9865, swd: 4.6216, ept: 164.3397
      Epoch 10 composite train-obj: 2.661670
            No improvement (5.1147), counter 4/5
    Epoch [11/50], Train Losses: mse: 26.6229, mae: 3.0063, huber: 2.5737, swd: 3.9787, ept: 211.2658
    Epoch [11/50], Val Losses: mse: 69.3045, mae: 5.5369, huber: 5.0761, swd: 5.6284, ept: 138.6848
    Epoch [11/50], Test Losses: mse: 50.9447, mae: 4.4041, huber: 3.9548, swd: 4.5490, ept: 167.3909
      Epoch 11 composite train-obj: 2.573743
    Epoch [11/50], Test Losses: mse: 51.0124, mae: 4.5698, huber: 4.1154, swd: 6.8810, ept: 142.5613
    Best round's Test MSE: 51.0124, MAE: 4.5698, SWD: 6.8810
    Best round's Validation MSE: 65.6166, MAE: 5.4836, SWD: 8.1068
    Best round's Test verification MSE : 51.0124, MAE: 4.5698, SWD: 6.8810
    Time taken: 57.07 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred336_20250514_1915)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 51.1800 ± 1.1107
      mae: 4.4744 ± 0.2646
      huber: 4.0240 ± 0.2560
      swd: 5.7300 ± 2.4942
      ept: 156.6295 ± 29.8988
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 66.0287 ± 0.7161
      mae: 5.4316 ± 0.1116
      huber: 4.9716 ± 0.1059
      swd: 7.1205 ± 1.9046
      ept: 129.3993 ± 16.3435
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 255.21 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred336_20250514_1915
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### huber + 0.5 SWD


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.5, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 89.1695, mae: 7.0887, huber: 6.6066, swd: 1.5919, ept: 30.0389
    Epoch [1/50], Val Losses: mse: 89.7280, mae: 6.8923, huber: 6.4136, swd: 1.2067, ept: 37.3966
    Epoch [1/50], Test Losses: mse: 77.1031, mae: 6.1700, huber: 5.6951, swd: 1.1021, ept: 42.4675
      Epoch 1 composite train-obj: 7.402524
            Val objective improved inf → 7.0169, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 79.3025, mae: 5.8994, huber: 5.4252, swd: 0.9485, ept: 67.7435
    Epoch [2/50], Val Losses: mse: 96.2485, mae: 6.4369, huber: 5.9624, swd: 1.0724, ept: 61.8413
    Epoch [2/50], Test Losses: mse: 82.7252, mae: 5.6674, huber: 5.1968, swd: 0.9315, ept: 63.4326
      Epoch 2 composite train-obj: 5.899445
            Val objective improved 7.0169 → 6.4986, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 76.5319, mae: 5.3216, huber: 4.8533, swd: 0.8416, ept: 90.6628
    Epoch [3/50], Val Losses: mse: 91.7520, mae: 6.2154, huber: 5.7418, swd: 1.0734, ept: 72.7391
    Epoch [3/50], Test Losses: mse: 78.9112, mae: 5.4154, huber: 4.9472, swd: 0.8974, ept: 77.4558
      Epoch 3 composite train-obj: 5.274123
            Val objective improved 6.4986 → 6.2785, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 70.1097, mae: 4.9226, huber: 4.4593, swd: 0.7932, ept: 108.8591
    Epoch [4/50], Val Losses: mse: 90.8154, mae: 6.1218, huber: 5.6502, swd: 1.0285, ept: 84.6023
    Epoch [4/50], Test Losses: mse: 76.6089, mae: 5.2359, huber: 4.7703, swd: 0.8363, ept: 92.1317
      Epoch 4 composite train-obj: 4.855952
            Val objective improved 6.2785 → 6.1644, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 65.0386, mae: 4.6334, huber: 4.1744, swd: 0.7650, ept: 123.5829
    Epoch [5/50], Val Losses: mse: 88.2401, mae: 6.0393, huber: 5.5684, swd: 1.0954, ept: 92.0678
    Epoch [5/50], Test Losses: mse: 74.5624, mae: 5.1419, huber: 4.6782, swd: 0.8027, ept: 100.5758
      Epoch 5 composite train-obj: 4.556889
            Val objective improved 6.1644 → 6.1161, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 60.9224, mae: 4.4022, huber: 3.9469, swd: 0.7344, ept: 133.5071
    Epoch [6/50], Val Losses: mse: 87.6074, mae: 6.0546, huber: 5.5855, swd: 1.0933, ept: 94.2677
    Epoch [6/50], Test Losses: mse: 71.6663, mae: 5.0402, huber: 4.5794, swd: 0.8963, ept: 108.6977
      Epoch 6 composite train-obj: 4.314076
            No improvement (6.1321), counter 1/5
    Epoch [7/50], Train Losses: mse: 57.3336, mae: 4.2107, huber: 3.7586, swd: 0.7176, ept: 142.4212
    Epoch [7/50], Val Losses: mse: 87.3000, mae: 6.0200, huber: 5.5521, swd: 1.1376, ept: 99.2853
    Epoch [7/50], Test Losses: mse: 71.1421, mae: 4.9870, huber: 4.5272, swd: 0.8954, ept: 115.7660
      Epoch 7 composite train-obj: 4.117346
            No improvement (6.1209), counter 2/5
    Epoch [8/50], Train Losses: mse: 54.2379, mae: 4.0487, huber: 3.5994, swd: 0.7051, ept: 149.5103
    Epoch [8/50], Val Losses: mse: 86.5014, mae: 6.0022, huber: 5.5346, swd: 1.2065, ept: 104.8006
    Epoch [8/50], Test Losses: mse: 70.9912, mae: 4.9996, huber: 4.5389, swd: 0.9429, ept: 121.2457
      Epoch 8 composite train-obj: 3.951909
            No improvement (6.1378), counter 3/5
    Epoch [9/50], Train Losses: mse: 51.5487, mae: 3.9125, huber: 3.4653, swd: 0.6952, ept: 155.4478
    Epoch [9/50], Val Losses: mse: 85.3949, mae: 5.9711, huber: 5.5047, swd: 1.2499, ept: 110.9707
    Epoch [9/50], Test Losses: mse: 69.9660, mae: 4.9474, huber: 4.4889, swd: 0.9199, ept: 127.4383
      Epoch 9 composite train-obj: 3.812944
            No improvement (6.1296), counter 4/5
    Epoch [10/50], Train Losses: mse: 49.1318, mae: 3.7879, huber: 3.3430, swd: 0.6930, ept: 161.2139
    Epoch [10/50], Val Losses: mse: 84.8777, mae: 5.9442, huber: 5.4789, swd: 1.3299, ept: 113.3233
    Epoch [10/50], Test Losses: mse: 69.3896, mae: 4.9195, huber: 4.4620, swd: 0.9333, ept: 130.1323
      Epoch 10 composite train-obj: 3.689517
    Epoch [10/50], Test Losses: mse: 74.5624, mae: 5.1419, huber: 4.6782, swd: 0.8027, ept: 100.5758
    Best round's Test MSE: 74.5624, MAE: 5.1419, SWD: 0.8027
    Best round's Validation MSE: 88.2401, MAE: 6.0393, SWD: 1.0954
    Best round's Test verification MSE : 74.5624, MAE: 5.1419, SWD: 0.8027
    Time taken: 48.60 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 90.9562, mae: 7.2049, huber: 6.7219, swd: 1.7247, ept: 24.8655
    Epoch [1/50], Val Losses: mse: 91.1334, mae: 7.0714, huber: 6.5910, swd: 1.2445, ept: 29.6011
    Epoch [1/50], Test Losses: mse: 79.2658, mae: 6.3182, huber: 5.8418, swd: 1.0752, ept: 34.0033
      Epoch 1 composite train-obj: 7.584264
            Val objective improved inf → 7.2133, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 73.9544, mae: 6.0827, huber: 5.6067, swd: 1.0121, ept: 55.4501
    Epoch [2/50], Val Losses: mse: 88.0693, mae: 6.6968, huber: 6.2197, swd: 1.2274, ept: 50.0840
    Epoch [2/50], Test Losses: mse: 74.9933, mae: 5.8828, huber: 5.4097, swd: 0.9507, ept: 55.8066
      Epoch 2 composite train-obj: 6.112771
            Val objective improved 7.2133 → 6.8334, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 65.4775, mae: 5.4561, huber: 4.9860, swd: 0.9025, ept: 79.7340
    Epoch [3/50], Val Losses: mse: 84.1222, mae: 6.4301, huber: 5.9549, swd: 1.2733, ept: 63.1408
    Epoch [3/50], Test Losses: mse: 70.3153, mae: 5.6143, huber: 5.1436, swd: 0.9879, ept: 69.1149
      Epoch 3 composite train-obj: 5.437205
            Val objective improved 6.8334 → 6.5915, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 59.4097, mae: 5.0223, huber: 4.5571, swd: 0.8434, ept: 97.5302
    Epoch [4/50], Val Losses: mse: 82.1731, mae: 6.2536, huber: 5.7810, swd: 1.2471, ept: 76.4910
    Epoch [4/50], Test Losses: mse: 68.0141, mae: 5.3918, huber: 4.9245, swd: 0.9379, ept: 83.0466
      Epoch 4 composite train-obj: 4.978786
            Val objective improved 6.5915 → 6.4046, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 55.0337, mae: 4.7109, huber: 4.2495, swd: 0.8206, ept: 111.5462
    Epoch [5/50], Val Losses: mse: 81.8170, mae: 6.1754, huber: 5.7047, swd: 1.3764, ept: 83.5284
    Epoch [5/50], Test Losses: mse: 67.0665, mae: 5.2735, huber: 4.8096, swd: 1.0533, ept: 91.6510
      Epoch 5 composite train-obj: 4.659826
            Val objective improved 6.4046 → 6.3929, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 51.7631, mae: 4.4615, huber: 4.0039, swd: 0.7831, ept: 124.9589
    Epoch [6/50], Val Losses: mse: 82.9624, mae: 6.1599, huber: 5.6901, swd: 1.2284, ept: 89.7508
    Epoch [6/50], Test Losses: mse: 67.3971, mae: 5.2255, huber: 4.7620, swd: 0.9186, ept: 98.5762
      Epoch 6 composite train-obj: 4.395432
            Val objective improved 6.3929 → 6.3043, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 48.9723, mae: 4.2602, huber: 3.8058, swd: 0.7546, ept: 136.2019
    Epoch [7/50], Val Losses: mse: 82.3858, mae: 6.1273, huber: 5.6578, swd: 1.2550, ept: 96.2850
    Epoch [7/50], Test Losses: mse: 66.8729, mae: 5.1671, huber: 4.7040, swd: 0.9576, ept: 106.6521
      Epoch 7 composite train-obj: 4.183150
            Val objective improved 6.3043 → 6.2853, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 46.6234, mae: 4.0919, huber: 3.6401, swd: 0.7442, ept: 145.9782
    Epoch [8/50], Val Losses: mse: 81.7417, mae: 6.0635, huber: 5.5957, swd: 1.3215, ept: 99.8875
    Epoch [8/50], Test Losses: mse: 66.8854, mae: 5.1258, huber: 4.6649, swd: 0.9529, ept: 114.1045
      Epoch 8 composite train-obj: 4.012211
            Val objective improved 6.2853 → 6.2564, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 44.2479, mae: 3.9303, huber: 3.4814, swd: 0.7191, ept: 155.7464
    Epoch [9/50], Val Losses: mse: 81.7939, mae: 6.0532, huber: 5.5856, swd: 1.3886, ept: 106.6911
    Epoch [9/50], Test Losses: mse: 66.6899, mae: 5.0931, huber: 4.6324, swd: 1.0061, ept: 120.9276
      Epoch 9 composite train-obj: 3.840900
            No improvement (6.2799), counter 1/5
    Epoch [10/50], Train Losses: mse: 42.0425, mae: 3.7779, huber: 3.3317, swd: 0.6959, ept: 164.9477
    Epoch [10/50], Val Losses: mse: 81.8162, mae: 6.0327, huber: 5.5661, swd: 1.3926, ept: 108.6376
    Epoch [10/50], Test Losses: mse: 65.1947, mae: 4.9888, huber: 4.5306, swd: 0.9805, ept: 127.0371
      Epoch 10 composite train-obj: 3.679687
            No improvement (6.2624), counter 2/5
    Epoch [11/50], Train Losses: mse: 40.0971, mae: 3.6598, huber: 3.2153, swd: 0.7064, ept: 172.1231
    Epoch [11/50], Val Losses: mse: 80.7025, mae: 5.9904, huber: 5.5242, swd: 1.5204, ept: 110.0004
    Epoch [11/50], Test Losses: mse: 64.7770, mae: 4.9395, huber: 4.4821, swd: 0.9870, ept: 134.1976
      Epoch 11 composite train-obj: 3.568488
            No improvement (6.2844), counter 3/5
    Epoch [12/50], Train Losses: mse: 38.1014, mae: 3.5242, huber: 3.0826, swd: 0.6810, ept: 179.4484
    Epoch [12/50], Val Losses: mse: 80.0211, mae: 5.9485, huber: 5.4833, swd: 1.5087, ept: 117.2699
    Epoch [12/50], Test Losses: mse: 63.6265, mae: 4.8890, huber: 4.4326, swd: 1.0596, ept: 139.9729
      Epoch 12 composite train-obj: 3.423072
            Val objective improved 6.2564 → 6.2376, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 36.4293, mae: 3.4188, huber: 2.9789, swd: 0.6780, ept: 185.9134
    Epoch [13/50], Val Losses: mse: 80.9660, mae: 5.9585, huber: 5.4933, swd: 1.6202, ept: 116.9868
    Epoch [13/50], Test Losses: mse: 63.8233, mae: 4.8568, huber: 4.4013, swd: 1.0670, ept: 142.0346
      Epoch 13 composite train-obj: 3.317939
            No improvement (6.3034), counter 1/5
    Epoch [14/50], Train Losses: mse: 34.8229, mae: 3.3154, huber: 2.8773, swd: 0.6728, ept: 191.1524
    Epoch [14/50], Val Losses: mse: 79.6080, mae: 5.9189, huber: 5.4547, swd: 1.7106, ept: 120.4016
    Epoch [14/50], Test Losses: mse: 62.9338, mae: 4.8037, huber: 4.3502, swd: 1.0493, ept: 149.3783
      Epoch 14 composite train-obj: 3.213747
            No improvement (6.3100), counter 2/5
    Epoch [15/50], Train Losses: mse: 33.2221, mae: 3.2159, huber: 2.7798, swd: 0.6587, ept: 196.9613
    Epoch [15/50], Val Losses: mse: 79.3138, mae: 5.9091, huber: 5.4448, swd: 1.8771, ept: 121.5616
    Epoch [15/50], Test Losses: mse: 62.5157, mae: 4.8044, huber: 4.3495, swd: 1.0794, ept: 151.0948
      Epoch 15 composite train-obj: 3.109169
            No improvement (6.3834), counter 3/5
    Epoch [16/50], Train Losses: mse: 31.8039, mae: 3.1268, huber: 2.6924, swd: 0.6572, ept: 201.9430
    Epoch [16/50], Val Losses: mse: 79.2685, mae: 5.8887, huber: 5.4251, swd: 1.8157, ept: 125.6966
    Epoch [16/50], Test Losses: mse: 61.6922, mae: 4.7368, huber: 4.2842, swd: 1.0610, ept: 155.4947
      Epoch 16 composite train-obj: 3.020971
            No improvement (6.3330), counter 4/5
    Epoch [17/50], Train Losses: mse: 30.3095, mae: 3.0368, huber: 2.6039, swd: 0.6514, ept: 207.1878
    Epoch [17/50], Val Losses: mse: 78.4681, mae: 5.8320, huber: 5.3693, swd: 1.9566, ept: 129.9206
    Epoch [17/50], Test Losses: mse: 60.9362, mae: 4.7036, huber: 4.2514, swd: 1.1676, ept: 157.0253
      Epoch 17 composite train-obj: 2.929584
    Epoch [17/50], Test Losses: mse: 63.6265, mae: 4.8890, huber: 4.4326, swd: 1.0596, ept: 139.9729
    Best round's Test MSE: 63.6265, MAE: 4.8890, SWD: 1.0596
    Best round's Validation MSE: 80.0211, MAE: 5.9485, SWD: 1.5087
    Best round's Test verification MSE : 63.6265, MAE: 4.8890, SWD: 1.0596
    Time taken: 76.95 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 84.1140, mae: 6.7853, huber: 6.3046, swd: 1.5266, ept: 34.8903
    Epoch [1/50], Val Losses: mse: 88.0866, mae: 6.7570, huber: 6.2792, swd: 1.1915, ept: 39.8555
    Epoch [1/50], Test Losses: mse: 75.8623, mae: 6.0568, huber: 5.5821, swd: 0.9903, ept: 44.5158
      Epoch 1 composite train-obj: 7.067898
            Val objective improved inf → 6.8750, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 66.4877, mae: 5.5533, huber: 5.0821, swd: 0.9977, ept: 73.7639
    Epoch [2/50], Val Losses: mse: 80.9615, mae: 6.2770, huber: 5.8031, swd: 1.2737, ept: 66.1083
    Epoch [2/50], Test Losses: mse: 69.1300, mae: 5.5636, huber: 5.0931, swd: 1.0599, ept: 67.2229
      Epoch 2 composite train-obj: 5.580952
            Val objective improved 6.8750 → 6.4399, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 60.0384, mae: 4.9843, huber: 4.5198, swd: 0.9008, ept: 100.9632
    Epoch [3/50], Val Losses: mse: 81.2049, mae: 6.1081, huber: 5.6369, swd: 1.2303, ept: 78.7130
    Epoch [3/50], Test Losses: mse: 68.1726, mae: 5.2716, huber: 4.8061, swd: 0.9823, ept: 82.0861
      Epoch 3 composite train-obj: 4.970218
            Val objective improved 6.4399 → 6.2520, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 56.1557, mae: 4.6185, huber: 4.1592, swd: 0.8379, ept: 119.7881
    Epoch [4/50], Val Losses: mse: 81.4360, mae: 6.0247, huber: 5.5553, swd: 1.2303, ept: 86.3003
    Epoch [4/50], Test Losses: mse: 68.6865, mae: 5.1764, huber: 4.7127, swd: 0.9536, ept: 95.1026
      Epoch 4 composite train-obj: 4.578169
            Val objective improved 6.2520 → 6.1705, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 52.7456, mae: 4.3571, huber: 3.9019, swd: 0.8089, ept: 132.9175
    Epoch [5/50], Val Losses: mse: 82.2872, mae: 6.0149, huber: 5.5465, swd: 1.2341, ept: 95.2611
    Epoch [5/50], Test Losses: mse: 68.1241, mae: 5.0879, huber: 4.6263, swd: 0.9345, ept: 104.9781
      Epoch 5 composite train-obj: 4.306310
            Val objective improved 6.1705 → 6.1636, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 50.0383, mae: 4.1454, huber: 3.6940, swd: 0.7753, ept: 142.0217
    Epoch [6/50], Val Losses: mse: 80.9786, mae: 5.9420, huber: 5.4749, swd: 1.2833, ept: 100.7959
    Epoch [6/50], Test Losses: mse: 66.9659, mae: 5.0214, huber: 4.5610, swd: 0.9479, ept: 111.7966
      Epoch 6 composite train-obj: 4.081629
            Val objective improved 6.1636 → 6.1166, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 47.6471, mae: 3.9724, huber: 3.5243, swd: 0.7593, ept: 149.2773
    Epoch [7/50], Val Losses: mse: 81.6485, mae: 5.9147, huber: 5.4489, swd: 1.3207, ept: 105.6910
    Epoch [7/50], Test Losses: mse: 67.3063, mae: 4.9292, huber: 4.4712, swd: 0.9374, ept: 120.9572
      Epoch 7 composite train-obj: 3.903920
            Val objective improved 6.1166 → 6.1092, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 45.6536, mae: 3.8221, huber: 3.3768, swd: 0.7443, ept: 156.0753
    Epoch [8/50], Val Losses: mse: 82.7451, mae: 5.9199, huber: 5.4544, swd: 1.3447, ept: 110.8012
    Epoch [8/50], Test Losses: mse: 67.6901, mae: 4.9309, huber: 4.4729, swd: 1.0110, ept: 125.6068
      Epoch 8 composite train-obj: 3.748960
            No improvement (6.1268), counter 1/5
    Epoch [9/50], Train Losses: mse: 43.7385, mae: 3.6944, huber: 3.2513, swd: 0.7339, ept: 161.5161
    Epoch [9/50], Val Losses: mse: 81.8871, mae: 5.8905, huber: 5.4263, swd: 1.3538, ept: 112.5295
    Epoch [9/50], Test Losses: mse: 66.5804, mae: 4.8688, huber: 4.4134, swd: 1.0079, ept: 130.1381
      Epoch 9 composite train-obj: 3.618230
            Val objective improved 6.1092 → 6.1032, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 41.9437, mae: 3.5711, huber: 3.1303, swd: 0.7233, ept: 166.6622
    Epoch [10/50], Val Losses: mse: 80.9622, mae: 5.8347, huber: 5.3706, swd: 1.3849, ept: 116.5117
    Epoch [10/50], Test Losses: mse: 66.1164, mae: 4.8229, huber: 4.3685, swd: 1.0223, ept: 134.5028
      Epoch 10 composite train-obj: 3.491957
            Val objective improved 6.1032 → 6.0630, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 40.2563, mae: 3.4643, huber: 3.0256, swd: 0.7112, ept: 171.2744
    Epoch [11/50], Val Losses: mse: 81.3307, mae: 5.8405, huber: 5.3769, swd: 1.4742, ept: 120.3088
    Epoch [11/50], Test Losses: mse: 65.1215, mae: 4.7684, huber: 4.3147, swd: 1.0359, ept: 138.3793
      Epoch 11 composite train-obj: 3.381255
            No improvement (6.1139), counter 1/5
    Epoch [12/50], Train Losses: mse: 38.7141, mae: 3.3704, huber: 2.9331, swd: 0.7085, ept: 175.4527
    Epoch [12/50], Val Losses: mse: 81.9284, mae: 5.8707, huber: 5.4073, swd: 1.5254, ept: 121.3891
    Epoch [12/50], Test Losses: mse: 65.3415, mae: 4.7780, huber: 4.3243, swd: 1.0769, ept: 143.9496
      Epoch 12 composite train-obj: 3.287405
            No improvement (6.1701), counter 2/5
    Epoch [13/50], Train Losses: mse: 37.1634, mae: 3.2755, huber: 2.8399, swd: 0.6976, ept: 179.8464
    Epoch [13/50], Val Losses: mse: 81.2887, mae: 5.8350, huber: 5.3727, swd: 1.5590, ept: 124.5925
    Epoch [13/50], Test Losses: mse: 66.1129, mae: 4.8025, huber: 4.3490, swd: 1.1519, ept: 144.9340
      Epoch 13 composite train-obj: 3.188653
            No improvement (6.1522), counter 3/5
    Epoch [14/50], Train Losses: mse: 35.8597, mae: 3.1952, huber: 2.7611, swd: 0.6900, ept: 183.3758
    Epoch [14/50], Val Losses: mse: 80.8606, mae: 5.8189, huber: 5.3569, swd: 1.6334, ept: 124.6381
    Epoch [14/50], Test Losses: mse: 65.1703, mae: 4.7220, huber: 4.2716, swd: 1.1226, ept: 146.8568
      Epoch 14 composite train-obj: 3.106070
            No improvement (6.1736), counter 4/5
    Epoch [15/50], Train Losses: mse: 34.5269, mae: 3.1126, huber: 2.6800, swd: 0.6795, ept: 187.0196
    Epoch [15/50], Val Losses: mse: 80.3581, mae: 5.8185, huber: 5.3561, swd: 1.5915, ept: 128.3244
    Epoch [15/50], Test Losses: mse: 64.5999, mae: 4.7066, huber: 4.2557, swd: 1.1039, ept: 151.8437
      Epoch 15 composite train-obj: 3.019730
    Epoch [15/50], Test Losses: mse: 66.1164, mae: 4.8229, huber: 4.3685, swd: 1.0223, ept: 134.5028
    Best round's Test MSE: 66.1164, MAE: 4.8229, SWD: 1.0223
    Best round's Validation MSE: 80.9622, MAE: 5.8347, SWD: 1.3849
    Best round's Test verification MSE : 66.1164, MAE: 4.8229, SWD: 1.0223
    Time taken: 66.76 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred336_20250514_1919)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 68.1018 ± 4.6801
      mae: 4.9513 ± 0.1375
      huber: 4.4931 ± 0.1335
      swd: 0.9615 ± 0.1133
      ept: 125.0172 ± 17.4263
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 83.0745 ± 3.6728
      mae: 5.9408 ± 0.0837
      huber: 5.4741 ± 0.0810
      swd: 1.3297 ± 0.1732
      ept: 108.6165 ± 11.7058
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 192.41 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred336_20250514_1919
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE


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
    loss_backward_weights = [1.0, 0.0, 0.0, 0.0, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.8765, mae: 6.2210, huber: 5.7406, swd: 17.2746, ept: 46.3937
    Epoch [1/50], Val Losses: mse: 67.6510, mae: 6.2287, huber: 5.7502, swd: 19.2162, ept: 49.3527
    Epoch [1/50], Test Losses: mse: 56.8074, mae: 5.6656, huber: 5.1883, swd: 20.0845, ept: 48.3815
      Epoch 1 composite train-obj: 66.876468
            Val objective improved inf → 67.6510, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.4350, mae: 5.1964, huber: 4.7221, swd: 15.3577, ept: 82.2238
    Epoch [2/50], Val Losses: mse: 60.1212, mae: 5.7614, huber: 5.2848, swd: 16.9886, ept: 76.8009
    Epoch [2/50], Test Losses: mse: 51.2232, mae: 5.2523, huber: 4.7783, swd: 15.8813, ept: 83.5731
      Epoch 2 composite train-obj: 49.434963
            Val objective improved 67.6510 → 60.1212, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.6823, mae: 4.7153, huber: 4.2452, swd: 12.4160, ept: 110.0382
    Epoch [3/50], Val Losses: mse: 61.1396, mae: 5.7219, huber: 5.2471, swd: 13.7089, ept: 76.5665
    Epoch [3/50], Test Losses: mse: 51.8868, mae: 5.1805, huber: 4.7087, swd: 12.7937, ept: 82.9875
      Epoch 3 composite train-obj: 42.682310
            No improvement (61.1396), counter 1/5
    Epoch [4/50], Train Losses: mse: 38.7838, mae: 4.4271, huber: 3.9599, swd: 10.0676, ept: 126.6578
    Epoch [4/50], Val Losses: mse: 62.5454, mae: 5.7212, huber: 5.2484, swd: 11.4703, ept: 84.6762
    Epoch [4/50], Test Losses: mse: 53.1249, mae: 5.1745, huber: 4.7044, swd: 10.4217, ept: 95.0538
      Epoch 4 composite train-obj: 38.783802
            No improvement (62.5454), counter 2/5
    Epoch [5/50], Train Losses: mse: 35.6925, mae: 4.2002, huber: 3.7356, swd: 8.4960, ept: 141.2542
    Epoch [5/50], Val Losses: mse: 63.8065, mae: 5.7601, huber: 5.2866, swd: 9.5043, ept: 96.6930
    Epoch [5/50], Test Losses: mse: 52.4300, mae: 5.1080, huber: 4.6382, swd: 8.8760, ept: 106.8175
      Epoch 5 composite train-obj: 35.692454
            No improvement (63.8065), counter 3/5
    Epoch [6/50], Train Losses: mse: 33.0536, mae: 4.0061, huber: 3.5437, swd: 7.2437, ept: 152.8890
    Epoch [6/50], Val Losses: mse: 64.3459, mae: 5.7383, huber: 5.2660, swd: 8.1083, ept: 99.6048
    Epoch [6/50], Test Losses: mse: 52.6318, mae: 5.0408, huber: 4.5725, swd: 7.6665, ept: 108.0338
      Epoch 6 composite train-obj: 33.053550
            No improvement (64.3459), counter 4/5
    Epoch [7/50], Train Losses: mse: 30.6619, mae: 3.8246, huber: 3.3644, swd: 6.2749, ept: 164.9573
    Epoch [7/50], Val Losses: mse: 67.0985, mae: 5.8038, huber: 5.3327, swd: 6.8287, ept: 103.7298
    Epoch [7/50], Test Losses: mse: 54.5021, mae: 5.0552, huber: 4.5877, swd: 6.7335, ept: 116.4652
      Epoch 7 composite train-obj: 30.661914
    Epoch [7/50], Test Losses: mse: 51.2232, mae: 5.2523, huber: 4.7783, swd: 15.8813, ept: 83.5731
    Best round's Test MSE: 51.2232, MAE: 5.2523, SWD: 15.8813
    Best round's Validation MSE: 60.1212, MAE: 5.7614, SWD: 16.9886
    Best round's Test verification MSE : 51.2232, MAE: 5.2523, SWD: 15.8813
    Time taken: 31.21 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.2083, mae: 6.3939, huber: 5.9127, swd: 15.8760, ept: 40.3978
    Epoch [1/50], Val Losses: mse: 67.8311, mae: 6.2874, huber: 5.8085, swd: 20.5630, ept: 43.9818
    Epoch [1/50], Test Losses: mse: 58.1051, mae: 5.7293, huber: 5.2518, swd: 21.7828, ept: 42.8920
      Epoch 1 composite train-obj: 70.208253
            Val objective improved inf → 67.8311, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 51.4533, mae: 5.3348, huber: 4.8594, swd: 16.4647, ept: 72.8185
    Epoch [2/50], Val Losses: mse: 61.4728, mae: 5.8506, huber: 5.3740, swd: 17.3785, ept: 62.4157
    Epoch [2/50], Test Losses: mse: 53.3922, mae: 5.3540, huber: 4.8802, swd: 17.3729, ept: 67.0410
      Epoch 2 composite train-obj: 51.453286
            Val objective improved 67.8311 → 61.4728, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 44.0800, mae: 4.8133, huber: 4.3424, swd: 13.3906, ept: 100.7959
    Epoch [3/50], Val Losses: mse: 61.9467, mae: 5.8150, huber: 5.3386, swd: 14.4728, ept: 70.9098
    Epoch [3/50], Test Losses: mse: 52.5589, mae: 5.2674, huber: 4.7941, swd: 14.2709, ept: 76.4671
      Epoch 3 composite train-obj: 44.080011
            No improvement (61.9467), counter 1/5
    Epoch [4/50], Train Losses: mse: 39.8111, mae: 4.4984, huber: 4.0306, swd: 10.9897, ept: 119.1284
    Epoch [4/50], Val Losses: mse: 62.5483, mae: 5.7754, huber: 5.3007, swd: 11.6793, ept: 79.2173
    Epoch [4/50], Test Losses: mse: 52.5593, mae: 5.1734, huber: 4.7022, swd: 11.3792, ept: 89.0839
      Epoch 4 composite train-obj: 39.811068
            No improvement (62.5483), counter 2/5
    Epoch [5/50], Train Losses: mse: 36.5289, mae: 4.2605, huber: 3.7951, swd: 9.1031, ept: 132.8768
    Epoch [5/50], Val Losses: mse: 63.5016, mae: 5.7643, huber: 5.2904, swd: 9.9463, ept: 88.1324
    Epoch [5/50], Test Losses: mse: 53.7942, mae: 5.1801, huber: 4.7093, swd: 9.8100, ept: 97.6722
      Epoch 5 composite train-obj: 36.528892
            No improvement (63.5016), counter 3/5
    Epoch [6/50], Train Losses: mse: 33.5611, mae: 4.0415, huber: 3.5786, swd: 7.6239, ept: 145.9330
    Epoch [6/50], Val Losses: mse: 65.8287, mae: 5.8305, huber: 5.3564, swd: 8.5392, ept: 94.8580
    Epoch [6/50], Test Losses: mse: 54.2395, mae: 5.1123, huber: 4.6427, swd: 7.8788, ept: 106.2696
      Epoch 6 composite train-obj: 33.561121
            No improvement (65.8287), counter 4/5
    Epoch [7/50], Train Losses: mse: 30.8397, mae: 3.8435, huber: 3.3827, swd: 6.4480, ept: 158.2489
    Epoch [7/50], Val Losses: mse: 67.3312, mae: 5.8412, huber: 5.3688, swd: 7.1706, ept: 94.3842
    Epoch [7/50], Test Losses: mse: 54.8874, mae: 5.0933, huber: 4.6252, swd: 6.5563, ept: 107.3506
      Epoch 7 composite train-obj: 30.839671
    Epoch [7/50], Test Losses: mse: 53.3922, mae: 5.3540, huber: 4.8802, swd: 17.3729, ept: 67.0410
    Best round's Test MSE: 53.3922, MAE: 5.3540, SWD: 17.3729
    Best round's Validation MSE: 61.4728, MAE: 5.8506, SWD: 17.3785
    Best round's Test verification MSE : 53.3922, MAE: 5.3540, SWD: 17.3729
    Time taken: 31.67 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 64.9265, mae: 6.1360, huber: 5.6560, swd: 17.7875, ept: 47.2764
    Epoch [1/50], Val Losses: mse: 63.4037, mae: 6.0025, huber: 5.5257, swd: 19.4047, ept: 55.5157
    Epoch [1/50], Test Losses: mse: 54.2579, mae: 5.4765, huber: 5.0011, swd: 19.9589, ept: 56.8716
      Epoch 1 composite train-obj: 64.926511
            Val objective improved inf → 63.4037, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.9870, mae: 5.0171, huber: 4.5443, swd: 14.7065, ept: 91.4128
    Epoch [2/50], Val Losses: mse: 60.1310, mae: 5.7118, huber: 5.2362, swd: 15.3624, ept: 80.3786
    Epoch [2/50], Test Losses: mse: 52.0504, mae: 5.2246, huber: 4.7517, swd: 13.9340, ept: 86.8286
      Epoch 2 composite train-obj: 46.987042
            Val objective improved 63.4037 → 60.1310, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.9663, mae: 4.5802, huber: 4.1119, swd: 11.7744, ept: 117.6595
    Epoch [3/50], Val Losses: mse: 61.5183, mae: 5.6981, huber: 5.2240, swd: 12.8382, ept: 88.5692
    Epoch [3/50], Test Losses: mse: 51.6183, mae: 5.0997, huber: 4.6288, swd: 11.6663, ept: 101.2945
      Epoch 3 composite train-obj: 40.966314
            No improvement (61.5183), counter 1/5
    Epoch [4/50], Train Losses: mse: 37.0110, mae: 4.2870, huber: 3.8218, swd: 9.6357, ept: 135.2053
    Epoch [4/50], Val Losses: mse: 62.1311, mae: 5.6442, huber: 5.1720, swd: 10.0232, ept: 92.9521
    Epoch [4/50], Test Losses: mse: 52.7130, mae: 5.0876, huber: 4.6185, swd: 10.2745, ept: 104.2715
      Epoch 4 composite train-obj: 37.011047
            No improvement (62.1311), counter 2/5
    Epoch [5/50], Train Losses: mse: 33.9765, mae: 4.0624, huber: 3.5997, swd: 8.1447, ept: 149.3547
    Epoch [5/50], Val Losses: mse: 63.3289, mae: 5.7075, huber: 5.2356, swd: 9.1015, ept: 98.0489
    Epoch [5/50], Test Losses: mse: 52.1059, mae: 5.0442, huber: 4.5756, swd: 8.7832, ept: 111.6647
      Epoch 5 composite train-obj: 33.976474
            No improvement (63.3289), counter 3/5
    Epoch [6/50], Train Losses: mse: 31.3345, mae: 3.8703, huber: 3.4098, swd: 6.9743, ept: 161.2416
    Epoch [6/50], Val Losses: mse: 63.9749, mae: 5.6642, huber: 5.1933, swd: 7.7650, ept: 105.6434
    Epoch [6/50], Test Losses: mse: 52.6460, mae: 4.9939, huber: 4.5273, swd: 7.9343, ept: 119.1076
      Epoch 6 composite train-obj: 31.334452
            No improvement (63.9749), counter 4/5
    Epoch [7/50], Train Losses: mse: 29.0203, mae: 3.6940, huber: 3.2356, swd: 6.0576, ept: 173.1022
    Epoch [7/50], Val Losses: mse: 64.6321, mae: 5.6819, huber: 5.2114, swd: 7.4430, ept: 109.2743
    Epoch [7/50], Test Losses: mse: 51.7537, mae: 4.8779, huber: 4.4124, swd: 6.6014, ept: 122.3735
      Epoch 7 composite train-obj: 29.020349
    Epoch [7/50], Test Losses: mse: 52.0504, mae: 5.2246, huber: 4.7517, swd: 13.9340, ept: 86.8286
    Best round's Test MSE: 52.0504, MAE: 5.2246, SWD: 13.9340
    Best round's Validation MSE: 60.1310, MAE: 5.7118, SWD: 15.3624
    Best round's Test verification MSE : 52.0504, MAE: 5.2246, SWD: 13.9340
    Time taken: 31.23 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred336_20250514_1922)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 52.2219 ± 0.8938
      mae: 5.2770 ± 0.0556
      huber: 4.8034 ± 0.0554
      swd: 15.7294 ± 1.4080
      ept: 79.1476 ± 8.6632
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 60.5750 ± 0.6349
      mae: 5.7746 ± 0.0574
      huber: 5.2984 ± 0.0571
      swd: 16.5765 ± 0.8732
      ept: 73.1984 ± 7.7632
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 94.19 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred336_20250514_1922
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE + 0.5 SWD


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
    loss_backward_weights = [1.0, 0.0, 0.0, 0.5, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.5, 0.0]
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)

```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 69.3558, mae: 6.3330, huber: 5.8518, swd: 9.2611, ept: 41.5565
    Epoch [1/50], Val Losses: mse: 75.0004, mae: 6.5830, huber: 6.1028, swd: 8.4614, ept: 42.5782
    Epoch [1/50], Test Losses: mse: 61.6655, mae: 5.9316, huber: 5.4524, swd: 8.3852, ept: 40.6080
      Epoch 1 composite train-obj: 73.986289
            Val objective improved inf → 79.2311, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 51.5692, mae: 5.3265, huber: 4.8504, swd: 6.4345, ept: 73.0668
    Epoch [2/50], Val Losses: mse: 64.8960, mae: 5.9704, huber: 5.4927, swd: 7.7663, ept: 69.8731
    Epoch [2/50], Test Losses: mse: 54.7881, mae: 5.4387, huber: 4.9631, swd: 6.5283, ept: 73.0378
      Epoch 2 composite train-obj: 54.786427
            Val objective improved 79.2311 → 68.7791, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 43.6904, mae: 4.7962, huber: 4.3242, swd: 5.0235, ept: 100.2467
    Epoch [3/50], Val Losses: mse: 65.3376, mae: 5.9012, huber: 5.4255, swd: 6.6020, ept: 73.0498
    Epoch [3/50], Test Losses: mse: 55.7225, mae: 5.3883, huber: 4.9140, swd: 5.0854, ept: 79.0426
      Epoch 3 composite train-obj: 46.202127
            Val objective improved 68.7791 → 68.6386, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.3014, mae: 4.4820, huber: 4.0130, swd: 4.1006, ept: 120.3967
    Epoch [4/50], Val Losses: mse: 67.2632, mae: 5.9520, huber: 5.4762, swd: 5.5758, ept: 80.4223
    Epoch [4/50], Test Losses: mse: 56.8124, mae: 5.3991, huber: 4.9248, swd: 4.2388, ept: 91.7352
      Epoch 4 composite train-obj: 41.351651
            No improvement (70.0511), counter 1/5
    Epoch [5/50], Train Losses: mse: 35.8412, mae: 4.2316, huber: 3.7652, swd: 3.5165, ept: 137.1897
    Epoch [5/50], Val Losses: mse: 67.3815, mae: 5.9033, huber: 5.4287, swd: 4.7985, ept: 91.3412
    Epoch [5/50], Test Losses: mse: 55.6343, mae: 5.2434, huber: 4.7718, swd: 3.8089, ept: 106.2779
      Epoch 5 composite train-obj: 37.599416
            No improvement (69.7807), counter 2/5
    Epoch [6/50], Train Losses: mse: 33.0172, mae: 4.0235, huber: 3.5594, swd: 3.1250, ept: 151.2811
    Epoch [6/50], Val Losses: mse: 68.6403, mae: 5.9090, huber: 5.4353, swd: 4.1821, ept: 100.5200
    Epoch [6/50], Test Losses: mse: 55.6508, mae: 5.1380, huber: 4.6684, swd: 3.6461, ept: 110.7826
      Epoch 6 composite train-obj: 34.579713
            No improvement (70.7313), counter 3/5
    Epoch [7/50], Train Losses: mse: 30.4874, mae: 3.8326, huber: 3.3708, swd: 2.7994, ept: 164.5021
    Epoch [7/50], Val Losses: mse: 71.5535, mae: 6.0134, huber: 5.5396, swd: 3.9354, ept: 98.3580
    Epoch [7/50], Test Losses: mse: 57.9228, mae: 5.2165, huber: 4.7462, swd: 3.2240, ept: 112.8986
      Epoch 7 composite train-obj: 31.887124
            No improvement (73.5212), counter 4/5
    Epoch [8/50], Train Losses: mse: 28.3177, mae: 3.6660, huber: 3.2062, swd: 2.5481, ept: 176.5102
    Epoch [8/50], Val Losses: mse: 68.6540, mae: 5.8612, huber: 5.3891, swd: 4.1513, ept: 100.7961
    Epoch [8/50], Test Losses: mse: 56.4687, mae: 5.0632, huber: 4.5954, swd: 2.9877, ept: 119.4625
      Epoch 8 composite train-obj: 29.591749
    Epoch [8/50], Test Losses: mse: 55.7225, mae: 5.3883, huber: 4.9140, swd: 5.0854, ept: 79.0426
    Best round's Test MSE: 55.7225, MAE: 5.3883, SWD: 5.0854
    Best round's Validation MSE: 65.3376, MAE: 5.9012, SWD: 6.6020
    Best round's Test verification MSE : 55.7225, MAE: 5.3883, SWD: 5.0854
    Time taken: 36.08 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.4234, mae: 6.4925, huber: 6.0107, swd: 9.1844, ept: 36.2886
    Epoch [1/50], Val Losses: mse: 75.2943, mae: 6.6420, huber: 6.1608, swd: 8.5322, ept: 37.1689
    Epoch [1/50], Test Losses: mse: 63.9440, mae: 6.0247, huber: 5.5452, swd: 8.2386, ept: 34.7902
      Epoch 1 composite train-obj: 77.015628
            Val objective improved inf → 79.5604, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 54.1577, mae: 5.4872, huber: 5.0102, swd: 7.0520, ept: 64.0850
    Epoch [2/50], Val Losses: mse: 68.0450, mae: 6.1642, huber: 5.6858, swd: 7.9986, ept: 55.9827
    Epoch [2/50], Test Losses: mse: 57.7181, mae: 5.5817, huber: 5.1055, swd: 6.9886, ept: 59.7871
      Epoch 2 composite train-obj: 57.683729
            Val objective improved 79.5604 → 72.0444, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 45.7891, mae: 4.9362, huber: 4.4632, swd: 5.6302, ept: 90.3570
    Epoch [3/50], Val Losses: mse: 68.0375, mae: 6.0928, huber: 5.6149, swd: 6.6304, ept: 62.6063
    Epoch [3/50], Test Losses: mse: 56.9136, mae: 5.5138, huber: 5.0376, swd: 5.6096, ept: 68.3317
      Epoch 3 composite train-obj: 48.604224
            Val objective improved 72.0444 → 71.3527, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 40.9714, mae: 4.6021, huber: 4.1319, swd: 4.5542, ept: 109.6404
    Epoch [4/50], Val Losses: mse: 67.4741, mae: 5.9946, huber: 5.5181, swd: 5.6933, ept: 74.0422
    Epoch [4/50], Test Losses: mse: 56.4380, mae: 5.3838, huber: 4.9098, swd: 4.7543, ept: 82.9935
      Epoch 4 composite train-obj: 43.248513
            Val objective improved 71.3527 → 70.3208, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 37.3289, mae: 4.3445, huber: 3.8767, swd: 3.8772, ept: 125.5136
    Epoch [5/50], Val Losses: mse: 67.8597, mae: 5.9491, huber: 5.4741, swd: 5.4169, ept: 83.2893
    Epoch [5/50], Test Losses: mse: 56.9665, mae: 5.3194, huber: 4.8470, swd: 4.6152, ept: 95.2899
      Epoch 5 composite train-obj: 39.267452
            No improvement (70.5682), counter 1/5
    Epoch [6/50], Train Losses: mse: 34.0203, mae: 4.1022, huber: 3.6369, swd: 3.3355, ept: 140.7049
    Epoch [6/50], Val Losses: mse: 69.7299, mae: 5.9869, huber: 5.5121, swd: 4.9015, ept: 91.7189
    Epoch [6/50], Test Losses: mse: 57.1684, mae: 5.2861, huber: 4.8140, swd: 3.7352, ept: 101.5234
      Epoch 6 composite train-obj: 35.688038
            No improvement (72.1807), counter 2/5
    Epoch [7/50], Train Losses: mse: 31.0594, mae: 3.8864, huber: 3.4234, swd: 2.8927, ept: 155.4054
    Epoch [7/50], Val Losses: mse: 69.8854, mae: 5.9294, huber: 5.4557, swd: 4.5671, ept: 94.4547
    Epoch [7/50], Test Losses: mse: 57.2053, mae: 5.2380, huber: 4.7671, swd: 3.3432, ept: 102.6337
      Epoch 7 composite train-obj: 32.505792
            No improvement (72.1690), counter 3/5
    Epoch [8/50], Train Losses: mse: 28.6816, mae: 3.7024, huber: 3.2417, swd: 2.5658, ept: 168.8224
    Epoch [8/50], Val Losses: mse: 73.0035, mae: 6.0295, huber: 5.5557, swd: 3.8334, ept: 98.4695
    Epoch [8/50], Test Losses: mse: 57.8610, mae: 5.1945, huber: 4.7244, swd: 2.4328, ept: 111.7648
      Epoch 8 composite train-obj: 29.964513
            No improvement (74.9202), counter 4/5
    Epoch [9/50], Train Losses: mse: 26.6336, mae: 3.5414, huber: 3.0828, swd: 2.3113, ept: 180.0818
    Epoch [9/50], Val Losses: mse: 70.7663, mae: 5.8855, huber: 5.4133, swd: 4.0231, ept: 109.9753
    Epoch [9/50], Test Losses: mse: 56.9880, mae: 5.1052, huber: 4.6367, swd: 2.3187, ept: 119.5233
      Epoch 9 composite train-obj: 27.789293
    Epoch [9/50], Test Losses: mse: 56.4380, mae: 5.3838, huber: 4.9098, swd: 4.7543, ept: 82.9935
    Best round's Test MSE: 56.4380, MAE: 5.3838, SWD: 4.7543
    Best round's Validation MSE: 67.4741, MAE: 5.9946, SWD: 5.6933
    Best round's Test verification MSE : 56.4380, MAE: 5.3838, SWD: 4.7543
    Time taken: 40.71 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 66.7888, mae: 6.2121, huber: 5.7315, swd: 9.5577, ept: 43.1840
    Epoch [1/50], Val Losses: mse: 68.7616, mae: 6.2792, huber: 5.7999, swd: 8.7171, ept: 51.1310
    Epoch [1/50], Test Losses: mse: 57.6104, mae: 5.6629, huber: 5.1854, swd: 8.3861, ept: 51.5872
      Epoch 1 composite train-obj: 71.567692
            Val objective improved inf → 73.1202, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.4276, mae: 5.1071, huber: 4.6328, swd: 6.0912, ept: 82.3273
    Epoch [2/50], Val Losses: mse: 64.0777, mae: 5.9004, huber: 5.4237, swd: 7.1700, ept: 71.6301
    Epoch [2/50], Test Losses: mse: 55.2937, mae: 5.3981, huber: 4.9233, swd: 5.8364, ept: 77.3665
      Epoch 2 composite train-obj: 51.473206
            Val objective improved 73.1202 → 67.6627, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.8165, mae: 4.6559, huber: 4.1855, swd: 4.8260, ept: 109.4918
    Epoch [3/50], Val Losses: mse: 66.2655, mae: 5.9079, huber: 5.4320, swd: 5.8370, ept: 83.6121
    Epoch [3/50], Test Losses: mse: 55.5528, mae: 5.2943, huber: 4.8211, swd: 4.5591, ept: 95.6702
      Epoch 3 composite train-obj: 44.229495
            No improvement (69.1840), counter 1/5
    Epoch [4/50], Train Losses: mse: 37.6029, mae: 4.3523, huber: 3.8849, swd: 4.0480, ept: 129.2421
    Epoch [4/50], Val Losses: mse: 67.7354, mae: 5.9043, huber: 5.4300, swd: 4.9871, ept: 89.3929
    Epoch [4/50], Test Losses: mse: 56.7377, mae: 5.2803, huber: 4.8086, swd: 4.4493, ept: 101.3161
      Epoch 4 composite train-obj: 39.626872
            No improvement (70.2290), counter 2/5
    Epoch [5/50], Train Losses: mse: 34.3795, mae: 4.1143, huber: 3.6494, swd: 3.5236, ept: 144.7525
    Epoch [5/50], Val Losses: mse: 68.0941, mae: 5.9434, huber: 5.4694, swd: 4.8174, ept: 92.8129
    Epoch [5/50], Test Losses: mse: 55.5475, mae: 5.1789, huber: 4.7084, swd: 3.6792, ept: 111.3538
      Epoch 5 composite train-obj: 36.141370
            No improvement (70.5028), counter 3/5
    Epoch [6/50], Train Losses: mse: 31.5749, mae: 3.9093, huber: 3.4467, swd: 3.1087, ept: 158.3415
    Epoch [6/50], Val Losses: mse: 68.1089, mae: 5.8760, huber: 5.4027, swd: 4.2594, ept: 103.3097
    Epoch [6/50], Test Losses: mse: 55.7583, mae: 5.1212, huber: 4.6518, swd: 3.5554, ept: 121.2913
      Epoch 6 composite train-obj: 33.129186
            No improvement (70.2386), counter 4/5
    Epoch [7/50], Train Losses: mse: 29.1530, mae: 3.7254, huber: 3.2650, swd: 2.7894, ept: 171.6115
    Epoch [7/50], Val Losses: mse: 68.9120, mae: 5.8812, huber: 5.4088, swd: 4.2760, ept: 104.8322
    Epoch [7/50], Test Losses: mse: 55.5299, mae: 5.0469, huber: 4.5789, swd: 3.1611, ept: 122.0071
      Epoch 7 composite train-obj: 30.547714
    Epoch [7/50], Test Losses: mse: 55.2937, mae: 5.3981, huber: 4.9233, swd: 5.8364, ept: 77.3665
    Best round's Test MSE: 55.2937, MAE: 5.3981, SWD: 5.8364
    Best round's Validation MSE: 64.0777, MAE: 5.9004, SWD: 7.1700
    Best round's Test verification MSE : 55.2937, MAE: 5.3981, SWD: 5.8364
    Time taken: 31.99 seconds
    
    ==================================================
    Experiment Summary (TimeMixer_lorenz_seq336_pred336_20250514_1924)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 55.8181 ± 0.4720
      mae: 5.3901 ± 0.0060
      huber: 4.9157 ± 0.0056
      swd: 5.2254 ± 0.4527
      ept: 79.8009 ± 2.3590
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 65.6298 ± 1.4019
      mae: 5.9321 ± 0.0442
      huber: 5.4557 ± 0.0441
      swd: 6.4884 ± 0.6082
      ept: 72.9074 ± 0.9899
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 108.85 seconds
    
    Experiment complete: TimeMixer_lorenz_seq336_pred336_20250514_1924
    Model: TimeMixer
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

## PatchTST 

### huber  


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
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.3631, mae: 6.3279, huber: 5.8480, swd: 10.6537, ept: 36.1925
    Epoch [1/50], Val Losses: mse: 64.9526, mae: 5.9413, huber: 5.4652, swd: 12.1339, ept: 66.6141
    Epoch [1/50], Test Losses: mse: 58.3317, mae: 5.5730, huber: 5.0983, swd: 13.1209, ept: 66.5582
      Epoch 1 composite train-obj: 5.847980
            Val objective improved inf → 5.4652, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 56.7507, mae: 5.5030, huber: 5.0278, swd: 9.8542, ept: 44.5526
    Epoch [2/50], Val Losses: mse: 57.5229, mae: 5.4106, huber: 4.9391, swd: 10.0631, ept: 101.6104
    Epoch [2/50], Test Losses: mse: 51.1887, mae: 5.0565, huber: 4.5870, swd: 11.6731, ept: 98.7353
      Epoch 2 composite train-obj: 5.027764
            Val objective improved 5.4652 → 4.9391, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.6730, mae: 5.0802, huber: 4.6089, swd: 9.0432, ept: 47.6925
    Epoch [3/50], Val Losses: mse: 57.3803, mae: 5.3658, huber: 4.8939, swd: 8.9634, ept: 111.3556
    Epoch [3/50], Test Losses: mse: 48.7107, mae: 4.8285, huber: 4.3598, swd: 9.9695, ept: 118.7665
      Epoch 3 composite train-obj: 4.608884
            Val objective improved 4.9391 → 4.8939, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 47.5047, mae: 4.8189, huber: 4.3509, swd: 8.2529, ept: 49.4518
    Epoch [4/50], Val Losses: mse: 58.3236, mae: 5.3137, huber: 4.8441, swd: 7.6812, ept: 120.8173
    Epoch [4/50], Test Losses: mse: 47.9454, mae: 4.7276, huber: 4.2610, swd: 8.7930, ept: 129.9525
      Epoch 4 composite train-obj: 4.350853
            Val objective improved 4.8939 → 4.8441, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 45.0544, mae: 4.6140, huber: 4.1489, swd: 7.6759, ept: 50.7091
    Epoch [5/50], Val Losses: mse: 58.7760, mae: 5.2252, huber: 4.7611, swd: 7.0261, ept: 126.5169
    Epoch [5/50], Test Losses: mse: 48.0040, mae: 4.5903, huber: 4.1298, swd: 8.0117, ept: 137.5828
      Epoch 5 composite train-obj: 4.148914
            Val objective improved 4.8441 → 4.7611, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 43.0476, mae: 4.4423, huber: 3.9798, swd: 7.2262, ept: 51.6553
    Epoch [6/50], Val Losses: mse: 55.8625, mae: 5.0701, huber: 4.6058, swd: 8.2859, ept: 133.9862
    Epoch [6/50], Test Losses: mse: 45.8183, mae: 4.4433, huber: 3.9831, swd: 7.9394, ept: 150.7014
      Epoch 6 composite train-obj: 3.979783
            Val objective improved 4.7611 → 4.6058, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 41.5180, mae: 4.3138, huber: 3.8532, swd: 6.7974, ept: 52.1210
    Epoch [7/50], Val Losses: mse: 56.1518, mae: 5.0261, huber: 4.5637, swd: 7.4032, ept: 139.6684
    Epoch [7/50], Test Losses: mse: 45.1429, mae: 4.3753, huber: 3.9169, swd: 7.8033, ept: 155.7879
      Epoch 7 composite train-obj: 3.853234
            Val objective improved 4.6058 → 4.5637, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 40.2613, mae: 4.2074, huber: 3.7487, swd: 6.4619, ept: 52.5529
    Epoch [8/50], Val Losses: mse: 58.3734, mae: 5.0614, huber: 4.6003, swd: 6.8617, ept: 139.8653
    Epoch [8/50], Test Losses: mse: 45.0964, mae: 4.3149, huber: 3.8590, swd: 7.0407, ept: 161.0537
      Epoch 8 composite train-obj: 3.748654
            No improvement (4.6003), counter 1/5
    Epoch [9/50], Train Losses: mse: 39.1707, mae: 4.1135, huber: 3.6564, swd: 6.1606, ept: 52.8368
    Epoch [9/50], Val Losses: mse: 56.9957, mae: 5.0621, huber: 4.6002, swd: 6.9942, ept: 127.9318
    Epoch [9/50], Test Losses: mse: 45.4769, mae: 4.3826, huber: 3.9251, swd: 7.2297, ept: 134.5290
      Epoch 9 composite train-obj: 3.656398
            No improvement (4.6002), counter 2/5
    Epoch [10/50], Train Losses: mse: 38.2241, mae: 4.0295, huber: 3.5741, swd: 5.8887, ept: 53.0832
    Epoch [10/50], Val Losses: mse: 57.5026, mae: 5.0250, huber: 4.5648, swd: 7.0246, ept: 142.0004
    Epoch [10/50], Test Losses: mse: 43.9745, mae: 4.2451, huber: 3.7898, swd: 7.3821, ept: 163.1123
      Epoch 10 composite train-obj: 3.574059
            No improvement (4.5648), counter 3/5
    Epoch [11/50], Train Losses: mse: 37.2458, mae: 3.9547, huber: 3.5006, swd: 5.7114, ept: 53.5176
    Epoch [11/50], Val Losses: mse: 56.2148, mae: 4.9348, huber: 4.4754, swd: 6.6168, ept: 149.5646
    Epoch [11/50], Test Losses: mse: 42.8212, mae: 4.1558, huber: 3.7027, swd: 6.5571, ept: 169.4006
      Epoch 11 composite train-obj: 3.500625
            Val objective improved 4.5637 → 4.4754, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 36.7670, mae: 3.9097, huber: 3.4566, swd: 5.5376, ept: 53.3105
    Epoch [12/50], Val Losses: mse: 57.0607, mae: 5.0302, huber: 4.5679, swd: 7.0532, ept: 145.1634
    Epoch [12/50], Test Losses: mse: 43.1583, mae: 4.2271, huber: 3.7697, swd: 7.3291, ept: 164.3565
      Epoch 12 composite train-obj: 3.456604
            No improvement (4.5679), counter 1/5
    Epoch [13/50], Train Losses: mse: 35.9509, mae: 3.8465, huber: 3.3947, swd: 5.4149, ept: 53.6487
    Epoch [13/50], Val Losses: mse: 54.2776, mae: 4.7873, huber: 4.3313, swd: 6.5521, ept: 152.7874
    Epoch [13/50], Test Losses: mse: 42.0582, mae: 4.0317, huber: 3.5803, swd: 5.7111, ept: 174.5572
      Epoch 13 composite train-obj: 3.394711
            Val objective improved 4.4754 → 4.3313, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 35.2429, mae: 3.7918, huber: 3.3410, swd: 5.1719, ept: 53.7775
    Epoch [14/50], Val Losses: mse: 57.1853, mae: 4.8977, huber: 4.4426, swd: 5.7928, ept: 150.8775
    Epoch [14/50], Test Losses: mse: 41.6617, mae: 3.9820, huber: 3.5339, swd: 5.3379, ept: 177.8852
      Epoch 14 composite train-obj: 3.341008
            No improvement (4.4426), counter 1/5
    Epoch [15/50], Train Losses: mse: 34.8936, mae: 3.7595, huber: 3.3096, swd: 5.0757, ept: 53.6734
    Epoch [15/50], Val Losses: mse: 55.1581, mae: 4.8372, huber: 4.3781, swd: 6.4843, ept: 152.5494
    Epoch [15/50], Test Losses: mse: 42.3131, mae: 4.0343, huber: 3.5799, swd: 5.3978, ept: 177.8401
      Epoch 15 composite train-obj: 3.309577
            No improvement (4.3781), counter 2/5
    Epoch [16/50], Train Losses: mse: 34.0449, mae: 3.6943, huber: 3.2455, swd: 4.9129, ept: 53.8991
    Epoch [16/50], Val Losses: mse: 54.1169, mae: 4.7743, huber: 4.3189, swd: 6.3140, ept: 154.0168
    Epoch [16/50], Test Losses: mse: 40.2478, mae: 3.9651, huber: 3.5151, swd: 5.9869, ept: 174.5130
      Epoch 16 composite train-obj: 3.245521
            Val objective improved 4.3313 → 4.3189, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 33.7330, mae: 3.6678, huber: 3.2197, swd: 4.7982, ept: 54.0734
    Epoch [17/50], Val Losses: mse: 59.0529, mae: 5.0654, huber: 4.6004, swd: 5.5427, ept: 149.7396
    Epoch [17/50], Test Losses: mse: 43.8216, mae: 4.1965, huber: 3.7369, swd: 4.6580, ept: 173.8794
      Epoch 17 composite train-obj: 3.219702
            No improvement (4.6004), counter 1/5
    Epoch [18/50], Train Losses: mse: 33.2715, mae: 3.6362, huber: 3.1884, swd: 4.6892, ept: 53.8856
    Epoch [18/50], Val Losses: mse: 54.6027, mae: 4.7944, huber: 4.3386, swd: 6.0188, ept: 149.3420
    Epoch [18/50], Test Losses: mse: 41.4760, mae: 4.0863, huber: 3.6344, swd: 6.3827, ept: 157.8976
      Epoch 18 composite train-obj: 3.188440
            No improvement (4.3386), counter 2/5
    Epoch [19/50], Train Losses: mse: 32.9010, mae: 3.6110, huber: 3.1638, swd: 4.6143, ept: 54.1204
    Epoch [19/50], Val Losses: mse: 56.2408, mae: 4.8085, huber: 4.3546, swd: 5.7106, ept: 157.9278
    Epoch [19/50], Test Losses: mse: 41.3677, mae: 3.9912, huber: 3.5417, swd: 5.5895, ept: 176.5873
      Epoch 19 composite train-obj: 3.163764
            No improvement (4.3546), counter 3/5
    Epoch [20/50], Train Losses: mse: 32.3947, mae: 3.5666, huber: 3.1205, swd: 4.4737, ept: 53.9526
    Epoch [20/50], Val Losses: mse: 55.5824, mae: 4.7627, huber: 4.3097, swd: 5.4255, ept: 163.8183
    Epoch [20/50], Test Losses: mse: 41.1495, mae: 3.9298, huber: 3.4816, swd: 5.1929, ept: 184.3209
      Epoch 20 composite train-obj: 3.120536
            Val objective improved 4.3189 → 4.3097, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 31.9381, mae: 3.5360, huber: 3.0904, swd: 4.3819, ept: 54.3829
    Epoch [21/50], Val Losses: mse: 55.2115, mae: 4.7450, huber: 4.2899, swd: 6.1068, ept: 159.7868
    Epoch [21/50], Test Losses: mse: 40.3265, mae: 3.9146, huber: 3.4650, swd: 5.7493, ept: 178.2359
      Epoch 21 composite train-obj: 3.090350
            Val objective improved 4.3097 → 4.2899, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 31.7487, mae: 3.5252, huber: 3.0796, swd: 4.3546, ept: 54.3193
    Epoch [22/50], Val Losses: mse: 53.8593, mae: 4.7548, huber: 4.2936, swd: 6.0605, ept: 164.7647
    Epoch [22/50], Test Losses: mse: 41.7118, mae: 4.0330, huber: 3.5791, swd: 5.4678, ept: 177.7009
      Epoch 22 composite train-obj: 3.079573
            No improvement (4.2936), counter 1/5
    Epoch [23/50], Train Losses: mse: 31.4112, mae: 3.4964, huber: 3.0515, swd: 4.2626, ept: 54.1945
    Epoch [23/50], Val Losses: mse: 54.1023, mae: 4.6805, huber: 4.2250, swd: 5.2450, ept: 165.5814
    Epoch [23/50], Test Losses: mse: 40.2764, mae: 3.8796, huber: 3.4291, swd: 4.3610, ept: 183.4612
      Epoch 23 composite train-obj: 3.051467
            Val objective improved 4.2899 → 4.2250, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 31.2517, mae: 3.4882, huber: 3.0434, swd: 4.2113, ept: 54.2841
    Epoch [24/50], Val Losses: mse: 54.7642, mae: 4.7354, huber: 4.2793, swd: 5.8360, ept: 162.6305
    Epoch [24/50], Test Losses: mse: 40.4930, mae: 3.9117, huber: 3.4603, swd: 4.7251, ept: 184.3777
      Epoch 24 composite train-obj: 3.043363
            No improvement (4.2793), counter 1/5
    Epoch [25/50], Train Losses: mse: 30.6842, mae: 3.4429, huber: 2.9991, swd: 4.1031, ept: 54.2868
    Epoch [25/50], Val Losses: mse: 53.9930, mae: 4.6830, huber: 4.2295, swd: 5.3374, ept: 161.9455
    Epoch [25/50], Test Losses: mse: 40.3239, mae: 3.8848, huber: 3.4371, swd: 4.5761, ept: 180.6720
      Epoch 25 composite train-obj: 2.999067
            No improvement (4.2295), counter 2/5
    Epoch [26/50], Train Losses: mse: 30.6175, mae: 3.4419, huber: 2.9980, swd: 4.0870, ept: 54.5794
    Epoch [26/50], Val Losses: mse: 57.1516, mae: 4.8814, huber: 4.4227, swd: 5.3592, ept: 164.0606
    Epoch [26/50], Test Losses: mse: 40.9424, mae: 4.0004, huber: 3.5488, swd: 4.9456, ept: 178.1579
      Epoch 26 composite train-obj: 2.998004
            No improvement (4.4227), counter 3/5
    Epoch [27/50], Train Losses: mse: 30.2812, mae: 3.4092, huber: 2.9662, swd: 3.9857, ept: 54.4734
    Epoch [27/50], Val Losses: mse: 57.0164, mae: 4.8384, huber: 4.3793, swd: 4.9517, ept: 166.8694
    Epoch [27/50], Test Losses: mse: 40.6224, mae: 3.9631, huber: 3.5087, swd: 4.5193, ept: 185.6369
      Epoch 27 composite train-obj: 2.966212
            No improvement (4.3793), counter 4/5
    Epoch [28/50], Train Losses: mse: 30.1786, mae: 3.4059, huber: 2.9627, swd: 3.9700, ept: 54.4372
    Epoch [28/50], Val Losses: mse: 54.8362, mae: 4.7422, huber: 4.2813, swd: 5.2372, ept: 166.4439
    Epoch [28/50], Test Losses: mse: 40.7014, mae: 3.9383, huber: 3.4819, swd: 4.0709, ept: 189.7306
      Epoch 28 composite train-obj: 2.962666
    Epoch [28/50], Test Losses: mse: 40.2764, mae: 3.8796, huber: 3.4291, swd: 4.3610, ept: 183.4612
    Best round's Test MSE: 40.2764, MAE: 3.8796, SWD: 4.3610
    Best round's Validation MSE: 54.1023, MAE: 4.6805, SWD: 5.2450
    Best round's Test verification MSE : 40.2764, MAE: 3.8796, SWD: 4.3610
    Time taken: 69.47 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.0858, mae: 6.3584, huber: 5.8784, swd: 10.5289, ept: 35.7649
    Epoch [1/50], Val Losses: mse: 64.0810, mae: 5.9722, huber: 5.4956, swd: 15.0569, ept: 63.7393
    Epoch [1/50], Test Losses: mse: 58.2809, mae: 5.6166, huber: 5.1413, swd: 15.5238, ept: 66.3906
      Epoch 1 composite train-obj: 5.878400
            Val objective improved inf → 5.4956, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 57.3644, mae: 5.5310, huber: 5.0556, swd: 9.9750, ept: 45.1124
    Epoch [2/50], Val Losses: mse: 58.6307, mae: 5.5576, huber: 5.0846, swd: 10.9278, ept: 87.7755
    Epoch [2/50], Test Losses: mse: 50.1017, mae: 5.0601, huber: 4.5896, swd: 11.6239, ept: 94.1353
      Epoch 2 composite train-obj: 5.055632
            Val objective improved 5.4956 → 5.0846, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.9560, mae: 5.0889, huber: 4.6177, swd: 8.8954, ept: 47.9601
    Epoch [3/50], Val Losses: mse: 56.2881, mae: 5.2844, huber: 4.8146, swd: 9.5308, ept: 110.2456
    Epoch [3/50], Test Losses: mse: 46.2227, mae: 4.6859, huber: 4.2193, swd: 10.4722, ept: 123.1389
      Epoch 3 composite train-obj: 4.617698
            Val objective improved 5.0846 → 4.8146, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 47.1669, mae: 4.7975, huber: 4.3297, swd: 8.2398, ept: 49.4460
    Epoch [4/50], Val Losses: mse: 56.0970, mae: 5.2220, huber: 4.7532, swd: 8.8037, ept: 121.8834
    Epoch [4/50], Test Losses: mse: 45.6703, mae: 4.5723, huber: 4.1078, swd: 9.7473, ept: 132.2419
      Epoch 4 composite train-obj: 4.329676
            Val objective improved 4.8146 → 4.7532, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 44.7762, mae: 4.6003, huber: 4.1351, swd: 7.6171, ept: 50.4801
    Epoch [5/50], Val Losses: mse: 56.4839, mae: 5.0883, huber: 4.6248, swd: 7.9011, ept: 133.4423
    Epoch [5/50], Test Losses: mse: 43.2028, mae: 4.3533, huber: 3.8942, swd: 8.6512, ept: 148.0780
      Epoch 5 composite train-obj: 4.135068
            Val objective improved 4.7532 → 4.6248, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 42.6746, mae: 4.4261, huber: 3.9634, swd: 7.0858, ept: 51.1377
    Epoch [6/50], Val Losses: mse: 55.3311, mae: 5.0060, huber: 4.5442, swd: 7.2565, ept: 142.5386
    Epoch [6/50], Test Losses: mse: 43.4302, mae: 4.3415, huber: 3.8832, swd: 8.3782, ept: 155.1230
      Epoch 6 composite train-obj: 3.963426
            Val objective improved 4.6248 → 4.5442, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 41.2437, mae: 4.3071, huber: 3.8462, swd: 6.6460, ept: 51.5220
    Epoch [7/50], Val Losses: mse: 58.3580, mae: 5.0871, huber: 4.6260, swd: 6.5895, ept: 141.0032
    Epoch [7/50], Test Losses: mse: 42.2782, mae: 4.2044, huber: 3.7482, swd: 7.1457, ept: 159.3196
      Epoch 7 composite train-obj: 3.846164
            No improvement (4.6260), counter 1/5
    Epoch [8/50], Train Losses: mse: 39.8982, mae: 4.1969, huber: 3.7377, swd: 6.2545, ept: 51.5191
    Epoch [8/50], Val Losses: mse: 55.4261, mae: 4.9717, huber: 4.5094, swd: 7.6729, ept: 138.6726
    Epoch [8/50], Test Losses: mse: 42.8605, mae: 4.2453, huber: 3.7859, swd: 7.0143, ept: 163.2942
      Epoch 8 composite train-obj: 3.737710
            Val objective improved 4.5442 → 4.5094, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 38.7183, mae: 4.1020, huber: 3.6445, swd: 6.0178, ept: 52.2564
    Epoch [9/50], Val Losses: mse: 55.5138, mae: 4.9119, huber: 4.4538, swd: 7.2786, ept: 143.3254
    Epoch [9/50], Test Losses: mse: 41.2924, mae: 4.0863, huber: 3.6337, swd: 7.0500, ept: 165.9172
      Epoch 9 composite train-obj: 3.644457
            Val objective improved 4.5094 → 4.4538, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 37.8602, mae: 4.0313, huber: 3.5750, swd: 5.7773, ept: 52.3003
    Epoch [10/50], Val Losses: mse: 54.6338, mae: 4.8455, huber: 4.3881, swd: 6.7409, ept: 147.7619
    Epoch [10/50], Test Losses: mse: 40.0093, mae: 3.9808, huber: 3.5302, swd: 6.6047, ept: 172.2181
      Epoch 10 composite train-obj: 3.575019
            Val objective improved 4.4538 → 4.3881, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 36.9524, mae: 3.9593, huber: 3.5044, swd: 5.5489, ept: 52.5266
    Epoch [11/50], Val Losses: mse: 56.1132, mae: 4.9388, huber: 4.4789, swd: 7.1700, ept: 143.5361
    Epoch [11/50], Test Losses: mse: 41.3211, mae: 4.1104, huber: 3.6557, swd: 6.9327, ept: 166.4090
      Epoch 11 composite train-obj: 3.504411
            No improvement (4.4789), counter 1/5
    Epoch [12/50], Train Losses: mse: 36.2297, mae: 3.8988, huber: 3.4450, swd: 5.3479, ept: 52.7161
    Epoch [12/50], Val Losses: mse: 54.3280, mae: 4.9329, huber: 4.4703, swd: 7.0002, ept: 139.6345
    Epoch [12/50], Test Losses: mse: 40.6297, mae: 4.1220, huber: 3.6631, swd: 6.0906, ept: 163.4872
      Epoch 12 composite train-obj: 3.444963
            No improvement (4.4703), counter 2/5
    Epoch [13/50], Train Losses: mse: 35.4645, mae: 3.8365, huber: 3.3840, swd: 5.1709, ept: 53.0068
    Epoch [13/50], Val Losses: mse: 52.8262, mae: 4.7795, huber: 4.3210, swd: 7.0916, ept: 152.5782
    Epoch [13/50], Test Losses: mse: 38.8372, mae: 3.9827, huber: 3.5287, swd: 6.7441, ept: 172.5844
      Epoch 13 composite train-obj: 3.384022
            Val objective improved 4.3881 → 4.3210, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 35.2603, mae: 3.8178, huber: 3.3657, swd: 5.1295, ept: 52.9463
    Epoch [14/50], Val Losses: mse: 54.0375, mae: 4.8211, huber: 4.3608, swd: 6.7372, ept: 152.3421
    Epoch [14/50], Test Losses: mse: 39.1227, mae: 3.9444, huber: 3.4900, swd: 6.4077, ept: 180.3648
      Epoch 14 composite train-obj: 3.365717
            No improvement (4.3608), counter 1/5
    Epoch [15/50], Train Losses: mse: 34.4493, mae: 3.7523, huber: 3.3015, swd: 4.9299, ept: 53.1150
    Epoch [15/50], Val Losses: mse: 55.7017, mae: 4.9338, huber: 4.4737, swd: 6.5915, ept: 148.0407
    Epoch [15/50], Test Losses: mse: 41.1387, mae: 4.1285, huber: 3.6716, swd: 6.0297, ept: 171.4778
      Epoch 15 composite train-obj: 3.301456
            No improvement (4.4737), counter 2/5
    Epoch [16/50], Train Losses: mse: 33.9347, mae: 3.7131, huber: 3.2631, swd: 4.8224, ept: 53.3188
    Epoch [16/50], Val Losses: mse: 54.9143, mae: 4.8378, huber: 4.3782, swd: 5.5165, ept: 154.7707
    Epoch [16/50], Test Losses: mse: 40.3346, mae: 4.0375, huber: 3.5832, swd: 5.7120, ept: 172.3065
      Epoch 16 composite train-obj: 3.263073
            No improvement (4.3782), counter 3/5
    Epoch [17/50], Train Losses: mse: 33.5553, mae: 3.6807, huber: 3.2311, swd: 4.6644, ept: 53.4044
    Epoch [17/50], Val Losses: mse: 56.3654, mae: 4.9721, huber: 4.5092, swd: 6.3389, ept: 153.2600
    Epoch [17/50], Test Losses: mse: 40.8677, mae: 4.0983, huber: 3.6395, swd: 5.8393, ept: 182.6239
      Epoch 17 composite train-obj: 3.231146
            No improvement (4.5092), counter 4/5
    Epoch [18/50], Train Losses: mse: 33.2218, mae: 3.6527, huber: 3.2039, swd: 4.6074, ept: 53.2585
    Epoch [18/50], Val Losses: mse: 56.7867, mae: 4.9852, huber: 4.5207, swd: 6.2088, ept: 151.5992
    Epoch [18/50], Test Losses: mse: 41.5292, mae: 4.0929, huber: 3.6342, swd: 4.9105, ept: 182.9027
      Epoch 18 composite train-obj: 3.203869
    Epoch [18/50], Test Losses: mse: 38.8372, mae: 3.9827, huber: 3.5287, swd: 6.7441, ept: 172.5844
    Best round's Test MSE: 38.8372, MAE: 3.9827, SWD: 6.7441
    Best round's Validation MSE: 52.8262, MAE: 4.7795, SWD: 7.0916
    Best round's Test verification MSE : 38.8372, MAE: 3.9827, SWD: 6.7441
    Time taken: 44.84 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.9640, mae: 6.3201, huber: 5.8402, swd: 10.9498, ept: 35.6477
    Epoch [1/50], Val Losses: mse: 63.4833, mae: 5.8993, huber: 5.4233, swd: 14.0726, ept: 63.0450
    Epoch [1/50], Test Losses: mse: 57.2777, mae: 5.5013, huber: 5.0272, swd: 15.2111, ept: 66.8645
      Epoch 1 composite train-obj: 5.840205
            Val objective improved inf → 5.4233, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 56.2265, mae: 5.4713, huber: 4.9965, swd: 10.6587, ept: 45.3480
    Epoch [2/50], Val Losses: mse: 56.5784, mae: 5.4328, huber: 4.9596, swd: 12.4183, ept: 96.0479
    Epoch [2/50], Test Losses: mse: 51.2805, mae: 5.0809, huber: 4.6101, swd: 13.1178, ept: 99.9732
      Epoch 2 composite train-obj: 4.996518
            Val objective improved 5.4233 → 4.9596, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.8301, mae: 5.0713, huber: 4.6007, swd: 9.9874, ept: 48.1008
    Epoch [3/50], Val Losses: mse: 55.4661, mae: 5.2466, huber: 4.7766, swd: 10.6966, ept: 109.8621
    Epoch [3/50], Test Losses: mse: 49.0371, mae: 4.8240, huber: 4.3569, swd: 11.2576, ept: 115.9583
      Epoch 3 composite train-obj: 4.600676
            Val objective improved 4.9596 → 4.7766, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 47.6358, mae: 4.8092, huber: 4.3418, swd: 9.1105, ept: 49.9286
    Epoch [4/50], Val Losses: mse: 53.5999, mae: 5.1035, huber: 4.6348, swd: 10.5772, ept: 122.6373
    Epoch [4/50], Test Losses: mse: 45.6872, mae: 4.5228, huber: 4.0606, swd: 10.3957, ept: 133.2487
      Epoch 4 composite train-obj: 4.341791
            Val objective improved 4.7766 → 4.6348, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 45.0245, mae: 4.5996, huber: 4.1349, swd: 8.4036, ept: 50.5408
    Epoch [5/50], Val Losses: mse: 54.5486, mae: 5.0692, huber: 4.6039, swd: 9.2792, ept: 127.8671
    Epoch [5/50], Test Losses: mse: 45.0019, mae: 4.5118, huber: 4.0494, swd: 9.8777, ept: 141.5359
      Epoch 5 composite train-obj: 4.134887
            Val objective improved 4.6348 → 4.6039, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 43.1204, mae: 4.4420, huber: 3.9797, swd: 7.7629, ept: 51.0552
    Epoch [6/50], Val Losses: mse: 54.0002, mae: 4.9471, huber: 4.4860, swd: 8.3890, ept: 137.9732
    Epoch [6/50], Test Losses: mse: 44.4033, mae: 4.3427, huber: 3.8862, swd: 8.1980, ept: 153.7328
      Epoch 6 composite train-obj: 3.979673
            Val objective improved 4.6039 → 4.4860, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 41.5061, mae: 4.3120, huber: 3.8516, swd: 7.3460, ept: 51.5000
    Epoch [7/50], Val Losses: mse: 55.8893, mae: 4.9712, huber: 4.5118, swd: 7.7197, ept: 140.7376
    Epoch [7/50], Test Losses: mse: 44.4824, mae: 4.2302, huber: 3.7764, swd: 7.1336, ept: 161.7673
      Epoch 7 composite train-obj: 3.851564
            No improvement (4.5118), counter 1/5
    Epoch [8/50], Train Losses: mse: 40.2754, mae: 4.2060, huber: 3.7475, swd: 6.8730, ept: 52.1091
    Epoch [8/50], Val Losses: mse: 53.4814, mae: 4.9006, huber: 4.4368, swd: 8.3224, ept: 141.2010
    Epoch [8/50], Test Losses: mse: 43.6412, mae: 4.2982, huber: 3.8377, swd: 8.0800, ept: 163.8818
      Epoch 8 composite train-obj: 3.747504
            Val objective improved 4.4860 → 4.4368, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 39.0601, mae: 4.1101, huber: 3.6532, swd: 6.6084, ept: 52.3730
    Epoch [9/50], Val Losses: mse: 55.4551, mae: 4.9853, huber: 4.5228, swd: 7.9188, ept: 134.6976
    Epoch [9/50], Test Losses: mse: 43.7397, mae: 4.2534, huber: 3.7965, swd: 7.4801, ept: 155.7251
      Epoch 9 composite train-obj: 3.653171
            No improvement (4.5228), counter 1/5
    Epoch [10/50], Train Losses: mse: 38.3973, mae: 4.0501, huber: 3.5945, swd: 6.3472, ept: 52.4947
    Epoch [10/50], Val Losses: mse: 54.7099, mae: 4.8747, huber: 4.4150, swd: 7.6310, ept: 147.7718
    Epoch [10/50], Test Losses: mse: 43.5354, mae: 4.2121, huber: 3.7568, swd: 7.5930, ept: 165.6778
      Epoch 10 composite train-obj: 3.594544
            Val objective improved 4.4368 → 4.4150, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 37.3556, mae: 3.9655, huber: 3.5114, swd: 6.1253, ept: 52.6077
    Epoch [11/50], Val Losses: mse: 56.9490, mae: 4.9106, huber: 4.4529, swd: 6.5377, ept: 149.3662
    Epoch [11/50], Test Losses: mse: 43.8850, mae: 4.1906, huber: 3.7371, swd: 6.5396, ept: 169.1633
      Epoch 11 composite train-obj: 3.511357
            No improvement (4.4529), counter 1/5
    Epoch [12/50], Train Losses: mse: 36.6527, mae: 3.9083, huber: 3.4553, swd: 5.9284, ept: 52.8448
    Epoch [12/50], Val Losses: mse: 57.7913, mae: 4.9614, huber: 4.5016, swd: 6.5868, ept: 151.8804
    Epoch [12/50], Test Losses: mse: 43.2329, mae: 4.0851, huber: 3.6331, swd: 6.1675, ept: 175.4275
      Epoch 12 composite train-obj: 3.455267
            No improvement (4.5016), counter 2/5
    Epoch [13/50], Train Losses: mse: 35.8661, mae: 3.8402, huber: 3.3885, swd: 5.6352, ept: 52.8687
    Epoch [13/50], Val Losses: mse: 56.2626, mae: 4.9365, huber: 4.4731, swd: 6.6790, ept: 152.0452
    Epoch [13/50], Test Losses: mse: 42.7399, mae: 4.1539, huber: 3.6966, swd: 6.1895, ept: 176.3463
      Epoch 13 composite train-obj: 3.388473
            No improvement (4.4731), counter 3/5
    Epoch [14/50], Train Losses: mse: 35.6199, mae: 3.8176, huber: 3.3665, swd: 5.5657, ept: 52.8683
    Epoch [14/50], Val Losses: mse: 57.2030, mae: 5.0146, huber: 4.5504, swd: 5.7748, ept: 153.0460
    Epoch [14/50], Test Losses: mse: 44.6453, mae: 4.2869, huber: 3.8275, swd: 5.8523, ept: 164.8553
      Epoch 14 composite train-obj: 3.366518
            No improvement (4.5504), counter 4/5
    Epoch [15/50], Train Losses: mse: 34.9129, mae: 3.7646, huber: 3.3144, swd: 5.4124, ept: 53.0189
    Epoch [15/50], Val Losses: mse: 56.0088, mae: 4.8173, huber: 4.3613, swd: 5.4489, ept: 159.0501
    Epoch [15/50], Test Losses: mse: 43.1349, mae: 4.1343, huber: 3.6826, swd: 6.1334, ept: 173.4727
      Epoch 15 composite train-obj: 3.314388
            Val objective improved 4.4150 → 4.3613, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 34.5384, mae: 3.7363, huber: 3.2866, swd: 5.3203, ept: 53.0535
    Epoch [16/50], Val Losses: mse: 58.5810, mae: 4.9998, huber: 4.5388, swd: 5.8891, ept: 151.2805
    Epoch [16/50], Test Losses: mse: 44.6865, mae: 4.1953, huber: 3.7397, swd: 5.1950, ept: 176.1889
      Epoch 16 composite train-obj: 3.286620
            No improvement (4.5388), counter 1/5
    Epoch [17/50], Train Losses: mse: 34.0229, mae: 3.6886, huber: 3.2401, swd: 5.1614, ept: 53.0044
    Epoch [17/50], Val Losses: mse: 57.8674, mae: 4.9132, huber: 4.4565, swd: 5.5254, ept: 160.8701
    Epoch [17/50], Test Losses: mse: 41.3377, mae: 3.9876, huber: 3.5368, swd: 5.7104, ept: 182.8409
      Epoch 17 composite train-obj: 3.240107
            No improvement (4.4565), counter 2/5
    Epoch [18/50], Train Losses: mse: 33.4953, mae: 3.6503, huber: 3.2022, swd: 5.0361, ept: 53.0141
    Epoch [18/50], Val Losses: mse: 56.9571, mae: 4.8064, huber: 4.3518, swd: 5.1056, ept: 166.6335
    Epoch [18/50], Test Losses: mse: 41.6286, mae: 3.9181, huber: 3.4707, swd: 4.4191, ept: 183.9312
      Epoch 18 composite train-obj: 3.202201
            Val objective improved 4.3613 → 4.3518, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 33.2660, mae: 3.6324, huber: 3.1848, swd: 4.9558, ept: 53.4092
    Epoch [19/50], Val Losses: mse: 57.9068, mae: 4.8738, huber: 4.4181, swd: 5.4183, ept: 163.8817
    Epoch [19/50], Test Losses: mse: 42.1040, mae: 4.0120, huber: 3.5635, swd: 5.4117, ept: 177.7068
      Epoch 19 composite train-obj: 3.184774
            No improvement (4.4181), counter 1/5
    Epoch [20/50], Train Losses: mse: 32.7598, mae: 3.5939, huber: 3.1471, swd: 4.8712, ept: 53.4340
    Epoch [20/50], Val Losses: mse: 56.7652, mae: 4.8918, huber: 4.4320, swd: 5.9299, ept: 162.1447
    Epoch [20/50], Test Losses: mse: 42.0919, mae: 4.0015, huber: 3.5497, swd: 4.8187, ept: 186.7314
      Epoch 20 composite train-obj: 3.147082
            No improvement (4.4320), counter 2/5
    Epoch [21/50], Train Losses: mse: 32.4107, mae: 3.5577, huber: 3.1120, swd: 4.7447, ept: 53.5372
    Epoch [21/50], Val Losses: mse: 54.6568, mae: 4.7918, huber: 4.3334, swd: 6.4201, ept: 160.5379
    Epoch [21/50], Test Losses: mse: 41.3934, mae: 4.0298, huber: 3.5760, swd: 5.3136, ept: 182.6258
      Epoch 21 composite train-obj: 3.111981
            Val objective improved 4.3518 → 4.3334, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 32.1547, mae: 3.5406, huber: 3.0952, swd: 4.7119, ept: 53.3443
    Epoch [22/50], Val Losses: mse: 57.0589, mae: 4.8721, huber: 4.4089, swd: 5.5361, ept: 172.1001
    Epoch [22/50], Test Losses: mse: 41.5635, mae: 4.0095, huber: 3.5522, swd: 5.0062, ept: 191.8084
      Epoch 22 composite train-obj: 3.095173
            No improvement (4.4089), counter 1/5
    Epoch [23/50], Train Losses: mse: 32.0808, mae: 3.5361, huber: 3.0905, swd: 4.6389, ept: 53.3906
    Epoch [23/50], Val Losses: mse: 55.8694, mae: 4.7530, huber: 4.2973, swd: 5.4288, ept: 166.7408
    Epoch [23/50], Test Losses: mse: 41.5613, mae: 3.9285, huber: 3.4794, swd: 4.9489, ept: 187.0417
      Epoch 23 composite train-obj: 3.090527
            Val objective improved 4.3334 → 4.2973, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 31.6544, mae: 3.5025, huber: 3.0577, swd: 4.5385, ept: 53.8479
    Epoch [24/50], Val Losses: mse: 55.4761, mae: 4.7548, huber: 4.2961, swd: 5.4503, ept: 174.0955
    Epoch [24/50], Test Losses: mse: 41.2938, mae: 3.8849, huber: 3.4326, swd: 4.3809, ept: 196.7607
      Epoch 24 composite train-obj: 3.057701
            Val objective improved 4.2973 → 4.2961, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 31.2689, mae: 3.4739, huber: 3.0297, swd: 4.4615, ept: 53.7296
    Epoch [25/50], Val Losses: mse: 55.8870, mae: 4.6890, huber: 4.2327, swd: 5.0714, ept: 171.4955
    Epoch [25/50], Test Losses: mse: 41.1650, mae: 3.8211, huber: 3.3713, swd: 3.8286, ept: 197.2946
      Epoch 25 composite train-obj: 3.029690
            Val objective improved 4.2961 → 4.2327, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 31.2942, mae: 3.4771, huber: 3.0329, swd: 4.4447, ept: 53.7393
    Epoch [26/50], Val Losses: mse: 55.7176, mae: 4.7377, huber: 4.2817, swd: 5.6248, ept: 164.9586
    Epoch [26/50], Test Losses: mse: 42.1890, mae: 3.9454, huber: 3.4955, swd: 4.7580, ept: 190.6817
      Epoch 26 composite train-obj: 3.032911
            No improvement (4.2817), counter 1/5
    Epoch [27/50], Train Losses: mse: 30.9625, mae: 3.4469, huber: 3.0034, swd: 4.3959, ept: 53.6615
    Epoch [27/50], Val Losses: mse: 57.0801, mae: 4.7991, huber: 4.3424, swd: 5.0850, ept: 166.0497
    Epoch [27/50], Test Losses: mse: 41.4839, mae: 3.8924, huber: 3.4427, swd: 4.1887, ept: 195.3501
      Epoch 27 composite train-obj: 3.003419
            No improvement (4.3424), counter 2/5
    Epoch [28/50], Train Losses: mse: 30.5827, mae: 3.4169, huber: 2.9741, swd: 4.2817, ept: 53.8051
    Epoch [28/50], Val Losses: mse: 55.3949, mae: 4.7603, huber: 4.3021, swd: 6.2779, ept: 164.2192
    Epoch [28/50], Test Losses: mse: 42.1187, mae: 3.9040, huber: 3.4548, swd: 4.6359, ept: 198.0030
      Epoch 28 composite train-obj: 2.974128
            No improvement (4.3021), counter 3/5
    Epoch [29/50], Train Losses: mse: 30.3798, mae: 3.4046, huber: 2.9619, swd: 4.2592, ept: 53.6019
    Epoch [29/50], Val Losses: mse: 57.7516, mae: 4.8438, huber: 4.3836, swd: 4.9511, ept: 174.4480
    Epoch [29/50], Test Losses: mse: 42.4502, mae: 3.9601, huber: 3.5085, swd: 3.9240, ept: 191.5581
      Epoch 29 composite train-obj: 2.961940
            No improvement (4.3836), counter 4/5
    Epoch [30/50], Train Losses: mse: 30.1945, mae: 3.3873, huber: 2.9451, swd: 4.2059, ept: 53.6045
    Epoch [30/50], Val Losses: mse: 52.9245, mae: 4.6520, huber: 4.1946, swd: 5.3853, ept: 170.7086
    Epoch [30/50], Test Losses: mse: 40.8898, mae: 3.8967, huber: 3.4466, swd: 4.5953, ept: 192.5896
      Epoch 30 composite train-obj: 2.945053
            Val objective improved 4.2327 → 4.1946, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 30.1491, mae: 3.3870, huber: 2.9446, swd: 4.2070, ept: 53.9515
    Epoch [31/50], Val Losses: mse: 55.0452, mae: 4.8422, huber: 4.3790, swd: 6.1540, ept: 168.9249
    Epoch [31/50], Test Losses: mse: 41.6557, mae: 4.0683, huber: 3.6094, swd: 4.9980, ept: 183.6508
      Epoch 31 composite train-obj: 2.944609
            No improvement (4.3790), counter 1/5
    Epoch [32/50], Train Losses: mse: 29.8367, mae: 3.3592, huber: 2.9175, swd: 4.1072, ept: 53.7101
    Epoch [32/50], Val Losses: mse: 52.7981, mae: 4.5609, huber: 4.1092, swd: 5.7685, ept: 170.5698
    Epoch [32/50], Test Losses: mse: 39.3536, mae: 3.7198, huber: 3.2749, swd: 4.5011, ept: 202.1552
      Epoch 32 composite train-obj: 2.917488
            Val objective improved 4.1946 → 4.1092, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 29.7559, mae: 3.3574, huber: 2.9157, swd: 4.1294, ept: 53.9259
    Epoch [33/50], Val Losses: mse: 52.5219, mae: 4.5934, huber: 4.1378, swd: 5.5745, ept: 172.1095
    Epoch [33/50], Test Losses: mse: 40.2911, mae: 3.8031, huber: 3.3547, swd: 4.3580, ept: 198.7963
      Epoch 33 composite train-obj: 2.915741
            No improvement (4.1378), counter 1/5
    Epoch [34/50], Train Losses: mse: 29.6245, mae: 3.3410, huber: 2.9000, swd: 4.0842, ept: 53.9956
    Epoch [34/50], Val Losses: mse: 54.2915, mae: 4.6048, huber: 4.1496, swd: 5.0529, ept: 174.1050
    Epoch [34/50], Test Losses: mse: 40.2306, mae: 3.7774, huber: 3.3282, swd: 4.6602, ept: 198.5884
      Epoch 34 composite train-obj: 2.900029
            No improvement (4.1496), counter 2/5
    Epoch [35/50], Train Losses: mse: 29.5146, mae: 3.3346, huber: 2.8933, swd: 4.0382, ept: 53.8633
    Epoch [35/50], Val Losses: mse: 57.9570, mae: 4.7711, huber: 4.3211, swd: 4.5286, ept: 176.7430
    Epoch [35/50], Test Losses: mse: 42.3288, mae: 3.8784, huber: 3.4343, swd: 3.9411, ept: 197.3659
      Epoch 35 composite train-obj: 2.893327
            No improvement (4.3211), counter 3/5
    Epoch [36/50], Train Losses: mse: 29.1987, mae: 3.3086, huber: 2.8682, swd: 4.0005, ept: 53.9467
    Epoch [36/50], Val Losses: mse: 56.7345, mae: 4.7170, huber: 4.2639, swd: 4.9860, ept: 175.6238
    Epoch [36/50], Test Losses: mse: 40.9929, mae: 3.7641, huber: 3.3199, swd: 3.9819, ept: 204.7082
      Epoch 36 composite train-obj: 2.868178
            No improvement (4.2639), counter 4/5
    Epoch [37/50], Train Losses: mse: 28.9959, mae: 3.2928, huber: 2.8526, swd: 3.9505, ept: 53.9530
    Epoch [37/50], Val Losses: mse: 56.7107, mae: 4.7148, huber: 4.2643, swd: 5.2117, ept: 170.6483
    Epoch [37/50], Test Losses: mse: 41.5441, mae: 3.7840, huber: 3.3430, swd: 4.4845, ept: 200.5011
      Epoch 37 composite train-obj: 2.852565
    Epoch [37/50], Test Losses: mse: 39.3536, mae: 3.7198, huber: 3.2749, swd: 4.5011, ept: 202.1552
    Best round's Test MSE: 39.3536, MAE: 3.7198, SWD: 4.5011
    Best round's Validation MSE: 52.7981, MAE: 4.5609, SWD: 5.7685
    Best round's Test verification MSE : 39.3536, MAE: 3.7198, SWD: 4.5011
    Time taken: 91.23 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred336_20250514_2127)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 39.4891 ± 0.5953
      mae: 3.8607 ± 0.1082
      huber: 3.4109 ± 0.1044
      swd: 5.2020 ± 1.0919
      ept: 186.0669 ± 12.2120
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 53.2422 ± 0.6083
      mae: 4.6736 ± 0.0894
      huber: 4.2184 ± 0.0866
      swd: 6.0350 ± 0.7771
      ept: 162.9098 ± 7.5841
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 205.63 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred336_20250514_2127
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### huber + 0.5 SWD


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.5, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 99.4185, mae: 7.6075, huber: 7.1231, swd: 3.4429, ept: 19.6252
    Epoch [1/50], Val Losses: mse: 93.5616, mae: 7.1875, huber: 6.7060, swd: 1.6077, ept: 22.1377
    Epoch [1/50], Test Losses: mse: 84.8924, mae: 6.6483, huber: 6.1693, swd: 1.4334, ept: 24.7804
      Epoch 1 composite train-obj: 8.844522
            Val objective improved inf → 7.5098, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 80.1989, mae: 6.6022, huber: 6.1219, swd: 1.4419, ept: 31.4858
    Epoch [2/50], Val Losses: mse: 82.2437, mae: 6.6356, huber: 6.1567, swd: 1.1467, ept: 37.1009
    Epoch [2/50], Test Losses: mse: 74.3677, mae: 6.0942, huber: 5.6182, swd: 1.0416, ept: 41.5554
      Epoch 2 composite train-obj: 6.842837
            Val objective improved 7.5098 → 6.7301, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 74.1876, mae: 6.2326, huber: 5.7544, swd: 1.3422, ept: 36.7182
    Epoch [3/50], Val Losses: mse: 81.2738, mae: 6.4902, huber: 6.0121, swd: 1.2408, ept: 49.4290
    Epoch [3/50], Test Losses: mse: 72.9075, mae: 5.9616, huber: 5.4857, swd: 1.1593, ept: 52.7960
      Epoch 3 composite train-obj: 6.425445
            Val objective improved 6.7301 → 6.6325, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 69.1245, mae: 5.9209, huber: 5.4445, swd: 1.2487, ept: 40.1455
    Epoch [4/50], Val Losses: mse: 76.7894, mae: 6.1850, huber: 5.7096, swd: 1.3258, ept: 67.3129
    Epoch [4/50], Test Losses: mse: 67.1937, mae: 5.6025, huber: 5.1293, swd: 1.1438, ept: 66.2767
      Epoch 4 composite train-obj: 6.068895
            Val objective improved 6.6325 → 6.3725, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 64.7891, mae: 5.6634, huber: 5.1887, swd: 1.2338, ept: 42.6660
    Epoch [5/50], Val Losses: mse: 73.8968, mae: 5.9772, huber: 5.5034, swd: 1.2817, ept: 78.5122
    Epoch [5/50], Test Losses: mse: 64.2776, mae: 5.4003, huber: 4.9294, swd: 1.1327, ept: 80.5059
      Epoch 5 composite train-obj: 5.805623
            Val objective improved 6.3725 → 6.1443, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 61.6487, mae: 5.4665, huber: 4.9935, swd: 1.1931, ept: 44.4299
    Epoch [6/50], Val Losses: mse: 72.4914, mae: 5.9070, huber: 5.4336, swd: 1.4409, ept: 81.0694
    Epoch [6/50], Test Losses: mse: 62.0378, mae: 5.2963, huber: 4.8255, swd: 1.1702, ept: 80.6142
      Epoch 6 composite train-obj: 5.590045
            No improvement (6.1541), counter 1/5
    Epoch [7/50], Train Losses: mse: 59.3264, mae: 5.3271, huber: 4.8551, swd: 1.2048, ept: 45.4716
    Epoch [7/50], Val Losses: mse: 74.2325, mae: 5.9551, huber: 5.4827, swd: 1.2365, ept: 87.8600
    Epoch [7/50], Test Losses: mse: 63.0363, mae: 5.2901, huber: 4.8207, swd: 0.9901, ept: 91.6633
      Epoch 7 composite train-obj: 5.457520
            Val objective improved 6.1443 → 6.1009, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 56.8157, mae: 5.1582, huber: 4.6880, swd: 1.1287, ept: 47.1664
    Epoch [8/50], Val Losses: mse: 70.9175, mae: 5.7503, huber: 5.2794, swd: 1.2174, ept: 98.6062
    Epoch [8/50], Test Losses: mse: 60.2441, mae: 5.1219, huber: 4.6541, swd: 1.0403, ept: 99.4031
      Epoch 8 composite train-obj: 5.252309
            Val objective improved 6.1009 → 5.8880, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 55.1597, mae: 5.0506, huber: 4.5815, swd: 1.1575, ept: 47.7792
    Epoch [9/50], Val Losses: mse: 71.0135, mae: 5.7408, huber: 5.2698, swd: 1.2970, ept: 101.3543
    Epoch [9/50], Test Losses: mse: 58.7242, mae: 5.0440, huber: 4.5763, swd: 1.0272, ept: 103.6939
      Epoch 9 composite train-obj: 5.160241
            No improvement (5.9182), counter 1/5
    Epoch [10/50], Train Losses: mse: 53.6358, mae: 4.9464, huber: 4.4784, swd: 1.1312, ept: 48.6062
    Epoch [10/50], Val Losses: mse: 70.9053, mae: 5.7197, huber: 5.2496, swd: 1.3563, ept: 105.1287
    Epoch [10/50], Test Losses: mse: 58.2527, mae: 5.0032, huber: 4.5362, swd: 1.0093, ept: 109.6444
      Epoch 10 composite train-obj: 5.043983
            No improvement (5.9277), counter 2/5
    Epoch [11/50], Train Losses: mse: 52.1253, mae: 4.8437, huber: 4.3768, swd: 1.1191, ept: 49.4143
    Epoch [11/50], Val Losses: mse: 70.4451, mae: 5.6811, huber: 5.2108, swd: 1.3024, ept: 105.8284
    Epoch [11/50], Test Losses: mse: 58.1701, mae: 4.9749, huber: 4.5077, swd: 1.0569, ept: 113.4557
      Epoch 11 composite train-obj: 4.936311
            Val objective improved 5.8880 → 5.8620, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 50.9137, mae: 4.7589, huber: 4.2930, swd: 1.1144, ept: 49.7878
    Epoch [12/50], Val Losses: mse: 70.8775, mae: 5.6775, huber: 5.2087, swd: 1.2747, ept: 108.8096
    Epoch [12/50], Test Losses: mse: 57.4998, mae: 4.8955, huber: 4.4307, swd: 1.0331, ept: 117.4781
      Epoch 12 composite train-obj: 4.850259
            Val objective improved 5.8620 → 5.8460, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 49.5689, mae: 4.6651, huber: 4.2004, swd: 1.0913, ept: 50.5392
    Epoch [13/50], Val Losses: mse: 70.0702, mae: 5.6110, huber: 5.1426, swd: 1.2515, ept: 113.0340
    Epoch [13/50], Test Losses: mse: 55.8460, mae: 4.7950, huber: 4.3310, swd: 0.9823, ept: 125.0435
      Epoch 13 composite train-obj: 4.746054
            Val objective improved 5.8460 → 5.7683, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 48.3385, mae: 4.5838, huber: 4.1202, swd: 1.0738, ept: 50.9688
    Epoch [14/50], Val Losses: mse: 69.1279, mae: 5.5383, huber: 5.0714, swd: 1.3485, ept: 114.4867
    Epoch [14/50], Test Losses: mse: 55.1543, mae: 4.7119, huber: 4.2506, swd: 0.9998, ept: 127.8170
      Epoch 14 composite train-obj: 4.657107
            Val objective improved 5.7683 → 5.7457, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 47.4188, mae: 4.5190, huber: 4.0562, swd: 1.0739, ept: 51.2138
    Epoch [15/50], Val Losses: mse: 69.2926, mae: 5.5505, huber: 5.0842, swd: 1.2915, ept: 114.1554
    Epoch [15/50], Test Losses: mse: 55.3999, mae: 4.7281, huber: 4.2662, swd: 0.9392, ept: 129.8025
      Epoch 15 composite train-obj: 4.593113
            Val objective improved 5.7457 → 5.7299, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 46.4477, mae: 4.4482, huber: 3.9862, swd: 1.0611, ept: 51.6716
    Epoch [16/50], Val Losses: mse: 69.4956, mae: 5.5479, huber: 5.0814, swd: 1.3326, ept: 116.8917
    Epoch [16/50], Test Losses: mse: 55.7371, mae: 4.7378, huber: 4.2761, swd: 1.1283, ept: 133.3326
      Epoch 16 composite train-obj: 4.516796
            No improvement (5.7477), counter 1/5
    Epoch [17/50], Train Losses: mse: 45.4452, mae: 4.3793, huber: 3.9183, swd: 1.0493, ept: 52.0248
    Epoch [17/50], Val Losses: mse: 72.8667, mae: 5.7022, huber: 5.2353, swd: 1.4438, ept: 110.8540
    Epoch [17/50], Test Losses: mse: 56.9418, mae: 4.7909, huber: 4.3289, swd: 1.1277, ept: 123.4351
      Epoch 17 composite train-obj: 4.442994
            No improvement (5.9572), counter 2/5
    Epoch [18/50], Train Losses: mse: 44.6996, mae: 4.3254, huber: 3.8653, swd: 1.0362, ept: 51.9287
    Epoch [18/50], Val Losses: mse: 69.7494, mae: 5.5697, huber: 5.1006, swd: 1.3701, ept: 119.9222
    Epoch [18/50], Test Losses: mse: 55.5145, mae: 4.7910, huber: 4.3253, swd: 1.0231, ept: 132.9436
      Epoch 18 composite train-obj: 4.383378
            No improvement (5.7856), counter 3/5
    Epoch [19/50], Train Losses: mse: 43.8073, mae: 4.2670, huber: 3.8077, swd: 1.0191, ept: 52.4182
    Epoch [19/50], Val Losses: mse: 68.9375, mae: 5.4899, huber: 5.0247, swd: 1.3805, ept: 124.2885
    Epoch [19/50], Test Losses: mse: 53.4472, mae: 4.6205, huber: 4.1601, swd: 1.1570, ept: 136.1467
      Epoch 19 composite train-obj: 4.317231
            Val objective improved 5.7299 → 5.7149, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 43.1379, mae: 4.2153, huber: 3.7568, swd: 1.0129, ept: 52.3504
    Epoch [20/50], Val Losses: mse: 68.4896, mae: 5.4772, huber: 5.0107, swd: 1.3185, ept: 123.7479
    Epoch [20/50], Test Losses: mse: 53.7221, mae: 4.6324, huber: 4.1708, swd: 0.9619, ept: 137.0253
      Epoch 20 composite train-obj: 4.263260
            Val objective improved 5.7149 → 5.6699, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 42.3753, mae: 4.1536, huber: 3.6963, swd: 0.9907, ept: 52.7317
    Epoch [21/50], Val Losses: mse: 68.7874, mae: 5.5225, huber: 5.0557, swd: 1.2636, ept: 119.9178
    Epoch [21/50], Test Losses: mse: 55.9022, mae: 4.8097, huber: 4.3455, swd: 1.1528, ept: 129.1979
      Epoch 21 composite train-obj: 4.191686
            No improvement (5.6874), counter 1/5
    Epoch [22/50], Train Losses: mse: 41.6867, mae: 4.1044, huber: 3.6480, swd: 0.9733, ept: 52.8514
    Epoch [22/50], Val Losses: mse: 69.5144, mae: 5.5689, huber: 5.0999, swd: 1.5256, ept: 123.3411
    Epoch [22/50], Test Losses: mse: 53.8809, mae: 4.7110, huber: 4.2462, swd: 1.1470, ept: 138.7145
      Epoch 22 composite train-obj: 4.134606
            No improvement (5.8627), counter 2/5
    Epoch [23/50], Train Losses: mse: 41.2237, mae: 4.0680, huber: 3.6122, swd: 0.9832, ept: 52.7803
    Epoch [23/50], Val Losses: mse: 69.1885, mae: 5.4652, huber: 5.0015, swd: 1.3627, ept: 122.0403
    Epoch [23/50], Test Losses: mse: 52.7864, mae: 4.5480, huber: 4.0889, swd: 0.9386, ept: 134.0508
      Epoch 23 composite train-obj: 4.103791
            No improvement (5.6829), counter 3/5
    Epoch [24/50], Train Losses: mse: 40.7402, mae: 4.0351, huber: 3.5799, swd: 0.9691, ept: 52.9836
    Epoch [24/50], Val Losses: mse: 71.0695, mae: 5.6703, huber: 5.1983, swd: 1.3250, ept: 116.7602
    Epoch [24/50], Test Losses: mse: 53.8876, mae: 4.7760, huber: 4.3076, swd: 1.0691, ept: 127.3194
      Epoch 24 composite train-obj: 4.064422
            No improvement (5.8608), counter 4/5
    Epoch [25/50], Train Losses: mse: 40.2163, mae: 3.9945, huber: 3.5400, swd: 0.9470, ept: 52.9969
    Epoch [25/50], Val Losses: mse: 71.9173, mae: 5.6445, huber: 5.1769, swd: 1.4300, ept: 124.9905
    Epoch [25/50], Test Losses: mse: 55.4772, mae: 4.7329, huber: 4.2698, swd: 1.0795, ept: 137.1361
      Epoch 25 composite train-obj: 4.013479
    Epoch [25/50], Test Losses: mse: 53.7221, mae: 4.6324, huber: 4.1708, swd: 0.9619, ept: 137.0253
    Best round's Test MSE: 53.7221, MAE: 4.6324, SWD: 0.9619
    Best round's Validation MSE: 68.4896, MAE: 5.4772, SWD: 1.3185
    Best round's Test verification MSE : 53.7221, MAE: 4.6324, SWD: 0.9619
    Time taken: 63.28 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 94.9488, mae: 7.3358, huber: 6.8527, swd: 3.2759, ept: 23.8758
    Epoch [1/50], Val Losses: mse: 90.8130, mae: 7.0778, huber: 6.5965, swd: 1.6211, ept: 24.7941
    Epoch [1/50], Test Losses: mse: 83.3269, mae: 6.6268, huber: 6.1477, swd: 1.5211, ept: 25.3622
      Epoch 1 composite train-obj: 8.490650
            Val objective improved inf → 7.4070, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 79.8484, mae: 6.5507, huber: 6.0708, swd: 1.4964, ept: 33.0141
    Epoch [2/50], Val Losses: mse: 83.8976, mae: 6.7096, huber: 6.2301, swd: 1.2031, ept: 36.4376
    Epoch [2/50], Test Losses: mse: 74.7704, mae: 6.1181, huber: 5.6416, swd: 1.0778, ept: 41.6710
      Epoch 2 composite train-obj: 6.818970
            Val objective improved 7.4070 → 6.8317, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 73.6343, mae: 6.2127, huber: 5.7343, swd: 1.3754, ept: 36.3498
    Epoch [3/50], Val Losses: mse: 76.9321, mae: 6.2960, huber: 5.8188, swd: 1.3614, ept: 53.4681
    Epoch [3/50], Test Losses: mse: 68.5854, mae: 5.7548, huber: 5.2803, swd: 1.0638, ept: 54.9858
      Epoch 3 composite train-obj: 6.421976
            Val objective improved 6.8317 → 6.4995, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 68.8769, mae: 5.9211, huber: 5.4443, swd: 1.3091, ept: 39.7781
    Epoch [4/50], Val Losses: mse: 75.2078, mae: 6.0962, huber: 5.6211, swd: 1.4231, ept: 66.6973
    Epoch [4/50], Test Losses: mse: 66.4408, mae: 5.5614, huber: 5.0888, swd: 1.0113, ept: 67.9326
      Epoch 4 composite train-obj: 6.098850
            Val objective improved 6.4995 → 6.3326, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 65.6631, mae: 5.7144, huber: 5.2390, swd: 1.3074, ept: 42.4004
    Epoch [5/50], Val Losses: mse: 73.4179, mae: 5.9951, huber: 5.5211, swd: 1.5212, ept: 74.4896
    Epoch [5/50], Test Losses: mse: 62.6002, mae: 5.3623, huber: 4.8912, swd: 1.2036, ept: 76.3993
      Epoch 5 composite train-obj: 5.892704
            Val objective improved 6.3326 → 6.2817, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 63.0315, mae: 5.5310, huber: 5.0571, swd: 1.2438, ept: 44.8223
    Epoch [6/50], Val Losses: mse: 73.1802, mae: 5.9272, huber: 5.4537, swd: 1.4925, ept: 84.6753
    Epoch [6/50], Test Losses: mse: 62.7244, mae: 5.3195, huber: 4.8485, swd: 1.2859, ept: 87.0444
      Epoch 6 composite train-obj: 5.679007
            Val objective improved 6.2817 → 6.2000, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 61.0844, mae: 5.3986, huber: 4.9260, swd: 1.2523, ept: 45.8290
    Epoch [7/50], Val Losses: mse: 72.4742, mae: 5.8426, huber: 5.3705, swd: 1.4073, ept: 90.5187
    Epoch [7/50], Test Losses: mse: 61.2864, mae: 5.1753, huber: 4.7063, swd: 0.9977, ept: 95.9971
      Epoch 7 composite train-obj: 5.552113
            Val objective improved 6.2000 → 6.0741, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 58.9020, mae: 5.2519, huber: 4.7808, swd: 1.1892, ept: 46.8933
    Epoch [8/50], Val Losses: mse: 70.3413, mae: 5.7200, huber: 5.2490, swd: 1.3268, ept: 97.8588
    Epoch [8/50], Test Losses: mse: 59.3967, mae: 5.0589, huber: 4.5909, swd: 1.0550, ept: 102.4116
      Epoch 8 composite train-obj: 5.375349
            Val objective improved 6.0741 → 5.9124, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 56.9736, mae: 5.1251, huber: 4.6553, swd: 1.1563, ept: 48.1594
    Epoch [9/50], Val Losses: mse: 70.7556, mae: 5.6859, huber: 5.2159, swd: 1.4113, ept: 100.6707
    Epoch [9/50], Test Losses: mse: 59.3756, mae: 4.9935, huber: 4.5273, swd: 1.0214, ept: 106.0824
      Epoch 9 composite train-obj: 5.233509
            No improvement (5.9215), counter 1/5
    Epoch [10/50], Train Losses: mse: 54.9508, mae: 4.9914, huber: 4.5231, swd: 1.1283, ept: 48.8140
    Epoch [10/50], Val Losses: mse: 70.9149, mae: 5.6626, huber: 5.1934, swd: 1.3483, ept: 105.6096
    Epoch [10/50], Test Losses: mse: 59.1811, mae: 4.9623, huber: 4.4964, swd: 0.9318, ept: 110.9543
      Epoch 10 composite train-obj: 5.087231
            Val objective improved 5.9124 → 5.8676, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 53.3961, mae: 4.8963, huber: 4.4289, swd: 1.1308, ept: 49.3320
    Epoch [11/50], Val Losses: mse: 68.8824, mae: 5.5649, huber: 5.0956, swd: 1.2931, ept: 107.5117
    Epoch [11/50], Test Losses: mse: 58.2979, mae: 4.9417, huber: 4.4758, swd: 0.9766, ept: 112.5141
      Epoch 11 composite train-obj: 4.994294
            Val objective improved 5.8676 → 5.7421, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 51.8099, mae: 4.7844, huber: 4.3184, swd: 1.0993, ept: 50.1368
    Epoch [12/50], Val Losses: mse: 69.9202, mae: 5.5925, huber: 5.1241, swd: 1.3508, ept: 110.9175
    Epoch [12/50], Test Losses: mse: 56.6443, mae: 4.8555, huber: 4.3903, swd: 0.9129, ept: 119.6889
      Epoch 12 composite train-obj: 4.868091
            No improvement (5.7995), counter 1/5
    Epoch [13/50], Train Losses: mse: 50.6443, mae: 4.7111, huber: 4.2459, swd: 1.0966, ept: 50.6479
    Epoch [13/50], Val Losses: mse: 70.7319, mae: 5.6050, huber: 5.1379, swd: 1.3778, ept: 113.7676
    Epoch [13/50], Test Losses: mse: 57.3092, mae: 4.8258, huber: 4.3625, swd: 0.9503, ept: 124.7990
      Epoch 13 composite train-obj: 4.794203
            No improvement (5.8267), counter 2/5
    Epoch [14/50], Train Losses: mse: 49.2243, mae: 4.6071, huber: 4.1433, swd: 1.0803, ept: 51.1347
    Epoch [14/50], Val Losses: mse: 67.6146, mae: 5.4803, huber: 5.0136, swd: 1.5175, ept: 119.9477
    Epoch [14/50], Test Losses: mse: 54.7521, mae: 4.7597, huber: 4.2959, swd: 1.0285, ept: 125.5649
      Epoch 14 composite train-obj: 4.683445
            No improvement (5.7724), counter 3/5
    Epoch [15/50], Train Losses: mse: 48.0639, mae: 4.5274, huber: 4.0646, swd: 1.0590, ept: 51.5137
    Epoch [15/50], Val Losses: mse: 66.9653, mae: 5.4198, huber: 4.9542, swd: 1.5810, ept: 120.4138
    Epoch [15/50], Test Losses: mse: 54.2223, mae: 4.6976, huber: 4.2355, swd: 1.0233, ept: 127.6007
      Epoch 15 composite train-obj: 4.594059
            No improvement (5.7446), counter 4/5
    Epoch [16/50], Train Losses: mse: 46.9228, mae: 4.4503, huber: 3.9886, swd: 1.0505, ept: 52.0321
    Epoch [16/50], Val Losses: mse: 67.8315, mae: 5.4739, huber: 5.0070, swd: 1.6771, ept: 118.3096
    Epoch [16/50], Test Losses: mse: 54.8673, mae: 4.6927, huber: 4.2300, swd: 1.0487, ept: 128.3860
      Epoch 16 composite train-obj: 4.513828
    Epoch [16/50], Test Losses: mse: 58.2979, mae: 4.9417, huber: 4.4758, swd: 0.9766, ept: 112.5141
    Best round's Test MSE: 58.2979, MAE: 4.9417, SWD: 0.9766
    Best round's Validation MSE: 68.8824, MAE: 5.5649, SWD: 1.2931
    Best round's Test verification MSE : 58.2979, MAE: 4.9417, SWD: 0.9766
    Time taken: 41.15 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 95.9484, mae: 7.3946, huber: 6.9112, swd: 3.3716, ept: 22.6101
    Epoch [1/50], Val Losses: mse: 90.2564, mae: 7.0182, huber: 6.5376, swd: 1.4009, ept: 25.3820
    Epoch [1/50], Test Losses: mse: 82.9340, mae: 6.5940, huber: 6.1152, swd: 1.2239, ept: 25.8719
      Epoch 1 composite train-obj: 8.596960
            Val objective improved inf → 7.2381, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 80.7943, mae: 6.6131, huber: 6.1328, swd: 1.6251, ept: 32.4372
    Epoch [2/50], Val Losses: mse: 83.6593, mae: 6.7072, huber: 6.2281, swd: 1.3947, ept: 36.4589
    Epoch [2/50], Test Losses: mse: 76.0566, mae: 6.2273, huber: 5.7505, swd: 1.2914, ept: 40.9433
      Epoch 2 composite train-obj: 6.945364
            Val objective improved 7.2381 → 6.9255, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 74.5009, mae: 6.2613, huber: 5.7829, swd: 1.4527, ept: 35.7716
    Epoch [3/50], Val Losses: mse: 79.3580, mae: 6.3881, huber: 5.9113, swd: 1.3341, ept: 52.9546
    Epoch [3/50], Test Losses: mse: 71.0394, mae: 5.8701, huber: 5.3956, swd: 1.1537, ept: 56.6517
      Epoch 3 composite train-obj: 6.509287
            Val objective improved 6.9255 → 6.5783, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 68.7598, mae: 5.9297, huber: 5.4530, swd: 1.4180, ept: 39.1108
    Epoch [4/50], Val Losses: mse: 74.3423, mae: 6.1105, huber: 5.6348, swd: 1.3546, ept: 65.5083
    Epoch [4/50], Test Losses: mse: 66.9571, mae: 5.6294, huber: 5.1557, swd: 1.1774, ept: 66.4847
      Epoch 4 composite train-obj: 6.162049
            Val objective improved 6.5783 → 6.3121, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 64.4507, mae: 5.6637, huber: 5.1889, swd: 1.3423, ept: 41.8284
    Epoch [5/50], Val Losses: mse: 74.0423, mae: 6.0302, huber: 5.5553, swd: 1.3664, ept: 72.8153
    Epoch [5/50], Test Losses: mse: 66.5489, mae: 5.5538, huber: 5.0811, swd: 1.2020, ept: 71.6392
      Epoch 5 composite train-obj: 5.860079
            Val objective improved 6.3121 → 6.2385, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 61.4436, mae: 5.4767, huber: 5.0035, swd: 1.3119, ept: 43.7154
    Epoch [6/50], Val Losses: mse: 68.0883, mae: 5.7353, huber: 5.2631, swd: 1.4948, ept: 86.6862
    Epoch [6/50], Test Losses: mse: 60.0447, mae: 5.2046, huber: 4.7345, swd: 1.1611, ept: 89.6880
      Epoch 6 composite train-obj: 5.659445
            Val objective improved 6.2385 → 6.0105, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 58.8501, mae: 5.3122, huber: 4.8404, swd: 1.3064, ept: 45.1698
    Epoch [7/50], Val Losses: mse: 69.0736, mae: 5.7540, huber: 5.2819, swd: 1.3832, ept: 89.8733
    Epoch [7/50], Test Losses: mse: 61.3695, mae: 5.2196, huber: 4.7502, swd: 1.1848, ept: 93.4554
      Epoch 7 composite train-obj: 5.493549
            Val objective improved 6.0105 → 5.9736, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 56.7370, mae: 5.1728, huber: 4.7022, swd: 1.2885, ept: 46.5746
    Epoch [8/50], Val Losses: mse: 67.0410, mae: 5.6174, huber: 5.1461, swd: 1.5857, ept: 96.2162
    Epoch [8/50], Test Losses: mse: 57.8099, mae: 5.0457, huber: 4.5770, swd: 1.2766, ept: 102.1951
      Epoch 8 composite train-obj: 5.346439
            Val objective improved 5.9736 → 5.9390, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 54.9155, mae: 5.0483, huber: 4.5790, swd: 1.2668, ept: 47.4607
    Epoch [9/50], Val Losses: mse: 69.2790, mae: 5.6962, huber: 5.2248, swd: 1.3665, ept: 101.6425
    Epoch [9/50], Test Losses: mse: 58.1055, mae: 4.9962, huber: 4.5289, swd: 1.1597, ept: 111.7577
      Epoch 9 composite train-obj: 5.212396
            Val objective improved 5.9390 → 5.9081, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 53.2015, mae: 4.9312, huber: 4.4630, swd: 1.2272, ept: 48.2672
    Epoch [10/50], Val Losses: mse: 67.6923, mae: 5.5525, huber: 5.0838, swd: 1.3969, ept: 109.0079
    Epoch [10/50], Test Losses: mse: 56.9571, mae: 4.9013, huber: 4.4363, swd: 1.0797, ept: 116.1059
      Epoch 10 composite train-obj: 5.076625
            Val objective improved 5.9081 → 5.7822, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 51.6234, mae: 4.8281, huber: 4.3611, swd: 1.2057, ept: 48.7894
    Epoch [11/50], Val Losses: mse: 67.3132, mae: 5.5025, huber: 5.0349, swd: 1.3301, ept: 113.2011
    Epoch [11/50], Test Losses: mse: 56.4725, mae: 4.8470, huber: 4.3836, swd: 1.0392, ept: 119.9887
      Epoch 11 composite train-obj: 4.963940
            Val objective improved 5.7822 → 5.6999, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 50.5497, mae: 4.7498, huber: 4.2837, swd: 1.2030, ept: 49.3813
    Epoch [12/50], Val Losses: mse: 70.0915, mae: 5.5855, huber: 5.1186, swd: 1.3756, ept: 108.6656
    Epoch [12/50], Test Losses: mse: 57.1341, mae: 4.8257, huber: 4.3632, swd: 1.0896, ept: 118.1886
      Epoch 12 composite train-obj: 4.885203
            No improvement (5.8065), counter 1/5
    Epoch [13/50], Train Losses: mse: 49.5399, mae: 4.6736, huber: 4.2086, swd: 1.1850, ept: 49.6169
    Epoch [13/50], Val Losses: mse: 67.6111, mae: 5.5036, huber: 5.0355, swd: 1.4405, ept: 115.8163
    Epoch [13/50], Test Losses: mse: 56.0523, mae: 4.8082, huber: 4.3439, swd: 1.0295, ept: 125.1082
      Epoch 13 composite train-obj: 4.801075
            No improvement (5.7558), counter 2/5
    Epoch [14/50], Train Losses: mse: 48.4012, mae: 4.5839, huber: 4.1202, swd: 1.1388, ept: 49.9910
    Epoch [14/50], Val Losses: mse: 68.8753, mae: 5.5204, huber: 5.0535, swd: 1.3884, ept: 117.4750
    Epoch [14/50], Test Losses: mse: 54.9497, mae: 4.7261, huber: 4.2633, swd: 1.0114, ept: 129.7527
      Epoch 14 composite train-obj: 4.689588
            No improvement (5.7477), counter 3/5
    Epoch [15/50], Train Losses: mse: 47.6169, mae: 4.5269, huber: 4.0639, swd: 1.1403, ept: 50.2452
    Epoch [15/50], Val Losses: mse: 68.7887, mae: 5.5332, huber: 5.0658, swd: 1.4349, ept: 117.7540
    Epoch [15/50], Test Losses: mse: 54.9557, mae: 4.7899, huber: 4.3261, swd: 1.0949, ept: 129.1619
      Epoch 15 composite train-obj: 4.634070
            No improvement (5.7833), counter 4/5
    Epoch [16/50], Train Losses: mse: 46.7527, mae: 4.4628, huber: 4.0008, swd: 1.1213, ept: 50.6041
    Epoch [16/50], Val Losses: mse: 71.7040, mae: 5.6618, huber: 5.1934, swd: 1.4810, ept: 113.7172
    Epoch [16/50], Test Losses: mse: 56.2656, mae: 4.8382, huber: 4.3741, swd: 1.1686, ept: 125.6456
      Epoch 16 composite train-obj: 4.561415
    Epoch [16/50], Test Losses: mse: 56.4725, mae: 4.8470, huber: 4.3836, swd: 1.0392, ept: 119.9887
    Best round's Test MSE: 56.4725, MAE: 4.8470, SWD: 1.0392
    Best round's Validation MSE: 67.3132, MAE: 5.5025, SWD: 1.3301
    Best round's Test verification MSE : 56.4725, MAE: 4.8470, SWD: 1.0392
    Time taken: 42.02 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred336_20250514_2131)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 56.1641 ± 1.8807
      mae: 4.8070 ± 0.1294
      huber: 4.3434 ± 0.1277
      swd: 0.9926 ± 0.0335
      ept: 123.1760 ± 10.2573
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 68.2284 ± 0.6667
      mae: 5.5149 ± 0.0369
      huber: 5.0470 ± 0.0357
      swd: 1.3139 ± 0.0155
      ept: 114.8202 ± 6.7265
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 146.54 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred336_20250514_2131
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE


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
    loss_backward_weights = [1.0, 0.0, 0.0, 0.0, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.0, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 69.8124, mae: 6.4039, huber: 5.9226, swd: 12.5485, ept: 34.6618
    Epoch [1/50], Val Losses: mse: 63.2514, mae: 6.0390, huber: 5.5608, swd: 16.8713, ept: 58.0080
    Epoch [1/50], Test Losses: mse: 57.2815, mae: 5.6515, huber: 5.1753, swd: 15.4975, ept: 64.0129
      Epoch 1 composite train-obj: 69.812372
            Val objective improved inf → 63.2514, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 55.5490, mae: 5.6150, huber: 5.1376, swd: 11.8586, ept: 42.2866
    Epoch [2/50], Val Losses: mse: 56.6323, mae: 5.5594, huber: 5.0844, swd: 13.6182, ept: 83.0660
    Epoch [2/50], Test Losses: mse: 51.1747, mae: 5.2271, huber: 4.7542, swd: 13.8932, ept: 89.7078
      Epoch 2 composite train-obj: 55.549037
            Val objective improved 63.2514 → 56.6323, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.4620, mae: 5.2765, huber: 4.8014, swd: 10.8407, ept: 45.5166
    Epoch [3/50], Val Losses: mse: 56.6708, mae: 5.4539, huber: 4.9813, swd: 11.4020, ept: 96.6969
    Epoch [3/50], Test Losses: mse: 48.6815, mae: 4.9897, huber: 4.5199, swd: 12.0133, ept: 102.6971
      Epoch 3 composite train-obj: 50.462013
            No improvement (56.6708), counter 1/5
    Epoch [4/50], Train Losses: mse: 47.5777, mae: 5.0681, huber: 4.5948, swd: 9.7992, ept: 47.8742
    Epoch [4/50], Val Losses: mse: 56.8030, mae: 5.4445, huber: 4.9726, swd: 10.9331, ept: 96.8614
    Epoch [4/50], Test Losses: mse: 48.9699, mae: 5.0042, huber: 4.5346, swd: 11.0769, ept: 104.3527
      Epoch 4 composite train-obj: 47.577665
            No improvement (56.8030), counter 2/5
    Epoch [5/50], Train Losses: mse: 44.9353, mae: 4.8778, huber: 4.4062, swd: 9.1514, ept: 49.5104
    Epoch [5/50], Val Losses: mse: 56.8936, mae: 5.4004, huber: 4.9295, swd: 10.4758, ept: 102.4590
    Epoch [5/50], Test Losses: mse: 48.5460, mae: 4.9412, huber: 4.4721, swd: 11.2286, ept: 108.1458
      Epoch 5 composite train-obj: 44.935284
            No improvement (56.8936), counter 3/5
    Epoch [6/50], Train Losses: mse: 42.9615, mae: 4.7306, huber: 4.2606, swd: 8.4997, ept: 50.5462
    Epoch [6/50], Val Losses: mse: 55.8629, mae: 5.3072, huber: 4.8374, swd: 9.7033, ept: 109.3633
    Epoch [6/50], Test Losses: mse: 48.7439, mae: 4.9010, huber: 4.4323, swd: 9.9518, ept: 120.1367
      Epoch 6 composite train-obj: 42.961506
            Val objective improved 56.6323 → 55.8629, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 41.5254, mae: 4.6188, huber: 4.1501, swd: 7.9741, ept: 51.2803
    Epoch [7/50], Val Losses: mse: 56.4435, mae: 5.2692, huber: 4.8040, swd: 9.7189, ept: 113.2446
    Epoch [7/50], Test Losses: mse: 47.2940, mae: 4.7500, huber: 4.2865, swd: 9.7052, ept: 120.4461
      Epoch 7 composite train-obj: 41.525421
            No improvement (56.4435), counter 1/5
    Epoch [8/50], Train Losses: mse: 40.0239, mae: 4.5043, huber: 4.0369, swd: 7.5787, ept: 51.8765
    Epoch [8/50], Val Losses: mse: 56.9139, mae: 5.2867, huber: 4.8188, swd: 7.9883, ept: 118.4857
    Epoch [8/50], Test Losses: mse: 48.0613, mae: 4.8175, huber: 4.3510, swd: 7.8229, ept: 126.7132
      Epoch 8 composite train-obj: 40.023904
            No improvement (56.9139), counter 2/5
    Epoch [9/50], Train Losses: mse: 39.0879, mae: 4.4345, huber: 3.9679, swd: 7.2775, ept: 52.1510
    Epoch [9/50], Val Losses: mse: 57.6321, mae: 5.3243, huber: 4.8560, swd: 8.4300, ept: 107.5720
    Epoch [9/50], Test Losses: mse: 47.5940, mae: 4.7794, huber: 4.3120, swd: 8.7932, ept: 117.3913
      Epoch 9 composite train-obj: 39.087874
            No improvement (57.6321), counter 3/5
    Epoch [10/50], Train Losses: mse: 38.0319, mae: 4.3491, huber: 3.8837, swd: 6.9761, ept: 52.6036
    Epoch [10/50], Val Losses: mse: 58.1982, mae: 5.3308, huber: 4.8632, swd: 8.3799, ept: 118.9155
    Epoch [10/50], Test Losses: mse: 46.3811, mae: 4.6687, huber: 4.2050, swd: 9.4647, ept: 130.1702
      Epoch 10 composite train-obj: 38.031864
            No improvement (58.1982), counter 4/5
    Epoch [11/50], Train Losses: mse: 37.2352, mae: 4.2891, huber: 3.8243, swd: 6.7780, ept: 53.0755
    Epoch [11/50], Val Losses: mse: 57.1300, mae: 5.2471, huber: 4.7802, swd: 7.2863, ept: 122.3165
    Epoch [11/50], Test Losses: mse: 47.1236, mae: 4.6780, huber: 4.2136, swd: 7.5141, ept: 131.2753
      Epoch 11 composite train-obj: 37.235228
    Epoch [11/50], Test Losses: mse: 48.7439, mae: 4.9010, huber: 4.4323, swd: 9.9518, ept: 120.1367
    Best round's Test MSE: 48.7439, MAE: 4.9010, SWD: 9.9518
    Best round's Validation MSE: 55.8629, MAE: 5.3072, SWD: 9.7033
    Best round's Test verification MSE : 48.7439, MAE: 4.9010, SWD: 9.9518
    Time taken: 29.41 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.0453, mae: 6.4086, huber: 5.9273, swd: 12.1739, ept: 34.7994
    Epoch [1/50], Val Losses: mse: 61.8554, mae: 5.9956, huber: 5.5175, swd: 17.1201, ept: 60.5688
    Epoch [1/50], Test Losses: mse: 56.5388, mae: 5.7162, huber: 5.2386, swd: 17.4894, ept: 59.4349
      Epoch 1 composite train-obj: 70.045256
            Val objective improved inf → 61.8554, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 55.9656, mae: 5.6397, huber: 5.1621, swd: 11.4473, ept: 42.7346
    Epoch [2/50], Val Losses: mse: 58.7910, mae: 5.7082, huber: 5.2322, swd: 12.4119, ept: 83.8630
    Epoch [2/50], Test Losses: mse: 50.3985, mae: 5.2434, huber: 4.7692, swd: 13.2437, ept: 83.6318
      Epoch 2 composite train-obj: 55.965600
            Val objective improved 61.8554 → 58.7910, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.4148, mae: 5.2688, huber: 4.7938, swd: 9.9908, ept: 46.2057
    Epoch [3/50], Val Losses: mse: 53.9421, mae: 5.4019, huber: 4.9280, swd: 11.7290, ept: 94.9102
    Epoch [3/50], Test Losses: mse: 46.7021, mae: 4.9455, huber: 4.4743, swd: 11.3938, ept: 104.2478
      Epoch 3 composite train-obj: 50.414848
            Val objective improved 58.7910 → 53.9421, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 47.0880, mae: 5.0383, huber: 4.5652, swd: 9.1951, ept: 48.0227
    Epoch [4/50], Val Losses: mse: 53.6430, mae: 5.2770, huber: 4.8054, swd: 10.2253, ept: 111.6322
    Epoch [4/50], Test Losses: mse: 45.8461, mae: 4.7957, huber: 4.3268, swd: 10.4738, ept: 121.7996
      Epoch 4 composite train-obj: 47.088011
            Val objective improved 53.9421 → 53.6430, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 44.5938, mae: 4.8582, huber: 4.3868, swd: 8.4640, ept: 49.4625
    Epoch [5/50], Val Losses: mse: 54.2153, mae: 5.2960, huber: 4.8264, swd: 9.9386, ept: 110.7081
    Epoch [5/50], Test Losses: mse: 45.5256, mae: 4.8212, huber: 4.3533, swd: 10.1715, ept: 117.5453
      Epoch 5 composite train-obj: 44.593759
            No improvement (54.2153), counter 1/5
    Epoch [6/50], Train Losses: mse: 42.3951, mae: 4.6936, huber: 4.2239, swd: 7.9622, ept: 50.4879
    Epoch [6/50], Val Losses: mse: 56.2617, mae: 5.3115, huber: 4.8420, swd: 8.4169, ept: 116.4203
    Epoch [6/50], Test Losses: mse: 46.9870, mae: 4.8293, huber: 4.3616, swd: 9.5256, ept: 122.1595
      Epoch 6 composite train-obj: 42.395086
            No improvement (56.2617), counter 2/5
    Epoch [7/50], Train Losses: mse: 41.0040, mae: 4.5855, huber: 4.1169, swd: 7.6243, ept: 51.1895
    Epoch [7/50], Val Losses: mse: 53.6950, mae: 5.1439, huber: 4.6767, swd: 8.9605, ept: 117.4708
    Epoch [7/50], Test Losses: mse: 43.6885, mae: 4.5589, huber: 4.0952, swd: 8.5646, ept: 132.0516
      Epoch 7 composite train-obj: 41.004022
            No improvement (53.6950), counter 3/5
    Epoch [8/50], Train Losses: mse: 39.1363, mae: 4.4464, huber: 3.9794, swd: 7.1079, ept: 51.4407
    Epoch [8/50], Val Losses: mse: 53.4482, mae: 5.1168, huber: 4.6501, swd: 8.6503, ept: 123.3447
    Epoch [8/50], Test Losses: mse: 44.8054, mae: 4.5907, huber: 4.1271, swd: 8.3159, ept: 130.0997
      Epoch 8 composite train-obj: 39.136340
            Val objective improved 53.6430 → 53.4482, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 38.0054, mae: 4.3587, huber: 3.8928, swd: 6.7783, ept: 52.1600
    Epoch [9/50], Val Losses: mse: 53.9614, mae: 5.1068, huber: 4.6400, swd: 8.1722, ept: 121.0060
    Epoch [9/50], Test Losses: mse: 43.9607, mae: 4.5131, huber: 4.0507, swd: 7.9844, ept: 128.6226
      Epoch 9 composite train-obj: 38.005372
            No improvement (53.9614), counter 1/5
    Epoch [10/50], Train Losses: mse: 37.0348, mae: 4.2815, huber: 3.8166, swd: 6.5762, ept: 52.3195
    Epoch [10/50], Val Losses: mse: 56.0109, mae: 5.1861, huber: 4.7189, swd: 7.8480, ept: 125.5811
    Epoch [10/50], Test Losses: mse: 42.8924, mae: 4.4261, huber: 3.9634, swd: 8.1797, ept: 138.9155
      Epoch 10 composite train-obj: 37.034830
            No improvement (56.0109), counter 2/5
    Epoch [11/50], Train Losses: mse: 36.1275, mae: 4.2096, huber: 3.7456, swd: 6.2773, ept: 52.6762
    Epoch [11/50], Val Losses: mse: 56.1732, mae: 5.2218, huber: 4.7524, swd: 7.0043, ept: 123.9436
    Epoch [11/50], Test Losses: mse: 45.3794, mae: 4.6558, huber: 4.1879, swd: 7.9335, ept: 136.0978
      Epoch 11 composite train-obj: 36.127531
            No improvement (56.1732), counter 3/5
    Epoch [12/50], Train Losses: mse: 35.0637, mae: 4.1308, huber: 3.6677, swd: 5.9992, ept: 53.0090
    Epoch [12/50], Val Losses: mse: 56.7588, mae: 5.2036, huber: 4.7361, swd: 6.8829, ept: 130.7117
    Epoch [12/50], Test Losses: mse: 42.9317, mae: 4.4272, huber: 3.9639, swd: 6.9934, ept: 142.7196
      Epoch 12 composite train-obj: 35.063710
            No improvement (56.7588), counter 4/5
    Epoch [13/50], Train Losses: mse: 34.4279, mae: 4.0787, huber: 3.6164, swd: 5.8224, ept: 53.2518
    Epoch [13/50], Val Losses: mse: 56.0971, mae: 5.1316, huber: 4.6677, swd: 7.2100, ept: 131.9519
    Epoch [13/50], Test Losses: mse: 43.9167, mae: 4.4496, huber: 3.9887, swd: 7.5671, ept: 140.9445
      Epoch 13 composite train-obj: 34.427878
    Epoch [13/50], Test Losses: mse: 44.8054, mae: 4.5907, huber: 4.1271, swd: 8.3159, ept: 130.0997
    Best round's Test MSE: 44.8054, MAE: 4.5907, SWD: 8.3159
    Best round's Validation MSE: 53.4482, MAE: 5.1168, SWD: 8.6503
    Best round's Test verification MSE : 44.8054, MAE: 4.5907, SWD: 8.3159
    Time taken: 34.11 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 69.3768, mae: 6.3836, huber: 5.9023, swd: 12.9363, ept: 34.4096
    Epoch [1/50], Val Losses: mse: 63.0211, mae: 6.0084, huber: 5.5305, swd: 15.7690, ept: 57.1510
    Epoch [1/50], Test Losses: mse: 56.9207, mae: 5.6506, huber: 5.1738, swd: 17.0701, ept: 60.8876
      Epoch 1 composite train-obj: 69.376797
            Val objective improved inf → 63.0211, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 56.0359, mae: 5.6416, huber: 5.1640, swd: 12.3954, ept: 41.8681
    Epoch [2/50], Val Losses: mse: 56.7624, mae: 5.5894, huber: 5.1142, swd: 13.9903, ept: 84.7182
    Epoch [2/50], Test Losses: mse: 51.5916, mae: 5.3216, huber: 4.8467, swd: 15.1601, ept: 86.6626
      Epoch 2 composite train-obj: 56.035856
            Val objective improved 63.0211 → 56.7624, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 50.5232, mae: 5.2749, huber: 4.7998, swd: 11.1797, ept: 45.3766
    Epoch [3/50], Val Losses: mse: 56.0519, mae: 5.4647, huber: 4.9911, swd: 12.0662, ept: 99.1986
    Epoch [3/50], Test Losses: mse: 49.2054, mae: 5.0620, huber: 4.5905, swd: 12.4640, ept: 100.1055
      Epoch 3 composite train-obj: 50.523215
            Val objective improved 56.7624 → 56.0519, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 47.3512, mae: 5.0483, huber: 4.5753, swd: 10.0805, ept: 48.0066
    Epoch [4/50], Val Losses: mse: 54.4036, mae: 5.3566, huber: 4.8838, swd: 11.3856, ept: 103.1675
    Epoch [4/50], Test Losses: mse: 46.4391, mae: 4.8249, huber: 4.3561, swd: 11.1219, ept: 114.9986
      Epoch 4 composite train-obj: 47.351156
            Val objective improved 56.0519 → 54.4036, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 45.0100, mae: 4.8836, huber: 4.4119, swd: 9.2809, ept: 49.1795
    Epoch [5/50], Val Losses: mse: 54.9625, mae: 5.2461, huber: 4.7774, swd: 10.8395, ept: 106.5462
    Epoch [5/50], Test Losses: mse: 46.1761, mae: 4.7643, huber: 4.2968, swd: 10.4318, ept: 121.6862
      Epoch 5 composite train-obj: 45.010005
            No improvement (54.9625), counter 1/5
    Epoch [6/50], Train Losses: mse: 42.6822, mae: 4.7094, huber: 4.2396, swd: 8.5335, ept: 50.2452
    Epoch [6/50], Val Losses: mse: 54.6560, mae: 5.2818, huber: 4.8120, swd: 9.4593, ept: 102.0413
    Epoch [6/50], Test Losses: mse: 45.9926, mae: 4.7569, huber: 4.2899, swd: 8.6338, ept: 117.4680
      Epoch 6 composite train-obj: 42.682243
            No improvement (54.6560), counter 2/5
    Epoch [7/50], Train Losses: mse: 40.9500, mae: 4.5818, huber: 4.1133, swd: 8.0260, ept: 50.8878
    Epoch [7/50], Val Losses: mse: 56.6258, mae: 5.3007, huber: 4.8304, swd: 9.0683, ept: 116.4162
    Epoch [7/50], Test Losses: mse: 45.7271, mae: 4.6309, huber: 4.1651, swd: 8.2276, ept: 139.3237
      Epoch 7 composite train-obj: 40.950033
            No improvement (56.6258), counter 3/5
    Epoch [8/50], Train Losses: mse: 39.5728, mae: 4.4760, huber: 4.0087, swd: 7.5287, ept: 51.4779
    Epoch [8/50], Val Losses: mse: 55.0622, mae: 5.2242, huber: 4.7536, swd: 8.5793, ept: 116.6997
    Epoch [8/50], Test Losses: mse: 45.3344, mae: 4.6477, huber: 4.1807, swd: 8.0406, ept: 136.5351
      Epoch 8 composite train-obj: 39.572802
            No improvement (55.0622), counter 4/5
    Epoch [9/50], Train Losses: mse: 38.2537, mae: 4.3764, huber: 3.9102, swd: 7.1963, ept: 51.9744
    Epoch [9/50], Val Losses: mse: 57.0524, mae: 5.2801, huber: 4.8120, swd: 8.8600, ept: 114.1115
    Epoch [9/50], Test Losses: mse: 46.8872, mae: 4.6807, huber: 4.2160, swd: 8.2629, ept: 130.9140
      Epoch 9 composite train-obj: 38.253729
    Epoch [9/50], Test Losses: mse: 46.4391, mae: 4.8249, huber: 4.3561, swd: 11.1219, ept: 114.9986
    Best round's Test MSE: 46.4391, MAE: 4.8249, SWD: 11.1219
    Best round's Validation MSE: 54.4036, MAE: 5.3566, SWD: 11.3856
    Best round's Test verification MSE : 46.4391, MAE: 4.8249, SWD: 11.1219
    Time taken: 23.68 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred336_20250514_2133)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 46.6628 ± 1.6157
      mae: 4.7722 ± 0.1320
      huber: 4.3052 ± 0.1297
      swd: 9.7965 ± 1.1508
      ept: 121.7450 ± 6.2690
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 54.5716 ± 0.9930
      mae: 5.2602 ± 0.1034
      huber: 4.7904 ± 0.1010
      swd: 9.9130 ± 1.1265
      ept: 111.9585 ± 8.4393
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 87.28 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred336_20250514_2133
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE + 0.5 SWD


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.5, 0.0]
)
exp_patch_tst = execute_model_evaluation('lorenz', cfg_patch_tst, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 99.4185, mae: 7.6075, huber: 7.1231, swd: 3.4429, ept: 19.6252
    Epoch [1/50], Val Losses: mse: 93.5616, mae: 7.1875, huber: 6.7060, swd: 1.6077, ept: 22.1377
    Epoch [1/50], Test Losses: mse: 84.8924, mae: 6.6483, huber: 6.1693, swd: 1.4334, ept: 24.7804
      Epoch 1 composite train-obj: 8.844522
            Val objective improved inf → 7.5098, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 80.1989, mae: 6.6022, huber: 6.1219, swd: 1.4419, ept: 31.4858
    Epoch [2/50], Val Losses: mse: 82.2437, mae: 6.6356, huber: 6.1567, swd: 1.1467, ept: 37.1009
    Epoch [2/50], Test Losses: mse: 74.3677, mae: 6.0942, huber: 5.6182, swd: 1.0416, ept: 41.5554
      Epoch 2 composite train-obj: 6.842837
            Val objective improved 7.5098 → 6.7301, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 74.1876, mae: 6.2326, huber: 5.7544, swd: 1.3422, ept: 36.7182
    Epoch [3/50], Val Losses: mse: 81.2738, mae: 6.4902, huber: 6.0121, swd: 1.2408, ept: 49.4290
    Epoch [3/50], Test Losses: mse: 72.9075, mae: 5.9616, huber: 5.4857, swd: 1.1593, ept: 52.7960
      Epoch 3 composite train-obj: 6.425445
            Val objective improved 6.7301 → 6.6325, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 69.1245, mae: 5.9209, huber: 5.4445, swd: 1.2487, ept: 40.1455
    Epoch [4/50], Val Losses: mse: 76.7894, mae: 6.1850, huber: 5.7096, swd: 1.3258, ept: 67.3129
    Epoch [4/50], Test Losses: mse: 67.1937, mae: 5.6025, huber: 5.1293, swd: 1.1438, ept: 66.2767
      Epoch 4 composite train-obj: 6.068895
            Val objective improved 6.6325 → 6.3725, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 64.7891, mae: 5.6634, huber: 5.1887, swd: 1.2338, ept: 42.6660
    Epoch [5/50], Val Losses: mse: 73.8968, mae: 5.9772, huber: 5.5034, swd: 1.2817, ept: 78.5122
    Epoch [5/50], Test Losses: mse: 64.2776, mae: 5.4003, huber: 4.9294, swd: 1.1327, ept: 80.5059
      Epoch 5 composite train-obj: 5.805623
            Val objective improved 6.3725 → 6.1443, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 61.6487, mae: 5.4665, huber: 4.9935, swd: 1.1931, ept: 44.4299
    Epoch [6/50], Val Losses: mse: 72.4914, mae: 5.9070, huber: 5.4336, swd: 1.4409, ept: 81.0694
    Epoch [6/50], Test Losses: mse: 62.0378, mae: 5.2963, huber: 4.8255, swd: 1.1702, ept: 80.6142
      Epoch 6 composite train-obj: 5.590045
            No improvement (6.1541), counter 1/5
    Epoch [7/50], Train Losses: mse: 59.3264, mae: 5.3271, huber: 4.8551, swd: 1.2048, ept: 45.4716
    Epoch [7/50], Val Losses: mse: 74.2325, mae: 5.9551, huber: 5.4827, swd: 1.2365, ept: 87.8600
    Epoch [7/50], Test Losses: mse: 63.0363, mae: 5.2901, huber: 4.8207, swd: 0.9901, ept: 91.6633
      Epoch 7 composite train-obj: 5.457520
            Val objective improved 6.1443 → 6.1009, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 56.8157, mae: 5.1582, huber: 4.6880, swd: 1.1287, ept: 47.1664
    Epoch [8/50], Val Losses: mse: 70.9175, mae: 5.7503, huber: 5.2794, swd: 1.2174, ept: 98.6062
    Epoch [8/50], Test Losses: mse: 60.2441, mae: 5.1219, huber: 4.6541, swd: 1.0403, ept: 99.4031
      Epoch 8 composite train-obj: 5.252309
            Val objective improved 6.1009 → 5.8880, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 55.1597, mae: 5.0506, huber: 4.5815, swd: 1.1575, ept: 47.7792
    Epoch [9/50], Val Losses: mse: 71.0135, mae: 5.7408, huber: 5.2698, swd: 1.2970, ept: 101.3543
    Epoch [9/50], Test Losses: mse: 58.7242, mae: 5.0440, huber: 4.5763, swd: 1.0272, ept: 103.6939
      Epoch 9 composite train-obj: 5.160241
            No improvement (5.9182), counter 1/5
    Epoch [10/50], Train Losses: mse: 53.6358, mae: 4.9464, huber: 4.4784, swd: 1.1312, ept: 48.6062
    Epoch [10/50], Val Losses: mse: 70.9053, mae: 5.7197, huber: 5.2496, swd: 1.3563, ept: 105.1287
    Epoch [10/50], Test Losses: mse: 58.2527, mae: 5.0032, huber: 4.5362, swd: 1.0093, ept: 109.6444
      Epoch 10 composite train-obj: 5.043983
            No improvement (5.9277), counter 2/5
    Epoch [11/50], Train Losses: mse: 52.1253, mae: 4.8437, huber: 4.3768, swd: 1.1191, ept: 49.4143
    Epoch [11/50], Val Losses: mse: 70.4451, mae: 5.6811, huber: 5.2108, swd: 1.3024, ept: 105.8284
    Epoch [11/50], Test Losses: mse: 58.1701, mae: 4.9749, huber: 4.5077, swd: 1.0569, ept: 113.4557
      Epoch 11 composite train-obj: 4.936311
            Val objective improved 5.8880 → 5.8620, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 50.9137, mae: 4.7589, huber: 4.2930, swd: 1.1144, ept: 49.7878
    Epoch [12/50], Val Losses: mse: 70.8775, mae: 5.6775, huber: 5.2087, swd: 1.2747, ept: 108.8096
    Epoch [12/50], Test Losses: mse: 57.4998, mae: 4.8955, huber: 4.4307, swd: 1.0331, ept: 117.4781
      Epoch 12 composite train-obj: 4.850259
            Val objective improved 5.8620 → 5.8460, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 49.5689, mae: 4.6651, huber: 4.2004, swd: 1.0913, ept: 50.5392
    Epoch [13/50], Val Losses: mse: 70.0702, mae: 5.6110, huber: 5.1426, swd: 1.2515, ept: 113.0340
    Epoch [13/50], Test Losses: mse: 55.8460, mae: 4.7950, huber: 4.3310, swd: 0.9823, ept: 125.0435
      Epoch 13 composite train-obj: 4.746054
            Val objective improved 5.8460 → 5.7683, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 48.3385, mae: 4.5838, huber: 4.1202, swd: 1.0738, ept: 50.9688
    Epoch [14/50], Val Losses: mse: 69.1279, mae: 5.5383, huber: 5.0714, swd: 1.3485, ept: 114.4867
    Epoch [14/50], Test Losses: mse: 55.1543, mae: 4.7119, huber: 4.2506, swd: 0.9998, ept: 127.8170
      Epoch 14 composite train-obj: 4.657107
            Val objective improved 5.7683 → 5.7457, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 47.4188, mae: 4.5190, huber: 4.0562, swd: 1.0739, ept: 51.2138
    Epoch [15/50], Val Losses: mse: 69.2926, mae: 5.5505, huber: 5.0842, swd: 1.2915, ept: 114.1554
    Epoch [15/50], Test Losses: mse: 55.3999, mae: 4.7281, huber: 4.2662, swd: 0.9392, ept: 129.8025
      Epoch 15 composite train-obj: 4.593113
            Val objective improved 5.7457 → 5.7299, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 46.4477, mae: 4.4482, huber: 3.9862, swd: 1.0611, ept: 51.6716
    Epoch [16/50], Val Losses: mse: 69.4956, mae: 5.5479, huber: 5.0814, swd: 1.3326, ept: 116.8917
    Epoch [16/50], Test Losses: mse: 55.7371, mae: 4.7378, huber: 4.2761, swd: 1.1283, ept: 133.3326
      Epoch 16 composite train-obj: 4.516796
            No improvement (5.7477), counter 1/5
    Epoch [17/50], Train Losses: mse: 45.4452, mae: 4.3793, huber: 3.9183, swd: 1.0493, ept: 52.0248
    Epoch [17/50], Val Losses: mse: 72.8667, mae: 5.7022, huber: 5.2353, swd: 1.4438, ept: 110.8540
    Epoch [17/50], Test Losses: mse: 56.9418, mae: 4.7909, huber: 4.3289, swd: 1.1277, ept: 123.4351
      Epoch 17 composite train-obj: 4.442994
            No improvement (5.9572), counter 2/5
    Epoch [18/50], Train Losses: mse: 44.6996, mae: 4.3254, huber: 3.8653, swd: 1.0362, ept: 51.9287
    Epoch [18/50], Val Losses: mse: 69.7494, mae: 5.5697, huber: 5.1006, swd: 1.3701, ept: 119.9222
    Epoch [18/50], Test Losses: mse: 55.5145, mae: 4.7910, huber: 4.3253, swd: 1.0231, ept: 132.9436
      Epoch 18 composite train-obj: 4.383378
            No improvement (5.7856), counter 3/5
    Epoch [19/50], Train Losses: mse: 43.8073, mae: 4.2670, huber: 3.8077, swd: 1.0191, ept: 52.4182
    Epoch [19/50], Val Losses: mse: 68.9375, mae: 5.4899, huber: 5.0247, swd: 1.3805, ept: 124.2885
    Epoch [19/50], Test Losses: mse: 53.4472, mae: 4.6205, huber: 4.1601, swd: 1.1570, ept: 136.1467
      Epoch 19 composite train-obj: 4.317231
            Val objective improved 5.7299 → 5.7149, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 43.1379, mae: 4.2153, huber: 3.7568, swd: 1.0129, ept: 52.3504
    Epoch [20/50], Val Losses: mse: 68.4896, mae: 5.4772, huber: 5.0107, swd: 1.3185, ept: 123.7479
    Epoch [20/50], Test Losses: mse: 53.7221, mae: 4.6324, huber: 4.1708, swd: 0.9619, ept: 137.0253
      Epoch 20 composite train-obj: 4.263260
            Val objective improved 5.7149 → 5.6699, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 42.3753, mae: 4.1536, huber: 3.6963, swd: 0.9907, ept: 52.7317
    Epoch [21/50], Val Losses: mse: 68.7874, mae: 5.5225, huber: 5.0557, swd: 1.2636, ept: 119.9178
    Epoch [21/50], Test Losses: mse: 55.9022, mae: 4.8097, huber: 4.3455, swd: 1.1528, ept: 129.1979
      Epoch 21 composite train-obj: 4.191686
            No improvement (5.6874), counter 1/5
    Epoch [22/50], Train Losses: mse: 41.6867, mae: 4.1044, huber: 3.6480, swd: 0.9733, ept: 52.8514
    Epoch [22/50], Val Losses: mse: 69.5144, mae: 5.5689, huber: 5.0999, swd: 1.5256, ept: 123.3411
    Epoch [22/50], Test Losses: mse: 53.8809, mae: 4.7110, huber: 4.2462, swd: 1.1470, ept: 138.7145
      Epoch 22 composite train-obj: 4.134606
            No improvement (5.8627), counter 2/5
    Epoch [23/50], Train Losses: mse: 41.2237, mae: 4.0680, huber: 3.6122, swd: 0.9832, ept: 52.7803
    Epoch [23/50], Val Losses: mse: 69.1885, mae: 5.4652, huber: 5.0015, swd: 1.3627, ept: 122.0403
    Epoch [23/50], Test Losses: mse: 52.7864, mae: 4.5480, huber: 4.0889, swd: 0.9386, ept: 134.0508
      Epoch 23 composite train-obj: 4.103791
            No improvement (5.6829), counter 3/5
    Epoch [24/50], Train Losses: mse: 40.7402, mae: 4.0351, huber: 3.5799, swd: 0.9691, ept: 52.9836
    Epoch [24/50], Val Losses: mse: 71.0695, mae: 5.6703, huber: 5.1983, swd: 1.3250, ept: 116.7602
    Epoch [24/50], Test Losses: mse: 53.8876, mae: 4.7760, huber: 4.3076, swd: 1.0691, ept: 127.3194
      Epoch 24 composite train-obj: 4.064422
            No improvement (5.8608), counter 4/5
    Epoch [25/50], Train Losses: mse: 40.2163, mae: 3.9945, huber: 3.5400, swd: 0.9470, ept: 52.9969
    Epoch [25/50], Val Losses: mse: 71.9173, mae: 5.6445, huber: 5.1769, swd: 1.4300, ept: 124.9905
    Epoch [25/50], Test Losses: mse: 55.4772, mae: 4.7329, huber: 4.2698, swd: 1.0795, ept: 137.1361
      Epoch 25 composite train-obj: 4.013479
    Epoch [25/50], Test Losses: mse: 53.7221, mae: 4.6324, huber: 4.1708, swd: 0.9619, ept: 137.0253
    Best round's Test MSE: 53.7221, MAE: 4.6324, SWD: 0.9619
    Best round's Validation MSE: 68.4896, MAE: 5.4772, SWD: 1.3185
    Best round's Test verification MSE : 53.7221, MAE: 4.6324, SWD: 0.9619
    Time taken: 65.77 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 94.9488, mae: 7.3358, huber: 6.8527, swd: 3.2759, ept: 23.8758
    Epoch [1/50], Val Losses: mse: 90.8130, mae: 7.0778, huber: 6.5965, swd: 1.6211, ept: 24.7941
    Epoch [1/50], Test Losses: mse: 83.3269, mae: 6.6268, huber: 6.1477, swd: 1.5211, ept: 25.3622
      Epoch 1 composite train-obj: 8.490650
            Val objective improved inf → 7.4070, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 79.8484, mae: 6.5507, huber: 6.0708, swd: 1.4964, ept: 33.0141
    Epoch [2/50], Val Losses: mse: 83.8976, mae: 6.7096, huber: 6.2301, swd: 1.2031, ept: 36.4376
    Epoch [2/50], Test Losses: mse: 74.7704, mae: 6.1181, huber: 5.6416, swd: 1.0778, ept: 41.6710
      Epoch 2 composite train-obj: 6.818970
            Val objective improved 7.4070 → 6.8317, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 73.6343, mae: 6.2127, huber: 5.7343, swd: 1.3754, ept: 36.3498
    Epoch [3/50], Val Losses: mse: 76.9321, mae: 6.2960, huber: 5.8188, swd: 1.3614, ept: 53.4681
    Epoch [3/50], Test Losses: mse: 68.5854, mae: 5.7548, huber: 5.2803, swd: 1.0638, ept: 54.9858
      Epoch 3 composite train-obj: 6.421976
            Val objective improved 6.8317 → 6.4995, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 68.8769, mae: 5.9211, huber: 5.4443, swd: 1.3091, ept: 39.7781
    Epoch [4/50], Val Losses: mse: 75.2078, mae: 6.0962, huber: 5.6211, swd: 1.4231, ept: 66.6973
    Epoch [4/50], Test Losses: mse: 66.4408, mae: 5.5614, huber: 5.0888, swd: 1.0113, ept: 67.9326
      Epoch 4 composite train-obj: 6.098850
            Val objective improved 6.4995 → 6.3326, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 65.6631, mae: 5.7144, huber: 5.2390, swd: 1.3074, ept: 42.4004
    Epoch [5/50], Val Losses: mse: 73.4179, mae: 5.9951, huber: 5.5211, swd: 1.5212, ept: 74.4896
    Epoch [5/50], Test Losses: mse: 62.6002, mae: 5.3623, huber: 4.8912, swd: 1.2036, ept: 76.3993
      Epoch 5 composite train-obj: 5.892704
            Val objective improved 6.3326 → 6.2817, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 63.0315, mae: 5.5310, huber: 5.0571, swd: 1.2438, ept: 44.8223
    Epoch [6/50], Val Losses: mse: 73.1802, mae: 5.9272, huber: 5.4537, swd: 1.4925, ept: 84.6753
    Epoch [6/50], Test Losses: mse: 62.7244, mae: 5.3195, huber: 4.8485, swd: 1.2859, ept: 87.0444
      Epoch 6 composite train-obj: 5.679007
            Val objective improved 6.2817 → 6.2000, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 61.0844, mae: 5.3986, huber: 4.9260, swd: 1.2523, ept: 45.8290
    Epoch [7/50], Val Losses: mse: 72.4742, mae: 5.8426, huber: 5.3705, swd: 1.4073, ept: 90.5187
    Epoch [7/50], Test Losses: mse: 61.2864, mae: 5.1753, huber: 4.7063, swd: 0.9977, ept: 95.9971
      Epoch 7 composite train-obj: 5.552113
            Val objective improved 6.2000 → 6.0741, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 58.9020, mae: 5.2519, huber: 4.7808, swd: 1.1892, ept: 46.8933
    Epoch [8/50], Val Losses: mse: 70.3413, mae: 5.7200, huber: 5.2490, swd: 1.3268, ept: 97.8588
    Epoch [8/50], Test Losses: mse: 59.3967, mae: 5.0589, huber: 4.5909, swd: 1.0550, ept: 102.4116
      Epoch 8 composite train-obj: 5.375349
            Val objective improved 6.0741 → 5.9124, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 56.9736, mae: 5.1251, huber: 4.6553, swd: 1.1563, ept: 48.1594
    Epoch [9/50], Val Losses: mse: 70.7556, mae: 5.6859, huber: 5.2159, swd: 1.4113, ept: 100.6707
    Epoch [9/50], Test Losses: mse: 59.3756, mae: 4.9935, huber: 4.5273, swd: 1.0214, ept: 106.0824
      Epoch 9 composite train-obj: 5.233509
            No improvement (5.9215), counter 1/5
    Epoch [10/50], Train Losses: mse: 54.9508, mae: 4.9914, huber: 4.5231, swd: 1.1283, ept: 48.8140
    Epoch [10/50], Val Losses: mse: 70.9149, mae: 5.6626, huber: 5.1934, swd: 1.3483, ept: 105.6096
    Epoch [10/50], Test Losses: mse: 59.1811, mae: 4.9623, huber: 4.4964, swd: 0.9318, ept: 110.9543
      Epoch 10 composite train-obj: 5.087231
            Val objective improved 5.9124 → 5.8676, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 53.3961, mae: 4.8963, huber: 4.4289, swd: 1.1308, ept: 49.3320
    Epoch [11/50], Val Losses: mse: 68.8824, mae: 5.5649, huber: 5.0956, swd: 1.2931, ept: 107.5117
    Epoch [11/50], Test Losses: mse: 58.2979, mae: 4.9417, huber: 4.4758, swd: 0.9766, ept: 112.5141
      Epoch 11 composite train-obj: 4.994294
            Val objective improved 5.8676 → 5.7421, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 51.8099, mae: 4.7844, huber: 4.3184, swd: 1.0993, ept: 50.1368
    Epoch [12/50], Val Losses: mse: 69.9202, mae: 5.5925, huber: 5.1241, swd: 1.3508, ept: 110.9175
    Epoch [12/50], Test Losses: mse: 56.6443, mae: 4.8555, huber: 4.3903, swd: 0.9129, ept: 119.6889
      Epoch 12 composite train-obj: 4.868091
            No improvement (5.7995), counter 1/5
    Epoch [13/50], Train Losses: mse: 50.6443, mae: 4.7111, huber: 4.2459, swd: 1.0966, ept: 50.6479
    Epoch [13/50], Val Losses: mse: 70.7319, mae: 5.6050, huber: 5.1379, swd: 1.3778, ept: 113.7676
    Epoch [13/50], Test Losses: mse: 57.3092, mae: 4.8258, huber: 4.3625, swd: 0.9503, ept: 124.7990
      Epoch 13 composite train-obj: 4.794203
            No improvement (5.8267), counter 2/5
    Epoch [14/50], Train Losses: mse: 49.2243, mae: 4.6071, huber: 4.1433, swd: 1.0803, ept: 51.1347
    Epoch [14/50], Val Losses: mse: 67.6146, mae: 5.4803, huber: 5.0136, swd: 1.5175, ept: 119.9477
    Epoch [14/50], Test Losses: mse: 54.7521, mae: 4.7597, huber: 4.2959, swd: 1.0285, ept: 125.5649
      Epoch 14 composite train-obj: 4.683445
            No improvement (5.7724), counter 3/5
    Epoch [15/50], Train Losses: mse: 48.0639, mae: 4.5274, huber: 4.0646, swd: 1.0590, ept: 51.5137
    Epoch [15/50], Val Losses: mse: 66.9653, mae: 5.4198, huber: 4.9542, swd: 1.5810, ept: 120.4138
    Epoch [15/50], Test Losses: mse: 54.2223, mae: 4.6976, huber: 4.2355, swd: 1.0233, ept: 127.6007
      Epoch 15 composite train-obj: 4.594059
            No improvement (5.7446), counter 4/5
    Epoch [16/50], Train Losses: mse: 46.9228, mae: 4.4503, huber: 3.9886, swd: 1.0505, ept: 52.0321
    Epoch [16/50], Val Losses: mse: 67.8315, mae: 5.4739, huber: 5.0070, swd: 1.6771, ept: 118.3096
    Epoch [16/50], Test Losses: mse: 54.8673, mae: 4.6927, huber: 4.2300, swd: 1.0487, ept: 128.3860
      Epoch 16 composite train-obj: 4.513828
    Epoch [16/50], Test Losses: mse: 58.2979, mae: 4.9417, huber: 4.4758, swd: 0.9766, ept: 112.5141
    Best round's Test MSE: 58.2979, MAE: 4.9417, SWD: 0.9766
    Best round's Validation MSE: 68.8824, MAE: 5.5649, SWD: 1.2931
    Best round's Test verification MSE : 58.2979, MAE: 4.9417, SWD: 0.9766
    Time taken: 42.02 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 95.9484, mae: 7.3946, huber: 6.9112, swd: 3.3716, ept: 22.6101
    Epoch [1/50], Val Losses: mse: 90.2564, mae: 7.0182, huber: 6.5376, swd: 1.4009, ept: 25.3820
    Epoch [1/50], Test Losses: mse: 82.9340, mae: 6.5940, huber: 6.1152, swd: 1.2239, ept: 25.8719
      Epoch 1 composite train-obj: 8.596960
            Val objective improved inf → 7.2381, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 80.7943, mae: 6.6131, huber: 6.1328, swd: 1.6251, ept: 32.4372
    Epoch [2/50], Val Losses: mse: 83.6593, mae: 6.7072, huber: 6.2281, swd: 1.3947, ept: 36.4589
    Epoch [2/50], Test Losses: mse: 76.0566, mae: 6.2273, huber: 5.7505, swd: 1.2914, ept: 40.9433
      Epoch 2 composite train-obj: 6.945364
            Val objective improved 7.2381 → 6.9255, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 74.5009, mae: 6.2613, huber: 5.7829, swd: 1.4527, ept: 35.7716
    Epoch [3/50], Val Losses: mse: 79.3580, mae: 6.3881, huber: 5.9113, swd: 1.3341, ept: 52.9546
    Epoch [3/50], Test Losses: mse: 71.0394, mae: 5.8701, huber: 5.3956, swd: 1.1537, ept: 56.6517
      Epoch 3 composite train-obj: 6.509287
            Val objective improved 6.9255 → 6.5783, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 68.7598, mae: 5.9297, huber: 5.4530, swd: 1.4180, ept: 39.1108
    Epoch [4/50], Val Losses: mse: 74.3423, mae: 6.1105, huber: 5.6348, swd: 1.3546, ept: 65.5083
    Epoch [4/50], Test Losses: mse: 66.9571, mae: 5.6294, huber: 5.1557, swd: 1.1774, ept: 66.4847
      Epoch 4 composite train-obj: 6.162049
            Val objective improved 6.5783 → 6.3121, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 64.4507, mae: 5.6637, huber: 5.1889, swd: 1.3423, ept: 41.8284
    Epoch [5/50], Val Losses: mse: 74.0423, mae: 6.0302, huber: 5.5553, swd: 1.3664, ept: 72.8153
    Epoch [5/50], Test Losses: mse: 66.5489, mae: 5.5538, huber: 5.0811, swd: 1.2020, ept: 71.6392
      Epoch 5 composite train-obj: 5.860079
            Val objective improved 6.3121 → 6.2385, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 61.4436, mae: 5.4767, huber: 5.0035, swd: 1.3119, ept: 43.7154
    Epoch [6/50], Val Losses: mse: 68.0883, mae: 5.7353, huber: 5.2631, swd: 1.4948, ept: 86.6862
    Epoch [6/50], Test Losses: mse: 60.0447, mae: 5.2046, huber: 4.7345, swd: 1.1611, ept: 89.6880
      Epoch 6 composite train-obj: 5.659445
            Val objective improved 6.2385 → 6.0105, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 58.8501, mae: 5.3122, huber: 4.8404, swd: 1.3064, ept: 45.1698
    Epoch [7/50], Val Losses: mse: 69.0736, mae: 5.7540, huber: 5.2819, swd: 1.3832, ept: 89.8733
    Epoch [7/50], Test Losses: mse: 61.3695, mae: 5.2196, huber: 4.7502, swd: 1.1848, ept: 93.4554
      Epoch 7 composite train-obj: 5.493549
            Val objective improved 6.0105 → 5.9736, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 56.7370, mae: 5.1728, huber: 4.7022, swd: 1.2885, ept: 46.5746
    Epoch [8/50], Val Losses: mse: 67.0410, mae: 5.6174, huber: 5.1461, swd: 1.5857, ept: 96.2162
    Epoch [8/50], Test Losses: mse: 57.8099, mae: 5.0457, huber: 4.5770, swd: 1.2766, ept: 102.1951
      Epoch 8 composite train-obj: 5.346439
            Val objective improved 5.9736 → 5.9390, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 54.9155, mae: 5.0483, huber: 4.5790, swd: 1.2668, ept: 47.4607
    Epoch [9/50], Val Losses: mse: 69.2790, mae: 5.6962, huber: 5.2248, swd: 1.3665, ept: 101.6425
    Epoch [9/50], Test Losses: mse: 58.1055, mae: 4.9962, huber: 4.5289, swd: 1.1597, ept: 111.7577
      Epoch 9 composite train-obj: 5.212396
            Val objective improved 5.9390 → 5.9081, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 53.2015, mae: 4.9312, huber: 4.4630, swd: 1.2272, ept: 48.2672
    Epoch [10/50], Val Losses: mse: 67.6923, mae: 5.5525, huber: 5.0838, swd: 1.3969, ept: 109.0079
    Epoch [10/50], Test Losses: mse: 56.9571, mae: 4.9013, huber: 4.4363, swd: 1.0797, ept: 116.1059
      Epoch 10 composite train-obj: 5.076625
            Val objective improved 5.9081 → 5.7822, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 51.6234, mae: 4.8281, huber: 4.3611, swd: 1.2057, ept: 48.7894
    Epoch [11/50], Val Losses: mse: 67.3132, mae: 5.5025, huber: 5.0349, swd: 1.3301, ept: 113.2011
    Epoch [11/50], Test Losses: mse: 56.4725, mae: 4.8470, huber: 4.3836, swd: 1.0392, ept: 119.9887
      Epoch 11 composite train-obj: 4.963940
            Val objective improved 5.7822 → 5.6999, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 50.5497, mae: 4.7498, huber: 4.2837, swd: 1.2030, ept: 49.3813
    Epoch [12/50], Val Losses: mse: 70.0915, mae: 5.5855, huber: 5.1186, swd: 1.3756, ept: 108.6656
    Epoch [12/50], Test Losses: mse: 57.1341, mae: 4.8257, huber: 4.3632, swd: 1.0896, ept: 118.1886
      Epoch 12 composite train-obj: 4.885203
            No improvement (5.8065), counter 1/5
    Epoch [13/50], Train Losses: mse: 49.5399, mae: 4.6736, huber: 4.2086, swd: 1.1850, ept: 49.6169
    Epoch [13/50], Val Losses: mse: 67.6111, mae: 5.5036, huber: 5.0355, swd: 1.4405, ept: 115.8163
    Epoch [13/50], Test Losses: mse: 56.0523, mae: 4.8082, huber: 4.3439, swd: 1.0295, ept: 125.1082
      Epoch 13 composite train-obj: 4.801075
            No improvement (5.7558), counter 2/5
    Epoch [14/50], Train Losses: mse: 48.4012, mae: 4.5839, huber: 4.1202, swd: 1.1388, ept: 49.9910
    Epoch [14/50], Val Losses: mse: 68.8753, mae: 5.5204, huber: 5.0535, swd: 1.3884, ept: 117.4750
    Epoch [14/50], Test Losses: mse: 54.9497, mae: 4.7261, huber: 4.2633, swd: 1.0114, ept: 129.7527
      Epoch 14 composite train-obj: 4.689588
            No improvement (5.7477), counter 3/5
    Epoch [15/50], Train Losses: mse: 47.6169, mae: 4.5269, huber: 4.0639, swd: 1.1403, ept: 50.2452
    Epoch [15/50], Val Losses: mse: 68.7887, mae: 5.5332, huber: 5.0658, swd: 1.4349, ept: 117.7540
    Epoch [15/50], Test Losses: mse: 54.9557, mae: 4.7899, huber: 4.3261, swd: 1.0949, ept: 129.1619
      Epoch 15 composite train-obj: 4.634070
            No improvement (5.7833), counter 4/5
    Epoch [16/50], Train Losses: mse: 46.7527, mae: 4.4628, huber: 4.0008, swd: 1.1213, ept: 50.6041
    Epoch [16/50], Val Losses: mse: 71.7040, mae: 5.6618, huber: 5.1934, swd: 1.4810, ept: 113.7172
    Epoch [16/50], Test Losses: mse: 56.2656, mae: 4.8382, huber: 4.3741, swd: 1.1686, ept: 125.6456
      Epoch 16 composite train-obj: 4.561415
    Epoch [16/50], Test Losses: mse: 56.4725, mae: 4.8470, huber: 4.3836, swd: 1.0392, ept: 119.9887
    Best round's Test MSE: 56.4725, MAE: 4.8470, SWD: 1.0392
    Best round's Validation MSE: 67.3132, MAE: 5.5025, SWD: 1.3301
    Best round's Test verification MSE : 56.4725, MAE: 4.8470, SWD: 1.0392
    Time taken: 41.93 seconds
    
    ==================================================
    Experiment Summary (PatchTST_lorenz_seq336_pred336_20250514_2135)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 56.1641 ± 1.8807
      mae: 4.8070 ± 0.1294
      huber: 4.3434 ± 0.1277
      swd: 0.9926 ± 0.0335
      ept: 123.1760 ± 10.2573
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 68.2284 ± 0.6667
      mae: 5.5149 ± 0.0369
      huber: 5.0470 ± 0.0357
      swd: 1.3139 ± 0.0155
      ept: 114.8202 ± 6.7265
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 149.79 seconds
    
    Experiment complete: PatchTST_lorenz_seq336_pred336_20250514_2135
    Model: PatchTST
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

## DLinear

### huber  


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.2890, mae: 6.3031, huber: 5.8260, swd: 22.7382, ept: 42.5628
    Epoch [1/50], Val Losses: mse: 67.3290, mae: 6.1486, huber: 5.6718, swd: 22.3454, ept: 46.4953
    Epoch [1/50], Test Losses: mse: 63.8532, mae: 5.7553, huber: 5.2837, swd: 23.2319, ept: 49.6169
      Epoch 1 composite train-obj: 5.826010
            Val objective improved inf → 5.6718, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.8551, mae: 5.8278, huber: 5.3545, swd: 20.2152, ept: 63.0417
    Epoch [2/50], Val Losses: mse: 68.4698, mae: 6.1540, huber: 5.6790, swd: 19.4068, ept: 52.6691
    Epoch [2/50], Test Losses: mse: 65.4457, mae: 5.7970, huber: 5.3260, swd: 20.3472, ept: 54.2611
      Epoch 2 composite train-obj: 5.354531
            No improvement (5.6790), counter 1/5
    Epoch [3/50], Train Losses: mse: 62.7067, mae: 5.7928, huber: 5.3207, swd: 19.3861, ept: 67.2212
    Epoch [3/50], Val Losses: mse: 67.9943, mae: 6.1083, huber: 5.6345, swd: 20.0288, ept: 53.7823
    Epoch [3/50], Test Losses: mse: 64.6646, mae: 5.7586, huber: 5.2884, swd: 21.3924, ept: 54.2753
      Epoch 3 composite train-obj: 5.320748
            Val objective improved 5.6718 → 5.6345, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 62.5401, mae: 5.7759, huber: 5.3044, swd: 19.4429, ept: 69.2148
    Epoch [4/50], Val Losses: mse: 67.8025, mae: 6.1030, huber: 5.6293, swd: 19.6406, ept: 56.7486
    Epoch [4/50], Test Losses: mse: 65.0160, mae: 5.7667, huber: 5.2969, swd: 20.3972, ept: 56.1023
      Epoch 4 composite train-obj: 5.304408
            Val objective improved 5.6345 → 5.6293, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 62.5821, mae: 5.7768, huber: 5.3057, swd: 19.2444, ept: 69.7341
    Epoch [5/50], Val Losses: mse: 69.7788, mae: 6.1738, huber: 5.7010, swd: 18.2981, ept: 55.4413
    Epoch [5/50], Test Losses: mse: 65.8459, mae: 5.7929, huber: 5.3234, swd: 19.4367, ept: 56.3404
      Epoch 5 composite train-obj: 5.305662
            No improvement (5.7010), counter 1/5
    Epoch [6/50], Train Losses: mse: 62.5449, mae: 5.7666, huber: 5.2958, swd: 18.9690, ept: 70.4013
    Epoch [6/50], Val Losses: mse: 68.2501, mae: 6.1080, huber: 5.6348, swd: 19.1490, ept: 56.0234
    Epoch [6/50], Test Losses: mse: 65.3346, mae: 5.7932, huber: 5.3235, swd: 20.5657, ept: 56.9582
      Epoch 6 composite train-obj: 5.295794
            No improvement (5.6348), counter 2/5
    Epoch [7/50], Train Losses: mse: 62.3965, mae: 5.7587, huber: 5.2882, swd: 19.0953, ept: 71.0023
    Epoch [7/50], Val Losses: mse: 68.5332, mae: 6.1140, huber: 5.6412, swd: 18.0260, ept: 58.3882
    Epoch [7/50], Test Losses: mse: 65.5397, mae: 5.7627, huber: 5.2931, swd: 19.1864, ept: 58.1450
      Epoch 7 composite train-obj: 5.288198
            No improvement (5.6412), counter 3/5
    Epoch [8/50], Train Losses: mse: 62.4932, mae: 5.7569, huber: 5.2866, swd: 18.8296, ept: 71.2789
    Epoch [8/50], Val Losses: mse: 67.0764, mae: 6.0519, huber: 5.5784, swd: 19.5094, ept: 57.0761
    Epoch [8/50], Test Losses: mse: 64.0306, mae: 5.7110, huber: 5.2418, swd: 20.7007, ept: 58.9445
      Epoch 8 composite train-obj: 5.286617
            Val objective improved 5.6293 → 5.5784, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 62.4640, mae: 5.7576, huber: 5.2875, swd: 18.9427, ept: 71.4517
    Epoch [9/50], Val Losses: mse: 68.5156, mae: 6.1166, huber: 5.6445, swd: 18.9817, ept: 58.5907
    Epoch [9/50], Test Losses: mse: 65.0640, mae: 5.7521, huber: 5.2838, swd: 19.8443, ept: 57.4856
      Epoch 9 composite train-obj: 5.287515
            No improvement (5.6445), counter 1/5
    Epoch [10/50], Train Losses: mse: 62.3644, mae: 5.7495, huber: 5.2795, swd: 18.9173, ept: 71.2939
    Epoch [10/50], Val Losses: mse: 68.0363, mae: 6.0898, huber: 5.6176, swd: 18.7749, ept: 56.3114
    Epoch [10/50], Test Losses: mse: 64.7314, mae: 5.7492, huber: 5.2794, swd: 19.6622, ept: 56.0481
      Epoch 10 composite train-obj: 5.279469
            No improvement (5.6176), counter 2/5
    Epoch [11/50], Train Losses: mse: 62.4356, mae: 5.7533, huber: 5.2833, swd: 18.7305, ept: 71.5174
    Epoch [11/50], Val Losses: mse: 67.7816, mae: 6.0906, huber: 5.6178, swd: 19.2907, ept: 59.1342
    Epoch [11/50], Test Losses: mse: 64.2823, mae: 5.7321, huber: 5.2626, swd: 20.4120, ept: 59.7252
      Epoch 11 composite train-obj: 5.283266
            No improvement (5.6178), counter 3/5
    Epoch [12/50], Train Losses: mse: 62.3671, mae: 5.7478, huber: 5.2782, swd: 18.8050, ept: 71.6412
    Epoch [12/50], Val Losses: mse: 67.5444, mae: 6.0615, huber: 5.5897, swd: 19.1958, ept: 57.0835
    Epoch [12/50], Test Losses: mse: 64.2605, mae: 5.7124, huber: 5.2443, swd: 20.3183, ept: 57.3613
      Epoch 12 composite train-obj: 5.278157
            No improvement (5.5897), counter 4/5
    Epoch [13/50], Train Losses: mse: 62.3465, mae: 5.7462, huber: 5.2766, swd: 18.8528, ept: 71.7745
    Epoch [13/50], Val Losses: mse: 67.7341, mae: 6.0892, huber: 5.6165, swd: 19.4987, ept: 56.6604
    Epoch [13/50], Test Losses: mse: 64.0078, mae: 5.7163, huber: 5.2478, swd: 20.8908, ept: 57.3835
      Epoch 13 composite train-obj: 5.276577
    Epoch [13/50], Test Losses: mse: 64.0306, mae: 5.7110, huber: 5.2418, swd: 20.7007, ept: 58.9445
    Best round's Test MSE: 64.0306, MAE: 5.7110, SWD: 20.7007
    Best round's Validation MSE: 67.0764, MAE: 6.0519, SWD: 19.5094
    Best round's Test verification MSE : 64.0306, MAE: 5.7110, SWD: 20.7007
    Time taken: 13.81 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.9232, mae: 6.3184, huber: 5.8413, swd: 22.8791, ept: 42.4907
    Epoch [1/50], Val Losses: mse: 68.5666, mae: 6.1925, huber: 5.7163, swd: 20.7036, ept: 46.2669
    Epoch [1/50], Test Losses: mse: 64.8247, mae: 5.8189, huber: 5.3465, swd: 22.0656, ept: 49.2264
      Epoch 1 composite train-obj: 5.841256
            Val objective improved inf → 5.7163, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.8869, mae: 5.8297, huber: 5.3563, swd: 20.1435, ept: 63.3874
    Epoch [2/50], Val Losses: mse: 68.1672, mae: 6.1334, huber: 5.6584, swd: 20.2696, ept: 54.1481
    Epoch [2/50], Test Losses: mse: 65.3866, mae: 5.7962, huber: 5.3250, swd: 21.4428, ept: 54.1935
      Epoch 2 composite train-obj: 5.356324
            Val objective improved 5.7163 → 5.6584, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 62.7708, mae: 5.7983, huber: 5.3262, swd: 19.6298, ept: 67.3144
    Epoch [3/50], Val Losses: mse: 68.5615, mae: 6.1459, huber: 5.6714, swd: 19.3367, ept: 55.8942
    Epoch [3/50], Test Losses: mse: 64.6704, mae: 5.7555, huber: 5.2860, swd: 20.8952, ept: 55.8576
      Epoch 3 composite train-obj: 5.326154
            No improvement (5.6714), counter 1/5
    Epoch [4/50], Train Losses: mse: 62.6176, mae: 5.7831, huber: 5.3114, swd: 19.4092, ept: 68.7429
    Epoch [4/50], Val Losses: mse: 67.7168, mae: 6.1022, huber: 5.6284, swd: 20.8040, ept: 54.4342
    Epoch [4/50], Test Losses: mse: 63.9687, mae: 5.7271, huber: 5.2576, swd: 22.1154, ept: 55.4324
      Epoch 4 composite train-obj: 5.311356
            Val objective improved 5.6584 → 5.6284, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 62.5841, mae: 5.7751, huber: 5.3041, swd: 19.3403, ept: 69.7345
    Epoch [5/50], Val Losses: mse: 67.6812, mae: 6.1035, huber: 5.6305, swd: 20.1985, ept: 56.3589
    Epoch [5/50], Test Losses: mse: 64.5980, mae: 5.7794, huber: 5.3089, swd: 21.9119, ept: 57.0703
      Epoch 5 composite train-obj: 5.304067
            No improvement (5.6305), counter 1/5
    Epoch [6/50], Train Losses: mse: 62.4647, mae: 5.7647, huber: 5.2940, swd: 19.2526, ept: 70.8145
    Epoch [6/50], Val Losses: mse: 66.9212, mae: 6.0622, huber: 5.5888, swd: 20.5561, ept: 56.2629
    Epoch [6/50], Test Losses: mse: 63.9122, mae: 5.7406, huber: 5.2703, swd: 21.6712, ept: 56.3066
      Epoch 6 composite train-obj: 5.293984
            Val objective improved 5.6284 → 5.5888, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 62.5219, mae: 5.7664, huber: 5.2959, swd: 19.2236, ept: 70.6420
    Epoch [7/50], Val Losses: mse: 67.9889, mae: 6.0925, huber: 5.6200, swd: 18.8608, ept: 57.3773
    Epoch [7/50], Test Losses: mse: 64.0592, mae: 5.6964, huber: 5.2281, swd: 20.9072, ept: 57.7268
      Epoch 7 composite train-obj: 5.295925
            No improvement (5.6200), counter 1/5
    Epoch [8/50], Train Losses: mse: 62.4751, mae: 5.7632, huber: 5.2927, swd: 19.0350, ept: 70.7137
    Epoch [8/50], Val Losses: mse: 67.3852, mae: 6.0747, huber: 5.6023, swd: 20.0705, ept: 57.2822
    Epoch [8/50], Test Losses: mse: 64.3587, mae: 5.7418, huber: 5.2726, swd: 21.4357, ept: 56.4325
      Epoch 8 composite train-obj: 5.292684
            No improvement (5.6023), counter 2/5
    Epoch [9/50], Train Losses: mse: 62.3698, mae: 5.7538, huber: 5.2836, swd: 19.1206, ept: 71.0771
    Epoch [9/50], Val Losses: mse: 67.8470, mae: 6.0961, huber: 5.6230, swd: 19.1479, ept: 57.8247
    Epoch [9/50], Test Losses: mse: 64.3789, mae: 5.7316, huber: 5.2628, swd: 21.0384, ept: 57.1600
      Epoch 9 composite train-obj: 5.283637
            No improvement (5.6230), counter 3/5
    Epoch [10/50], Train Losses: mse: 62.3749, mae: 5.7533, huber: 5.2832, swd: 19.0372, ept: 71.3269
    Epoch [10/50], Val Losses: mse: 69.1027, mae: 6.1619, huber: 5.6899, swd: 18.6522, ept: 57.3349
    Epoch [10/50], Test Losses: mse: 65.9949, mae: 5.8377, huber: 5.3680, swd: 20.1822, ept: 56.1361
      Epoch 10 composite train-obj: 5.283182
            No improvement (5.6899), counter 4/5
    Epoch [11/50], Train Losses: mse: 62.3140, mae: 5.7494, huber: 5.2795, swd: 19.0309, ept: 71.6620
    Epoch [11/50], Val Losses: mse: 68.6655, mae: 6.1227, huber: 5.6498, swd: 18.2732, ept: 58.6400
    Epoch [11/50], Test Losses: mse: 65.1986, mae: 5.7578, huber: 5.2887, swd: 19.8386, ept: 58.6894
      Epoch 11 composite train-obj: 5.279454
    Epoch [11/50], Test Losses: mse: 63.9122, mae: 5.7406, huber: 5.2703, swd: 21.6712, ept: 56.3066
    Best round's Test MSE: 63.9122, MAE: 5.7406, SWD: 21.6712
    Best round's Validation MSE: 66.9212, MAE: 6.0622, SWD: 20.5561
    Best round's Test verification MSE : 63.9122, MAE: 5.7406, SWD: 21.6712
    Time taken: 11.77 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.4941, mae: 6.3135, huber: 5.8364, swd: 22.8446, ept: 42.1256
    Epoch [1/50], Val Losses: mse: 67.8450, mae: 6.1480, huber: 5.6724, swd: 21.4894, ept: 46.4845
    Epoch [1/50], Test Losses: mse: 65.1083, mae: 5.8251, huber: 5.3536, swd: 22.6894, ept: 49.7783
      Epoch 1 composite train-obj: 5.836392
            Val objective improved inf → 5.6724, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 62.9087, mae: 5.8269, huber: 5.3537, swd: 20.0695, ept: 63.9333
    Epoch [2/50], Val Losses: mse: 68.3130, mae: 6.1480, huber: 5.6733, swd: 20.2385, ept: 53.4379
    Epoch [2/50], Test Losses: mse: 64.2142, mae: 5.7377, huber: 5.2676, swd: 21.7069, ept: 55.2905
      Epoch 2 composite train-obj: 5.353718
            No improvement (5.6733), counter 1/5
    Epoch [3/50], Train Losses: mse: 62.7097, mae: 5.7956, huber: 5.3236, swd: 19.7102, ept: 67.7217
    Epoch [3/50], Val Losses: mse: 68.0359, mae: 6.0975, huber: 5.6233, swd: 19.1034, ept: 55.5866
    Epoch [3/50], Test Losses: mse: 65.3652, mae: 5.7757, huber: 5.3057, swd: 20.3220, ept: 55.5732
      Epoch 3 composite train-obj: 5.323649
            Val objective improved 5.6724 → 5.6233, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 62.5678, mae: 5.7806, huber: 5.3089, swd: 19.4245, ept: 69.0358
    Epoch [4/50], Val Losses: mse: 68.0526, mae: 6.1205, huber: 5.6471, swd: 19.6861, ept: 57.2512
    Epoch [4/50], Test Losses: mse: 65.0097, mae: 5.7819, huber: 5.3117, swd: 20.9800, ept: 57.5959
      Epoch 4 composite train-obj: 5.308944
            No improvement (5.6471), counter 1/5
    Epoch [5/50], Train Losses: mse: 62.6636, mae: 5.7745, huber: 5.3035, swd: 19.1875, ept: 69.9853
    Epoch [5/50], Val Losses: mse: 67.6649, mae: 6.0953, huber: 5.6217, swd: 19.6676, ept: 56.5898
    Epoch [5/50], Test Losses: mse: 64.9424, mae: 5.7598, huber: 5.2903, swd: 20.6722, ept: 56.7973
      Epoch 5 composite train-obj: 5.303549
            Val objective improved 5.6233 → 5.6217, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 62.4254, mae: 5.7638, huber: 5.2929, swd: 19.2378, ept: 70.6889
    Epoch [6/50], Val Losses: mse: 68.1567, mae: 6.1213, huber: 5.6483, swd: 19.4219, ept: 56.9223
    Epoch [6/50], Test Losses: mse: 64.9734, mae: 5.7736, huber: 5.3037, swd: 20.5851, ept: 57.1439
      Epoch 6 composite train-obj: 5.292927
            No improvement (5.6483), counter 1/5
    Epoch [7/50], Train Losses: mse: 62.5666, mae: 5.7689, huber: 5.2982, swd: 19.1009, ept: 70.6656
    Epoch [7/50], Val Losses: mse: 68.1660, mae: 6.1082, huber: 5.6355, swd: 19.3623, ept: 56.2367
    Epoch [7/50], Test Losses: mse: 64.3287, mae: 5.7155, huber: 5.2472, swd: 20.4309, ept: 57.3304
      Epoch 7 composite train-obj: 5.298248
            No improvement (5.6355), counter 2/5
    Epoch [8/50], Train Losses: mse: 62.4203, mae: 5.7545, huber: 5.2843, swd: 19.0575, ept: 71.2135
    Epoch [8/50], Val Losses: mse: 68.0928, mae: 6.0958, huber: 5.6232, swd: 18.7616, ept: 57.4132
    Epoch [8/50], Test Losses: mse: 64.7528, mae: 5.7414, huber: 5.2728, swd: 20.1558, ept: 56.8599
      Epoch 8 composite train-obj: 5.284333
            No improvement (5.6232), counter 3/5
    Epoch [9/50], Train Losses: mse: 62.4433, mae: 5.7559, huber: 5.2857, swd: 18.9060, ept: 71.4644
    Epoch [9/50], Val Losses: mse: 67.4753, mae: 6.0531, huber: 5.5812, swd: 19.3366, ept: 57.6448
    Epoch [9/50], Test Losses: mse: 65.1055, mae: 5.7461, huber: 5.2780, swd: 20.6689, ept: 57.9452
      Epoch 9 composite train-obj: 5.285651
            Val objective improved 5.6217 → 5.5812, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 62.3681, mae: 5.7470, huber: 5.2771, swd: 18.9511, ept: 71.7788
    Epoch [10/50], Val Losses: mse: 67.4055, mae: 6.0629, huber: 5.5907, swd: 19.5797, ept: 57.9316
    Epoch [10/50], Test Losses: mse: 64.2096, mae: 5.7162, huber: 5.2476, swd: 20.7772, ept: 57.8891
      Epoch 10 composite train-obj: 5.277055
            No improvement (5.5907), counter 1/5
    Epoch [11/50], Train Losses: mse: 62.4501, mae: 5.7560, huber: 5.2859, swd: 18.9103, ept: 71.1718
    Epoch [11/50], Val Losses: mse: 67.5214, mae: 6.0727, huber: 5.5998, swd: 19.4016, ept: 57.5618
    Epoch [11/50], Test Losses: mse: 64.7399, mae: 5.7348, huber: 5.2663, swd: 20.8299, ept: 58.2448
      Epoch 11 composite train-obj: 5.285935
            No improvement (5.5998), counter 2/5
    Epoch [12/50], Train Losses: mse: 62.4236, mae: 5.7516, huber: 5.2817, swd: 18.8001, ept: 71.6296
    Epoch [12/50], Val Losses: mse: 67.3253, mae: 6.0710, huber: 5.5980, swd: 19.6583, ept: 56.6918
    Epoch [12/50], Test Losses: mse: 63.4354, mae: 5.6767, huber: 5.2093, swd: 21.1976, ept: 56.2839
      Epoch 12 composite train-obj: 5.281694
            No improvement (5.5980), counter 3/5
    Epoch [13/50], Train Losses: mse: 62.3306, mae: 5.7468, huber: 5.2771, swd: 18.8959, ept: 71.4861
    Epoch [13/50], Val Losses: mse: 68.2081, mae: 6.0979, huber: 5.6259, swd: 18.8557, ept: 58.1591
    Epoch [13/50], Test Losses: mse: 64.9518, mae: 5.7496, huber: 5.2812, swd: 20.1792, ept: 58.8545
      Epoch 13 composite train-obj: 5.277145
            No improvement (5.6259), counter 4/5
    Epoch [14/50], Train Losses: mse: 62.2760, mae: 5.7419, huber: 5.2724, swd: 18.8312, ept: 71.8628
    Epoch [14/50], Val Losses: mse: 68.4382, mae: 6.1043, huber: 5.6327, swd: 18.5640, ept: 57.5365
    Epoch [14/50], Test Losses: mse: 64.9606, mae: 5.7317, huber: 5.2636, swd: 19.8123, ept: 58.8883
      Epoch 14 composite train-obj: 5.272396
    Epoch [14/50], Test Losses: mse: 65.1055, mae: 5.7461, huber: 5.2780, swd: 20.6689, ept: 57.9452
    Best round's Test MSE: 65.1055, MAE: 5.7461, SWD: 20.6689
    Best round's Validation MSE: 67.4753, MAE: 6.0531, SWD: 19.3366
    Best round's Test verification MSE : 65.1055, MAE: 5.7461, SWD: 20.6689
    Time taken: 15.17 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred336_20250514_2137)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 64.3494 ± 0.5368
      mae: 5.7326 ± 0.0154
      huber: 5.2633 ± 0.0156
      swd: 21.0136 ± 0.4652
      ept: 57.7321 ± 1.0874
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 67.1576 ± 0.2334
      mae: 6.0557 ± 0.0046
      huber: 5.5828 ± 0.0044
      swd: 19.8007 ± 0.5388
      ept: 56.9946 ± 0.5671
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 40.77 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred336_20250514_2137
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### huber + 0.5 SWD


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.5, 0.0],
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 139.4832, mae: 8.2891, huber: 7.8061, swd: 2.5330, ept: 9.9655
    Epoch [1/50], Val Losses: mse: 85.6057, mae: 6.8084, huber: 6.3296, swd: 1.1561, ept: 21.2067
    Epoch [1/50], Test Losses: mse: 82.5137, mae: 6.4089, huber: 5.9337, swd: 0.9944, ept: 23.6570
      Epoch 1 composite train-obj: 9.072575
            Val objective improved inf → 6.9077, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 79.6146, mae: 6.4082, huber: 5.9325, swd: 1.2885, ept: 33.8860
    Epoch [2/50], Val Losses: mse: 86.3391, mae: 6.6853, huber: 6.2089, swd: 0.9872, ept: 32.6132
    Epoch [2/50], Test Losses: mse: 84.0607, mae: 6.2755, huber: 5.8032, swd: 0.8525, ept: 36.2112
      Epoch 2 composite train-obj: 6.576736
            Val objective improved 6.9077 → 6.7024, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 79.3506, mae: 6.2938, huber: 5.8203, swd: 1.2081, ept: 39.4632
    Epoch [3/50], Val Losses: mse: 86.6472, mae: 6.6374, huber: 6.1622, swd: 0.9774, ept: 34.4509
    Epoch [3/50], Test Losses: mse: 84.1852, mae: 6.2238, huber: 5.7529, swd: 0.8700, ept: 37.5757
      Epoch 3 composite train-obj: 6.424330
            Val objective improved 6.7024 → 6.6509, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 79.4507, mae: 6.2504, huber: 5.7778, swd: 1.1754, ept: 41.2115
    Epoch [4/50], Val Losses: mse: 86.5890, mae: 6.5984, huber: 6.1243, swd: 1.0359, ept: 34.3452
    Epoch [4/50], Test Losses: mse: 84.4611, mae: 6.1942, huber: 5.7243, swd: 0.8749, ept: 36.4928
      Epoch 4 composite train-obj: 6.365490
            Val objective improved 6.6509 → 6.6423, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 79.7210, mae: 6.2253, huber: 5.7533, swd: 1.1195, ept: 42.0815
    Epoch [5/50], Val Losses: mse: 88.5967, mae: 6.6407, huber: 6.1666, swd: 0.9515, ept: 34.2672
    Epoch [5/50], Test Losses: mse: 86.1750, mae: 6.2319, huber: 5.7618, swd: 0.8174, ept: 36.2782
      Epoch 5 composite train-obj: 6.313104
            No improvement (6.6423), counter 1/5
    Epoch [6/50], Train Losses: mse: 80.3375, mae: 6.2145, huber: 5.7429, swd: 1.1084, ept: 43.0586
    Epoch [6/50], Val Losses: mse: 89.1651, mae: 6.6292, huber: 6.1551, swd: 0.9288, ept: 34.0748
    Epoch [6/50], Test Losses: mse: 86.7735, mae: 6.2376, huber: 5.7675, swd: 0.7923, ept: 35.1749
      Epoch 6 composite train-obj: 6.297054
            Val objective improved 6.6423 → 6.6195, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 81.0157, mae: 6.2056, huber: 5.7342, swd: 1.0494, ept: 43.2818
    Epoch [7/50], Val Losses: mse: 89.5865, mae: 6.6169, huber: 6.1429, swd: 0.9787, ept: 34.3020
    Epoch [7/50], Test Losses: mse: 87.2218, mae: 6.2282, huber: 5.7584, swd: 0.8164, ept: 34.9681
      Epoch 7 composite train-obj: 6.258914
            No improvement (6.6323), counter 1/5
    Epoch [8/50], Train Losses: mse: 81.8798, mae: 6.1966, huber: 5.7254, swd: 1.0244, ept: 44.2517
    Epoch [8/50], Val Losses: mse: 89.8692, mae: 6.5875, huber: 6.1137, swd: 0.9327, ept: 35.0289
    Epoch [8/50], Test Losses: mse: 87.3475, mae: 6.2072, huber: 5.7375, swd: 0.8058, ept: 35.6755
      Epoch 8 composite train-obj: 6.237599
            Val objective improved 6.6195 → 6.5800, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 82.6731, mae: 6.1884, huber: 5.7176, swd: 0.9947, ept: 44.5980
    Epoch [9/50], Val Losses: mse: 91.0804, mae: 6.6052, huber: 6.1316, swd: 0.9955, ept: 36.0588
    Epoch [9/50], Test Losses: mse: 88.4213, mae: 6.2265, huber: 5.7570, swd: 0.8459, ept: 36.1029
      Epoch 9 composite train-obj: 6.214920
            No improvement (6.6294), counter 1/5
    Epoch [10/50], Train Losses: mse: 83.4820, mae: 6.1829, huber: 5.7121, swd: 1.0060, ept: 45.0070
    Epoch [10/50], Val Losses: mse: 91.5555, mae: 6.5874, huber: 6.1139, swd: 0.9553, ept: 36.0220
    Epoch [10/50], Test Losses: mse: 88.8828, mae: 6.2137, huber: 5.7440, swd: 0.8241, ept: 35.3641
      Epoch 10 composite train-obj: 6.215086
            No improvement (6.5915), counter 2/5
    Epoch [11/50], Train Losses: mse: 84.1571, mae: 6.1786, huber: 5.7079, swd: 0.9724, ept: 45.5296
    Epoch [11/50], Val Losses: mse: 92.0170, mae: 6.5780, huber: 6.1047, swd: 0.9460, ept: 36.6981
    Epoch [11/50], Test Losses: mse: 89.3220, mae: 6.2027, huber: 5.7338, swd: 0.8204, ept: 36.2758
      Epoch 11 composite train-obj: 6.194047
            Val objective improved 6.5800 → 6.5777, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 84.7688, mae: 6.1705, huber: 5.7000, swd: 0.9703, ept: 45.4685
    Epoch [12/50], Val Losses: mse: 92.5552, mae: 6.5726, huber: 6.0990, swd: 0.9564, ept: 36.5669
    Epoch [12/50], Test Losses: mse: 89.9046, mae: 6.1978, huber: 5.7286, swd: 0.8229, ept: 35.9938
      Epoch 12 composite train-obj: 6.185136
            Val objective improved 6.5777 → 6.5772, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 85.3079, mae: 6.1657, huber: 5.6952, swd: 0.9585, ept: 45.6245
    Epoch [13/50], Val Losses: mse: 93.1002, mae: 6.5672, huber: 6.0939, swd: 0.9914, ept: 36.3205
    Epoch [13/50], Test Losses: mse: 90.3300, mae: 6.1859, huber: 5.7165, swd: 0.8377, ept: 35.9853
      Epoch 13 composite train-obj: 6.174466
            No improvement (6.5896), counter 1/5
    Epoch [14/50], Train Losses: mse: 85.8576, mae: 6.1609, huber: 5.6905, swd: 0.9476, ept: 45.6540
    Epoch [14/50], Val Losses: mse: 93.6369, mae: 6.5518, huber: 6.0794, swd: 0.9772, ept: 36.8220
    Epoch [14/50], Test Losses: mse: 91.4988, mae: 6.2213, huber: 5.7517, swd: 0.8467, ept: 36.0953
      Epoch 14 composite train-obj: 6.164314
            Val objective improved 6.5772 → 6.5680, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 86.2139, mae: 6.1556, huber: 5.6853, swd: 0.9502, ept: 46.0286
    Epoch [15/50], Val Losses: mse: 94.2270, mae: 6.5619, huber: 6.0883, swd: 0.9395, ept: 36.7438
    Epoch [15/50], Test Losses: mse: 91.9626, mae: 6.2043, huber: 5.7350, swd: 0.7947, ept: 35.3458
      Epoch 15 composite train-obj: 6.160370
            Val objective improved 6.5680 → 6.5580, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 86.6054, mae: 6.1530, huber: 5.6828, swd: 0.9429, ept: 45.9372
    Epoch [16/50], Val Losses: mse: 94.6215, mae: 6.5641, huber: 6.0912, swd: 1.0328, ept: 36.7737
    Epoch [16/50], Test Losses: mse: 91.9490, mae: 6.1901, huber: 5.7214, swd: 0.8842, ept: 35.6785
      Epoch 16 composite train-obj: 6.154242
            No improvement (6.6076), counter 1/5
    Epoch [17/50], Train Losses: mse: 86.8581, mae: 6.1528, huber: 5.6825, swd: 0.9690, ept: 45.4940
    Epoch [17/50], Val Losses: mse: 95.4929, mae: 6.5863, huber: 6.1130, swd: 1.0124, ept: 36.9500
    Epoch [17/50], Test Losses: mse: 92.4868, mae: 6.2018, huber: 5.7330, swd: 0.8800, ept: 35.7508
      Epoch 17 composite train-obj: 6.166952
            No improvement (6.6192), counter 2/5
    Epoch [18/50], Train Losses: mse: 87.0400, mae: 6.1493, huber: 5.6792, swd: 0.9476, ept: 45.8537
    Epoch [18/50], Val Losses: mse: 95.5820, mae: 6.5772, huber: 6.1047, swd: 0.9468, ept: 36.8484
    Epoch [18/50], Test Losses: mse: 92.7724, mae: 6.2008, huber: 5.7321, swd: 0.8081, ept: 35.6411
      Epoch 18 composite train-obj: 6.152970
            No improvement (6.5781), counter 3/5
    Epoch [19/50], Train Losses: mse: 87.1371, mae: 6.1484, huber: 5.6783, swd: 0.9561, ept: 45.6591
    Epoch [19/50], Val Losses: mse: 95.1322, mae: 6.5594, huber: 6.0868, swd: 0.9538, ept: 36.1528
    Epoch [19/50], Test Losses: mse: 92.5396, mae: 6.1900, huber: 5.7218, swd: 0.8227, ept: 35.2354
      Epoch 19 composite train-obj: 6.156316
            No improvement (6.5637), counter 4/5
    Epoch [20/50], Train Losses: mse: 87.2671, mae: 6.1423, huber: 5.6723, swd: 0.9451, ept: 45.7798
    Epoch [20/50], Val Losses: mse: 95.5366, mae: 6.5641, huber: 6.0915, swd: 0.9688, ept: 36.6397
    Epoch [20/50], Test Losses: mse: 92.4869, mae: 6.1853, huber: 5.7166, swd: 0.8157, ept: 35.2065
      Epoch 20 composite train-obj: 6.144853
    Epoch [20/50], Test Losses: mse: 91.9626, mae: 6.2043, huber: 5.7350, swd: 0.7947, ept: 35.3458
    Best round's Test MSE: 91.9626, MAE: 6.2043, SWD: 0.7947
    Best round's Validation MSE: 94.2270, MAE: 6.5619, SWD: 0.9395
    Best round's Test verification MSE : 91.9626, MAE: 6.2043, SWD: 0.7947
    Time taken: 21.82 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 141.9100, mae: 8.3109, huber: 7.8279, swd: 2.6107, ept: 11.9015
    Epoch [1/50], Val Losses: mse: 86.5140, mae: 6.8442, huber: 6.3650, swd: 1.1016, ept: 22.2707
    Epoch [1/50], Test Losses: mse: 82.8797, mae: 6.4356, huber: 5.9602, swd: 0.9742, ept: 25.5080
      Epoch 1 composite train-obj: 9.133220
            Val objective improved inf → 6.9158, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 79.8634, mae: 6.4199, huber: 5.9443, swd: 1.3400, ept: 33.2493
    Epoch [2/50], Val Losses: mse: 86.9332, mae: 6.6879, huber: 6.2112, swd: 1.0886, ept: 30.2623
    Epoch [2/50], Test Losses: mse: 84.5348, mae: 6.2801, huber: 5.8082, swd: 0.9385, ept: 33.5417
      Epoch 2 composite train-obj: 6.614286
            Val objective improved 6.9158 → 6.7555, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 80.1261, mae: 6.3056, huber: 5.8320, swd: 1.2932, ept: 37.6484
    Epoch [3/50], Val Losses: mse: 89.1603, mae: 6.6875, huber: 6.2118, swd: 1.0407, ept: 30.6552
    Epoch [3/50], Test Losses: mse: 86.7392, mae: 6.2812, huber: 5.8097, swd: 0.8928, ept: 34.0991
      Epoch 3 composite train-obj: 6.478581
            Val objective improved 6.7555 → 6.7322, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 80.6016, mae: 6.2641, huber: 5.7913, swd: 1.2391, ept: 38.8891
    Epoch [4/50], Val Losses: mse: 89.3616, mae: 6.6465, huber: 6.1719, swd: 0.9993, ept: 32.0623
    Epoch [4/50], Test Losses: mse: 87.1014, mae: 6.2569, huber: 5.7864, swd: 0.8448, ept: 33.8960
      Epoch 4 composite train-obj: 6.410807
            Val objective improved 6.7322 → 6.6715, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 80.9668, mae: 6.2443, huber: 5.7720, swd: 1.1900, ept: 39.6234
    Epoch [5/50], Val Losses: mse: 89.0422, mae: 6.6117, huber: 6.1376, swd: 1.0690, ept: 32.1690
    Epoch [5/50], Test Losses: mse: 86.8651, mae: 6.2462, huber: 5.7757, swd: 0.9252, ept: 33.2000
      Epoch 5 composite train-obj: 6.367045
            No improvement (6.6721), counter 1/5
    Epoch [6/50], Train Losses: mse: 81.2317, mae: 6.2331, huber: 5.7612, swd: 1.1619, ept: 39.6086
    Epoch [6/50], Val Losses: mse: 89.5354, mae: 6.6177, huber: 6.1436, swd: 1.0277, ept: 31.6421
    Epoch [6/50], Test Losses: mse: 87.3737, mae: 6.2456, huber: 5.7751, swd: 0.8849, ept: 32.5557
      Epoch 6 composite train-obj: 6.342208
            Val objective improved 6.6715 → 6.6574, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 81.4685, mae: 6.2269, huber: 5.7553, swd: 1.1120, ept: 39.9347
    Epoch [7/50], Val Losses: mse: 90.1575, mae: 6.6342, huber: 6.1605, swd: 1.0402, ept: 31.9786
    Epoch [7/50], Test Losses: mse: 87.5371, mae: 6.2398, huber: 5.7698, swd: 0.8912, ept: 32.8313
      Epoch 7 composite train-obj: 6.311284
            No improvement (6.6806), counter 1/5
    Epoch [8/50], Train Losses: mse: 81.7237, mae: 6.2233, huber: 5.7518, swd: 1.1040, ept: 40.0444
    Epoch [8/50], Val Losses: mse: 90.2669, mae: 6.6235, huber: 6.1496, swd: 1.0069, ept: 32.2959
    Epoch [8/50], Test Losses: mse: 87.8777, mae: 6.2561, huber: 5.7861, swd: 0.8648, ept: 32.1424
      Epoch 8 composite train-obj: 6.303808
            Val objective improved 6.6574 → 6.6531, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 81.9649, mae: 6.2128, huber: 5.7416, swd: 1.0700, ept: 40.5456
    Epoch [9/50], Val Losses: mse: 90.6728, mae: 6.6236, huber: 6.1496, swd: 1.0529, ept: 32.8749
    Epoch [9/50], Test Losses: mse: 87.9998, mae: 6.2467, huber: 5.7766, swd: 0.9031, ept: 33.0591
      Epoch 9 composite train-obj: 6.276581
            No improvement (6.6761), counter 1/5
    Epoch [10/50], Train Losses: mse: 82.4635, mae: 6.2078, huber: 5.7367, swd: 1.0540, ept: 41.0899
    Epoch [10/50], Val Losses: mse: 91.9804, mae: 6.6379, huber: 6.1644, swd: 1.0155, ept: 33.3112
    Epoch [10/50], Test Losses: mse: 89.1618, mae: 6.2660, huber: 5.7959, swd: 0.8590, ept: 33.3601
      Epoch 10 composite train-obj: 6.263666
            No improvement (6.6721), counter 2/5
    Epoch [11/50], Train Losses: mse: 83.2401, mae: 6.2001, huber: 5.7289, swd: 1.0613, ept: 41.4323
    Epoch [11/50], Val Losses: mse: 92.4982, mae: 6.6242, huber: 6.1506, swd: 1.0878, ept: 34.0346
    Epoch [11/50], Test Losses: mse: 89.3352, mae: 6.2497, huber: 5.7794, swd: 0.9523, ept: 33.7130
      Epoch 11 composite train-obj: 6.259606
            No improvement (6.6945), counter 3/5
    Epoch [12/50], Train Losses: mse: 84.1767, mae: 6.1924, huber: 5.7215, swd: 1.0334, ept: 42.0025
    Epoch [12/50], Val Losses: mse: 93.1075, mae: 6.6125, huber: 6.1386, swd: 0.9960, ept: 33.7779
    Epoch [12/50], Test Losses: mse: 90.1556, mae: 6.2456, huber: 5.7757, swd: 0.8485, ept: 33.6659
      Epoch 12 composite train-obj: 6.238232
            Val objective improved 6.6531 → 6.6366, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 84.9278, mae: 6.1889, huber: 5.7180, swd: 1.0348, ept: 42.2110
    Epoch [13/50], Val Losses: mse: 94.6213, mae: 6.6421, huber: 6.1685, swd: 0.9889, ept: 34.1342
    Epoch [13/50], Test Losses: mse: 90.9220, mae: 6.2458, huber: 5.7762, swd: 0.8393, ept: 33.7856
      Epoch 13 composite train-obj: 6.235360
            No improvement (6.6630), counter 1/5
    Epoch [14/50], Train Losses: mse: 85.6163, mae: 6.1807, huber: 5.7097, swd: 1.0213, ept: 42.6842
    Epoch [14/50], Val Losses: mse: 94.1762, mae: 6.5886, huber: 6.1152, swd: 1.0126, ept: 34.5303
    Epoch [14/50], Test Losses: mse: 90.7990, mae: 6.2097, huber: 5.7402, swd: 0.8417, ept: 34.5319
      Epoch 14 composite train-obj: 6.220408
            Val objective improved 6.6366 → 6.6215, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 86.2588, mae: 6.1753, huber: 5.7044, swd: 1.0336, ept: 43.1873
    Epoch [15/50], Val Losses: mse: 95.5013, mae: 6.6086, huber: 6.1350, swd: 0.9904, ept: 34.7949
    Epoch [15/50], Test Losses: mse: 91.8723, mae: 6.2253, huber: 5.7556, swd: 0.8132, ept: 34.5229
      Epoch 15 composite train-obj: 6.221212
            No improvement (6.6302), counter 1/5
    Epoch [16/50], Train Losses: mse: 86.7927, mae: 6.1693, huber: 5.6986, swd: 1.0262, ept: 43.5228
    Epoch [16/50], Val Losses: mse: 95.7977, mae: 6.6000, huber: 6.1268, swd: 0.9602, ept: 35.2652
    Epoch [16/50], Test Losses: mse: 92.1263, mae: 6.2105, huber: 5.7413, swd: 0.8471, ept: 35.0044
      Epoch 16 composite train-obj: 6.211719
            Val objective improved 6.6215 → 6.6069, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 87.3840, mae: 6.1644, huber: 5.6937, swd: 1.0241, ept: 44.1711
    Epoch [17/50], Val Losses: mse: 95.9018, mae: 6.5713, huber: 6.0981, swd: 1.0648, ept: 35.9169
    Epoch [17/50], Test Losses: mse: 92.4049, mae: 6.1995, huber: 5.7303, swd: 0.9003, ept: 35.2261
      Epoch 17 composite train-obj: 6.205737
            No improvement (6.6305), counter 1/5
    Epoch [18/50], Train Losses: mse: 87.8196, mae: 6.1602, huber: 5.6896, swd: 1.0040, ept: 44.4118
    Epoch [18/50], Val Losses: mse: 96.1885, mae: 6.5673, huber: 6.0939, swd: 1.0452, ept: 35.4841
    Epoch [18/50], Test Losses: mse: 92.6167, mae: 6.1927, huber: 5.7232, swd: 0.8950, ept: 34.8889
      Epoch 18 composite train-obj: 6.191621
            No improvement (6.6165), counter 2/5
    Epoch [19/50], Train Losses: mse: 88.1827, mae: 6.1534, huber: 5.6828, swd: 0.9957, ept: 44.7855
    Epoch [19/50], Val Losses: mse: 96.5085, mae: 6.5704, huber: 6.0970, swd: 1.0296, ept: 35.7634
    Epoch [19/50], Test Losses: mse: 93.3807, mae: 6.2095, huber: 5.7399, swd: 0.8651, ept: 35.5002
      Epoch 19 composite train-obj: 6.180685
            No improvement (6.6119), counter 3/5
    Epoch [20/50], Train Losses: mse: 88.5256, mae: 6.1519, huber: 5.6814, swd: 1.0050, ept: 45.0283
    Epoch [20/50], Val Losses: mse: 97.1540, mae: 6.5816, huber: 6.1087, swd: 1.0141, ept: 35.8855
    Epoch [20/50], Test Losses: mse: 93.9531, mae: 6.2237, huber: 5.7541, swd: 0.8445, ept: 35.6085
      Epoch 20 composite train-obj: 6.183919
            No improvement (6.6157), counter 4/5
    Epoch [21/50], Train Losses: mse: 88.6833, mae: 6.1465, huber: 5.6761, swd: 1.0122, ept: 45.2556
    Epoch [21/50], Val Losses: mse: 97.0431, mae: 6.5661, huber: 6.0929, swd: 1.0095, ept: 35.3147
    Epoch [21/50], Test Losses: mse: 93.6988, mae: 6.1963, huber: 5.7271, swd: 0.8405, ept: 35.1398
      Epoch 21 composite train-obj: 6.182173
            Val objective improved 6.6069 → 6.5976, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 88.8975, mae: 6.1438, huber: 5.6735, swd: 1.0069, ept: 45.4921
    Epoch [22/50], Val Losses: mse: 97.6939, mae: 6.5708, huber: 6.0977, swd: 1.1656, ept: 36.8422
    Epoch [22/50], Test Losses: mse: 94.3098, mae: 6.2045, huber: 5.7354, swd: 0.9707, ept: 35.8607
      Epoch 22 composite train-obj: 6.176889
            No improvement (6.6805), counter 1/5
    Epoch [23/50], Train Losses: mse: 89.1931, mae: 6.1441, huber: 5.6737, swd: 0.9897, ept: 45.6745
    Epoch [23/50], Val Losses: mse: 98.0054, mae: 6.5836, huber: 6.1103, swd: 0.9935, ept: 36.4730
    Epoch [23/50], Test Losses: mse: 94.1415, mae: 6.2000, huber: 5.7301, swd: 0.8651, ept: 36.3166
      Epoch 23 composite train-obj: 6.168601
            No improvement (6.6071), counter 2/5
    Epoch [24/50], Train Losses: mse: 89.3280, mae: 6.1429, huber: 5.6724, swd: 0.9961, ept: 45.7300
    Epoch [24/50], Val Losses: mse: 98.0107, mae: 6.5610, huber: 6.0885, swd: 1.0114, ept: 36.8809
    Epoch [24/50], Test Losses: mse: 94.2297, mae: 6.1889, huber: 5.7200, swd: 0.8677, ept: 36.5522
      Epoch 24 composite train-obj: 6.170508
            Val objective improved 6.5976 → 6.5942, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 89.3750, mae: 6.1375, huber: 5.6673, swd: 0.9826, ept: 45.8835
    Epoch [25/50], Val Losses: mse: 98.0884, mae: 6.5746, huber: 6.1018, swd: 0.9839, ept: 36.7434
    Epoch [25/50], Test Losses: mse: 94.5584, mae: 6.1987, huber: 5.7294, swd: 0.8448, ept: 36.1977
      Epoch 25 composite train-obj: 6.158598
            Val objective improved 6.5942 → 6.5937, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 89.5842, mae: 6.1379, huber: 5.6677, swd: 0.9883, ept: 46.0500
    Epoch [26/50], Val Losses: mse: 97.7433, mae: 6.5537, huber: 6.0809, swd: 1.0109, ept: 37.2327
    Epoch [26/50], Test Losses: mse: 94.0575, mae: 6.1773, huber: 5.7084, swd: 0.8582, ept: 36.5756
      Epoch 26 composite train-obj: 6.161821
            Val objective improved 6.5937 → 6.5863, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 89.7592, mae: 6.1380, huber: 5.6676, swd: 1.0181, ept: 46.0943
    Epoch [27/50], Val Losses: mse: 98.2369, mae: 6.5652, huber: 6.0917, swd: 1.0208, ept: 37.6498
    Epoch [27/50], Test Losses: mse: 94.8791, mae: 6.1999, huber: 5.7305, swd: 0.8479, ept: 36.8616
      Epoch 27 composite train-obj: 6.176663
            No improvement (6.6021), counter 1/5
    Epoch [28/50], Train Losses: mse: 89.8182, mae: 6.1343, huber: 5.6640, swd: 0.9872, ept: 46.4244
    Epoch [28/50], Val Losses: mse: 98.3078, mae: 6.5560, huber: 6.0831, swd: 1.0366, ept: 37.3656
    Epoch [28/50], Test Losses: mse: 94.5642, mae: 6.1812, huber: 5.7123, swd: 0.9060, ept: 37.0728
      Epoch 28 composite train-obj: 6.157649
            No improvement (6.6014), counter 2/5
    Epoch [29/50], Train Losses: mse: 89.9734, mae: 6.1340, huber: 5.6636, swd: 1.0150, ept: 46.4691
    Epoch [29/50], Val Losses: mse: 98.5559, mae: 6.5690, huber: 6.0959, swd: 0.9830, ept: 38.1816
    Epoch [29/50], Test Losses: mse: 94.5235, mae: 6.1702, huber: 5.7010, swd: 0.8355, ept: 37.7714
      Epoch 29 composite train-obj: 6.171124
            No improvement (6.5873), counter 3/5
    Epoch [30/50], Train Losses: mse: 90.0372, mae: 6.1290, huber: 5.6590, swd: 0.9794, ept: 46.6319
    Epoch [30/50], Val Losses: mse: 98.5515, mae: 6.5425, huber: 6.0698, swd: 0.9697, ept: 38.2772
    Epoch [30/50], Test Losses: mse: 95.3715, mae: 6.1980, huber: 5.7287, swd: 0.8257, ept: 37.3714
      Epoch 30 composite train-obj: 6.148748
            Val objective improved 6.5863 → 6.5547, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 90.2223, mae: 6.1289, huber: 5.6589, swd: 0.9775, ept: 47.0484
    Epoch [31/50], Val Losses: mse: 99.0721, mae: 6.5585, huber: 6.0858, swd: 0.9939, ept: 38.4960
    Epoch [31/50], Test Losses: mse: 95.2538, mae: 6.1838, huber: 5.7149, swd: 0.8513, ept: 37.8973
      Epoch 31 composite train-obj: 6.147626
            No improvement (6.5828), counter 1/5
    Epoch [32/50], Train Losses: mse: 90.3515, mae: 6.1308, huber: 5.6606, swd: 1.0157, ept: 46.9067
    Epoch [32/50], Val Losses: mse: 98.5485, mae: 6.5493, huber: 6.0764, swd: 1.0107, ept: 38.7246
    Epoch [32/50], Test Losses: mse: 94.9664, mae: 6.1749, huber: 5.7064, swd: 0.8608, ept: 37.5527
      Epoch 32 composite train-obj: 6.168436
            No improvement (6.5818), counter 2/5
    Epoch [33/50], Train Losses: mse: 90.4005, mae: 6.1264, huber: 5.6563, swd: 0.9955, ept: 47.3483
    Epoch [33/50], Val Losses: mse: 99.4221, mae: 6.5607, huber: 6.0875, swd: 1.0762, ept: 39.0584
    Epoch [33/50], Test Losses: mse: 95.9054, mae: 6.2028, huber: 5.7337, swd: 0.8956, ept: 38.0025
      Epoch 33 composite train-obj: 6.154051
            No improvement (6.6256), counter 3/5
    Epoch [34/50], Train Losses: mse: 90.5836, mae: 6.1242, huber: 5.6541, swd: 0.9925, ept: 47.7170
    Epoch [34/50], Val Losses: mse: 99.3837, mae: 6.5396, huber: 6.0673, swd: 0.9392, ept: 39.1368
    Epoch [34/50], Test Losses: mse: 96.1696, mae: 6.1927, huber: 5.7243, swd: 0.7894, ept: 38.4091
      Epoch 34 composite train-obj: 6.150354
            Val objective improved 6.5547 → 6.5369, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 90.6263, mae: 6.1237, huber: 5.6536, swd: 0.9949, ept: 47.9014
    Epoch [35/50], Val Losses: mse: 99.1316, mae: 6.5383, huber: 6.0657, swd: 1.0206, ept: 39.4269
    Epoch [35/50], Test Losses: mse: 95.9145, mae: 6.2013, huber: 5.7320, swd: 0.8681, ept: 38.7611
      Epoch 35 composite train-obj: 6.151089
            No improvement (6.5760), counter 1/5
    Epoch [36/50], Train Losses: mse: 90.7849, mae: 6.1221, huber: 5.6521, swd: 1.0118, ept: 48.2702
    Epoch [36/50], Val Losses: mse: 99.6035, mae: 6.5388, huber: 6.0670, swd: 0.9540, ept: 39.6431
    Epoch [36/50], Test Losses: mse: 96.0842, mae: 6.1828, huber: 5.7146, swd: 0.8038, ept: 38.7766
      Epoch 36 composite train-obj: 6.157979
            No improvement (6.5440), counter 2/5
    Epoch [37/50], Train Losses: mse: 90.9174, mae: 6.1184, huber: 5.6485, swd: 0.9806, ept: 48.2629
    Epoch [37/50], Val Losses: mse: 99.7380, mae: 6.5555, huber: 6.0825, swd: 0.9905, ept: 40.1608
    Epoch [37/50], Test Losses: mse: 96.3724, mae: 6.2029, huber: 5.7336, swd: 0.8335, ept: 39.2232
      Epoch 37 composite train-obj: 6.138805
            No improvement (6.5777), counter 3/5
    Epoch [38/50], Train Losses: mse: 90.9623, mae: 6.1176, huber: 5.6477, swd: 0.9771, ept: 48.4968
    Epoch [38/50], Val Losses: mse: 99.9493, mae: 6.5371, huber: 6.0651, swd: 1.0260, ept: 39.4325
    Epoch [38/50], Test Losses: mse: 96.3394, mae: 6.1774, huber: 5.7092, swd: 0.8392, ept: 38.8786
      Epoch 38 composite train-obj: 6.136232
            No improvement (6.5781), counter 4/5
    Epoch [39/50], Train Losses: mse: 91.1008, mae: 6.1157, huber: 5.6459, swd: 0.9869, ept: 48.4698
    Epoch [39/50], Val Losses: mse: 99.6341, mae: 6.5377, huber: 6.0651, swd: 1.1028, ept: 40.0918
    Epoch [39/50], Test Losses: mse: 95.8643, mae: 6.1642, huber: 5.6957, swd: 0.9304, ept: 39.4601
      Epoch 39 composite train-obj: 6.139324
    Epoch [39/50], Test Losses: mse: 96.1696, mae: 6.1927, huber: 5.7243, swd: 0.7894, ept: 38.4091
    Best round's Test MSE: 96.1696, MAE: 6.1927, SWD: 0.7894
    Best round's Validation MSE: 99.3837, MAE: 6.5396, SWD: 0.9392
    Best round's Test verification MSE : 96.1696, MAE: 6.1927, SWD: 0.7894
    Time taken: 40.72 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 138.4952, mae: 8.2498, huber: 7.7669, swd: 2.6411, ept: 9.6041
    Epoch [1/50], Val Losses: mse: 85.7510, mae: 6.8299, huber: 6.3508, swd: 1.1466, ept: 19.3577
    Epoch [1/50], Test Losses: mse: 82.4392, mae: 6.4181, huber: 5.9425, swd: 1.0614, ept: 22.8336
      Epoch 1 composite train-obj: 9.087485
            Val objective improved inf → 6.9241, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 79.6455, mae: 6.4119, huber: 5.9362, swd: 1.3700, ept: 33.1807
    Epoch [2/50], Val Losses: mse: 86.0456, mae: 6.6740, huber: 6.1977, swd: 1.1793, ept: 32.5423
    Epoch [2/50], Test Losses: mse: 83.1505, mae: 6.2440, huber: 5.7719, swd: 1.0501, ept: 35.6675
      Epoch 2 composite train-obj: 6.621146
            Val objective improved 6.9241 → 6.7873, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 79.3551, mae: 6.2973, huber: 5.8237, swd: 1.3227, ept: 39.4610
    Epoch [3/50], Val Losses: mse: 86.8016, mae: 6.6391, huber: 6.1639, swd: 1.2748, ept: 34.2181
    Epoch [3/50], Test Losses: mse: 84.2004, mae: 6.2272, huber: 5.7563, swd: 1.0986, ept: 36.1439
      Epoch 3 composite train-obj: 6.485069
            No improvement (6.8012), counter 1/5
    Epoch [4/50], Train Losses: mse: 79.4250, mae: 6.2520, huber: 5.7794, swd: 1.3001, ept: 41.3786
    Epoch [4/50], Val Losses: mse: 87.3547, mae: 6.6318, huber: 6.1568, swd: 1.1237, ept: 34.5787
    Epoch [4/50], Test Losses: mse: 84.7339, mae: 6.2107, huber: 5.7400, swd: 0.9794, ept: 36.7380
      Epoch 4 composite train-obj: 6.429441
            Val objective improved 6.7873 → 6.7187, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 79.8488, mae: 6.2266, huber: 5.7546, swd: 1.2511, ept: 42.6935
    Epoch [5/50], Val Losses: mse: 87.8815, mae: 6.6130, huber: 6.1386, swd: 1.1804, ept: 35.1180
    Epoch [5/50], Test Losses: mse: 85.7109, mae: 6.2136, huber: 5.7436, swd: 1.0254, ept: 36.4839
      Epoch 5 composite train-obj: 6.380090
            No improvement (6.7288), counter 1/5
    Epoch [6/50], Train Losses: mse: 80.6669, mae: 6.2079, huber: 5.7362, swd: 1.2191, ept: 43.5237
    Epoch [6/50], Val Losses: mse: 88.7423, mae: 6.5834, huber: 6.1096, swd: 1.1299, ept: 34.9089
    Epoch [6/50], Test Losses: mse: 86.0684, mae: 6.1769, huber: 5.7073, swd: 0.9873, ept: 36.1871
      Epoch 6 composite train-obj: 6.345688
            Val objective improved 6.7187 → 6.6745, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 81.9377, mae: 6.1916, huber: 5.7203, swd: 1.1955, ept: 43.9168
    Epoch [7/50], Val Losses: mse: 90.6179, mae: 6.5943, huber: 6.1207, swd: 1.1111, ept: 36.1240
    Epoch [7/50], Test Losses: mse: 87.5744, mae: 6.1763, huber: 5.7073, swd: 0.9538, ept: 36.6526
      Epoch 7 composite train-obj: 6.318007
            No improvement (6.6762), counter 1/5
    Epoch [8/50], Train Losses: mse: 83.0122, mae: 6.1783, huber: 5.7072, swd: 1.1811, ept: 44.6194
    Epoch [8/50], Val Losses: mse: 91.0458, mae: 6.5636, huber: 6.0902, swd: 1.1504, ept: 35.6750
    Epoch [8/50], Test Losses: mse: 88.3213, mae: 6.1804, huber: 5.7108, swd: 1.0236, ept: 36.1586
      Epoch 8 composite train-obj: 6.297754
            Val objective improved 6.6745 → 6.6654, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 83.8407, mae: 6.1686, huber: 5.6976, swd: 1.1450, ept: 44.8909
    Epoch [9/50], Val Losses: mse: 92.0611, mae: 6.5632, huber: 6.0896, swd: 1.1406, ept: 36.2562
    Epoch [9/50], Test Losses: mse: 89.7294, mae: 6.1996, huber: 5.7300, swd: 0.9827, ept: 36.1795
      Epoch 9 composite train-obj: 6.270142
            Val objective improved 6.6654 → 6.6599, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 84.5505, mae: 6.1628, huber: 5.6919, swd: 1.1512, ept: 45.1905
    Epoch [10/50], Val Losses: mse: 92.7096, mae: 6.5664, huber: 6.0929, swd: 1.0662, ept: 36.8983
    Epoch [10/50], Test Losses: mse: 89.7873, mae: 6.1700, huber: 5.7006, swd: 0.9283, ept: 37.0822
      Epoch 10 composite train-obj: 6.267506
            Val objective improved 6.6599 → 6.6260, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 85.1234, mae: 6.1588, huber: 5.6880, swd: 1.1473, ept: 45.3920
    Epoch [11/50], Val Losses: mse: 92.8082, mae: 6.5424, huber: 6.0690, swd: 1.1625, ept: 36.7466
    Epoch [11/50], Test Losses: mse: 90.3202, mae: 6.1684, huber: 5.6994, swd: 0.9821, ept: 36.5479
      Epoch 11 composite train-obj: 6.261673
            No improvement (6.6502), counter 1/5
    Epoch [12/50], Train Losses: mse: 85.5228, mae: 6.1529, huber: 5.6824, swd: 1.1263, ept: 45.7665
    Epoch [12/50], Val Losses: mse: 93.5994, mae: 6.5556, huber: 6.0821, swd: 1.0927, ept: 36.5638
    Epoch [12/50], Test Losses: mse: 90.3656, mae: 6.1515, huber: 5.6827, swd: 0.9456, ept: 36.7590
      Epoch 12 composite train-obj: 6.245500
            No improvement (6.6285), counter 2/5
    Epoch [13/50], Train Losses: mse: 85.8361, mae: 6.1523, huber: 5.6818, swd: 1.1429, ept: 45.6802
    Epoch [13/50], Val Losses: mse: 94.1841, mae: 6.5621, huber: 6.0890, swd: 1.0807, ept: 36.9596
    Epoch [13/50], Test Losses: mse: 91.3720, mae: 6.1823, huber: 5.7134, swd: 0.9354, ept: 37.0680
      Epoch 13 composite train-obj: 6.253248
            No improvement (6.6293), counter 3/5
    Epoch [14/50], Train Losses: mse: 86.0526, mae: 6.1488, huber: 5.6783, swd: 1.1320, ept: 46.0608
    Epoch [14/50], Val Losses: mse: 94.4005, mae: 6.5598, huber: 6.0868, swd: 1.1029, ept: 37.9212
    Epoch [14/50], Test Losses: mse: 91.4129, mae: 6.1698, huber: 5.7011, swd: 0.9779, ept: 37.0765
      Epoch 14 composite train-obj: 6.244350
            No improvement (6.6383), counter 4/5
    Epoch [15/50], Train Losses: mse: 86.2103, mae: 6.1441, huber: 5.6737, swd: 1.1070, ept: 46.4948
    Epoch [15/50], Val Losses: mse: 93.8144, mae: 6.5185, huber: 6.0454, swd: 1.1148, ept: 37.9894
    Epoch [15/50], Test Losses: mse: 91.0738, mae: 6.1448, huber: 5.6759, swd: 0.9636, ept: 37.5331
      Epoch 15 composite train-obj: 6.227175
            Val objective improved 6.6260 → 6.6028, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 86.3678, mae: 6.1443, huber: 5.6740, swd: 1.1144, ept: 46.4734
    Epoch [16/50], Val Losses: mse: 94.1138, mae: 6.5227, huber: 6.0499, swd: 1.0932, ept: 38.3943
    Epoch [16/50], Test Losses: mse: 91.5240, mae: 6.1557, huber: 5.6872, swd: 0.9500, ept: 37.7986
      Epoch 16 composite train-obj: 6.231229
            Val objective improved 6.6028 → 6.5965, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 86.5072, mae: 6.1418, huber: 5.6716, swd: 1.1186, ept: 46.5357
    Epoch [17/50], Val Losses: mse: 94.5742, mae: 6.5393, huber: 6.0660, swd: 1.0971, ept: 37.8048
    Epoch [17/50], Test Losses: mse: 91.3179, mae: 6.1425, huber: 5.6742, swd: 0.9445, ept: 37.6903
      Epoch 17 composite train-obj: 6.230914
            No improvement (6.6145), counter 1/5
    Epoch [18/50], Train Losses: mse: 86.5998, mae: 6.1425, huber: 5.6721, swd: 1.1360, ept: 46.2451
    Epoch [18/50], Val Losses: mse: 94.5793, mae: 6.5488, huber: 6.0748, swd: 1.1893, ept: 37.7082
    Epoch [18/50], Test Losses: mse: 91.7923, mae: 6.1811, huber: 5.7107, swd: 1.0683, ept: 37.1092
      Epoch 18 composite train-obj: 6.240098
            No improvement (6.6695), counter 2/5
    Epoch [19/50], Train Losses: mse: 86.7085, mae: 6.1379, huber: 5.6678, swd: 1.1102, ept: 46.8728
    Epoch [19/50], Val Losses: mse: 95.2099, mae: 6.5615, huber: 6.0885, swd: 1.1120, ept: 38.5054
    Epoch [19/50], Test Losses: mse: 91.7810, mae: 6.1615, huber: 5.6931, swd: 0.9740, ept: 37.5262
      Epoch 19 composite train-obj: 6.222906
            No improvement (6.6445), counter 3/5
    Epoch [20/50], Train Losses: mse: 86.7980, mae: 6.1370, huber: 5.6669, swd: 1.1123, ept: 46.6674
    Epoch [20/50], Val Losses: mse: 94.7911, mae: 6.5367, huber: 6.0636, swd: 1.0723, ept: 37.8659
    Epoch [20/50], Test Losses: mse: 91.9569, mae: 6.1609, huber: 5.6920, swd: 0.9488, ept: 37.1807
      Epoch 20 composite train-obj: 6.223097
            No improvement (6.5998), counter 4/5
    Epoch [21/50], Train Losses: mse: 86.8461, mae: 6.1347, huber: 5.6646, swd: 1.1117, ept: 46.7326
    Epoch [21/50], Val Losses: mse: 95.4299, mae: 6.5679, huber: 6.0945, swd: 1.2281, ept: 38.5345
    Epoch [21/50], Test Losses: mse: 92.5898, mae: 6.1831, huber: 5.7143, swd: 1.0702, ept: 37.4462
      Epoch 21 composite train-obj: 6.220418
    Epoch [21/50], Test Losses: mse: 91.5240, mae: 6.1557, huber: 5.6872, swd: 0.9500, ept: 37.7986
    Best round's Test MSE: 91.5240, MAE: 6.1557, SWD: 0.9500
    Best round's Validation MSE: 94.1138, MAE: 6.5227, SWD: 1.0932
    Best round's Test verification MSE : 91.5240, MAE: 6.1557, SWD: 0.9500
    Time taken: 21.52 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred336_20250514_2138)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 93.2187 ± 2.0943
      mae: 6.1842 ± 0.0207
      huber: 5.7155 ± 0.0205
      swd: 0.8447 ± 0.0745
      ept: 37.1845 ± 1.3238
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 95.9082 ± 2.4580
      mae: 6.5414 ± 0.0160
      huber: 6.0685 ± 0.0157
      swd: 0.9907 ± 0.0725
      ept: 38.0917 ± 1.0001
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 84.10 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred336_20250514_2138
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE


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
    loss_backward_weights = [1.0, 0.0, 0.0, 0.0, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.0, 0.0],
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.1245, mae: 6.4317, huber: 5.9511, swd: 28.6219, ept: 34.9660
    Epoch [1/50], Val Losses: mse: 63.2887, mae: 6.1536, huber: 5.6736, swd: 29.4576, ept: 40.4814
    Epoch [1/50], Test Losses: mse: 58.6343, mae: 5.7763, huber: 5.2991, swd: 30.6438, ept: 44.5277
      Epoch 1 composite train-obj: 71.124484
            Val objective improved inf → 63.2887, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.2280, mae: 5.9555, huber: 5.4770, swd: 26.9487, ept: 53.4454
    Epoch [2/50], Val Losses: mse: 63.6956, mae: 6.1566, huber: 5.6773, swd: 26.4818, ept: 46.0533
    Epoch [2/50], Test Losses: mse: 59.4251, mae: 5.8074, huber: 5.3310, swd: 27.8174, ept: 50.2992
      Epoch 2 composite train-obj: 60.227985
            No improvement (63.6956), counter 1/5
    Epoch [3/50], Train Losses: mse: 59.7278, mae: 5.9161, huber: 5.4382, swd: 26.3059, ept: 56.7602
    Epoch [3/50], Val Losses: mse: 63.1941, mae: 6.1124, huber: 5.6338, swd: 26.6398, ept: 47.3880
    Epoch [3/50], Test Losses: mse: 58.5120, mae: 5.7496, huber: 5.2739, swd: 28.2535, ept: 51.7731
      Epoch 3 composite train-obj: 59.727813
            Val objective improved 63.2887 → 63.1941, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 59.5570, mae: 5.8992, huber: 5.4217, swd: 26.1220, ept: 58.2178
    Epoch [4/50], Val Losses: mse: 63.3846, mae: 6.1251, huber: 5.6468, swd: 25.9727, ept: 48.7279
    Epoch [4/50], Test Losses: mse: 58.8418, mae: 5.7768, huber: 5.3007, swd: 27.4288, ept: 54.0172
      Epoch 4 composite train-obj: 59.557045
            No improvement (63.3846), counter 1/5
    Epoch [5/50], Train Losses: mse: 59.5451, mae: 5.8951, huber: 5.4179, swd: 25.8543, ept: 58.8518
    Epoch [5/50], Val Losses: mse: 64.5843, mae: 6.1755, huber: 5.6975, swd: 24.9911, ept: 49.2138
    Epoch [5/50], Test Losses: mse: 59.6643, mae: 5.7977, huber: 5.3223, swd: 26.7307, ept: 55.0559
      Epoch 5 composite train-obj: 59.545133
            No improvement (64.5843), counter 2/5
    Epoch [6/50], Train Losses: mse: 59.4336, mae: 5.8862, huber: 5.4094, swd: 25.7541, ept: 59.8879
    Epoch [6/50], Val Losses: mse: 63.3956, mae: 6.1052, huber: 5.6275, swd: 25.7501, ept: 48.9021
    Epoch [6/50], Test Losses: mse: 59.1415, mae: 5.7699, huber: 5.2952, swd: 27.7362, ept: 53.3735
      Epoch 6 composite train-obj: 59.433643
            No improvement (63.3956), counter 3/5
    Epoch [7/50], Train Losses: mse: 59.3770, mae: 5.8824, huber: 5.4057, swd: 25.6938, ept: 60.3351
    Epoch [7/50], Val Losses: mse: 63.3303, mae: 6.1088, huber: 5.6310, swd: 25.5055, ept: 51.4669
    Epoch [7/50], Test Losses: mse: 58.9135, mae: 5.7585, huber: 5.2829, swd: 27.1210, ept: 57.4370
      Epoch 7 composite train-obj: 59.377037
            No improvement (63.3303), counter 4/5
    Epoch [8/50], Train Losses: mse: 59.3750, mae: 5.8783, huber: 5.4019, swd: 25.6533, ept: 60.8524
    Epoch [8/50], Val Losses: mse: 62.4607, mae: 6.0596, huber: 5.5821, swd: 26.4772, ept: 51.7498
    Epoch [8/50], Test Losses: mse: 58.1234, mae: 5.7237, huber: 5.2481, swd: 28.0973, ept: 57.7989
      Epoch 8 composite train-obj: 59.375022
            Val objective improved 63.1941 → 62.4607, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 59.4781, mae: 5.8842, huber: 5.4079, swd: 25.5872, ept: 60.6503
    Epoch [9/50], Val Losses: mse: 63.6455, mae: 6.1211, huber: 5.6444, swd: 25.4061, ept: 50.7651
    Epoch [9/50], Test Losses: mse: 58.9591, mae: 5.7585, huber: 5.2837, swd: 26.6638, ept: 55.6087
      Epoch 9 composite train-obj: 59.478086
            No improvement (63.6455), counter 1/5
    Epoch [10/50], Train Losses: mse: 59.3358, mae: 5.8758, huber: 5.3997, swd: 25.6274, ept: 60.5909
    Epoch [10/50], Val Losses: mse: 63.3492, mae: 6.1048, huber: 5.6273, swd: 26.0919, ept: 51.5044
    Epoch [10/50], Test Losses: mse: 58.7764, mae: 5.7588, huber: 5.2838, swd: 27.4731, ept: 57.1860
      Epoch 10 composite train-obj: 59.335757
            No improvement (63.3492), counter 2/5
    Epoch [11/50], Train Losses: mse: 59.4397, mae: 5.8800, huber: 5.4040, swd: 25.3470, ept: 61.3987
    Epoch [11/50], Val Losses: mse: 63.4418, mae: 6.1259, huber: 5.6482, swd: 24.9672, ept: 52.3620
    Epoch [11/50], Test Losses: mse: 58.4026, mae: 5.7474, huber: 5.2716, swd: 26.4434, ept: 58.4186
      Epoch 11 composite train-obj: 59.439722
            No improvement (63.4418), counter 3/5
    Epoch [12/50], Train Losses: mse: 59.3345, mae: 5.8732, huber: 5.3973, swd: 25.4088, ept: 61.4994
    Epoch [12/50], Val Losses: mse: 62.6007, mae: 6.0688, huber: 5.5913, swd: 26.5339, ept: 51.2398
    Epoch [12/50], Test Losses: mse: 58.1147, mae: 5.7212, huber: 5.2463, swd: 28.0246, ept: 54.7816
      Epoch 12 composite train-obj: 59.334482
            No improvement (62.6007), counter 4/5
    Epoch [13/50], Train Losses: mse: 59.3424, mae: 5.8723, huber: 5.3966, swd: 25.4294, ept: 61.8937
    Epoch [13/50], Val Losses: mse: 63.0670, mae: 6.0928, huber: 5.6158, swd: 25.8947, ept: 52.6592
    Epoch [13/50], Test Losses: mse: 58.0489, mae: 5.7178, huber: 5.2433, swd: 27.4461, ept: 58.7607
      Epoch 13 composite train-obj: 59.342424
    Epoch [13/50], Test Losses: mse: 58.1234, mae: 5.7237, huber: 5.2481, swd: 28.0973, ept: 57.7989
    Best round's Test MSE: 58.1234, MAE: 5.7237, SWD: 28.0973
    Best round's Validation MSE: 62.4607, MAE: 6.0596, SWD: 26.4772
    Best round's Test verification MSE : 58.1234, MAE: 5.7237, SWD: 28.0973
    Time taken: 13.46 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.8339, mae: 6.4495, huber: 5.9688, swd: 28.8183, ept: 34.6107
    Epoch [1/50], Val Losses: mse: 64.0547, mae: 6.1882, huber: 5.7085, swd: 28.3857, ept: 39.6991
    Epoch [1/50], Test Losses: mse: 59.4137, mae: 5.8233, huber: 5.3462, swd: 29.7756, ept: 42.8234
      Epoch 1 composite train-obj: 71.833899
            Val objective improved inf → 64.0547, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.2292, mae: 5.9576, huber: 5.4790, swd: 26.9287, ept: 53.5117
    Epoch [2/50], Val Losses: mse: 63.5719, mae: 6.1363, huber: 5.6571, swd: 27.5415, ept: 46.2157
    Epoch [2/50], Test Losses: mse: 59.3623, mae: 5.7914, huber: 5.3151, swd: 28.9365, ept: 50.5346
      Epoch 2 composite train-obj: 60.229243
            Val objective improved 64.0547 → 63.5719, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 59.8044, mae: 5.9195, huber: 5.4417, swd: 26.5431, ept: 56.8988
    Epoch [3/50], Val Losses: mse: 63.6464, mae: 6.1392, huber: 5.6606, swd: 26.3043, ept: 47.9917
    Epoch [3/50], Test Losses: mse: 58.7480, mae: 5.7741, huber: 5.2982, swd: 28.3014, ept: 52.6778
      Epoch 3 composite train-obj: 59.804397
            No improvement (63.6464), counter 1/5
    Epoch [4/50], Train Losses: mse: 59.6077, mae: 5.9042, huber: 5.4267, swd: 26.1544, ept: 58.1299
    Epoch [4/50], Val Losses: mse: 63.0171, mae: 6.0900, huber: 5.6122, swd: 27.2326, ept: 47.8650
    Epoch [4/50], Test Losses: mse: 58.3406, mae: 5.7344, huber: 5.2589, swd: 28.8191, ept: 52.5766
      Epoch 4 composite train-obj: 59.607704
            Val objective improved 63.5719 → 63.0171, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 59.5372, mae: 5.8944, huber: 5.4173, swd: 25.9934, ept: 58.6040
    Epoch [5/50], Val Losses: mse: 63.0311, mae: 6.0956, huber: 5.6179, swd: 27.2655, ept: 49.2481
    Epoch [5/50], Test Losses: mse: 58.8171, mae: 5.7784, huber: 5.3024, swd: 29.4460, ept: 52.7176
      Epoch 5 composite train-obj: 59.537174
            No improvement (63.0311), counter 1/5
    Epoch [6/50], Train Losses: mse: 59.4293, mae: 5.8875, huber: 5.4106, swd: 25.8695, ept: 59.7346
    Epoch [6/50], Val Losses: mse: 62.5878, mae: 6.0654, huber: 5.5878, swd: 26.4928, ept: 49.1511
    Epoch [6/50], Test Losses: mse: 58.1436, mae: 5.7226, huber: 5.2475, swd: 28.1615, ept: 54.9298
      Epoch 6 composite train-obj: 59.429301
            Val objective improved 63.0171 → 62.5878, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 59.4897, mae: 5.8877, huber: 5.4112, swd: 25.8574, ept: 59.4968
    Epoch [7/50], Val Losses: mse: 63.3954, mae: 6.1115, huber: 5.6339, swd: 25.9371, ept: 51.6206
    Epoch [7/50], Test Losses: mse: 58.1323, mae: 5.7188, huber: 5.2438, swd: 28.3190, ept: 57.0705
      Epoch 7 composite train-obj: 59.489665
            No improvement (63.3954), counter 1/5
    Epoch [8/50], Train Losses: mse: 59.4762, mae: 5.8866, huber: 5.4102, swd: 25.6322, ept: 60.1564
    Epoch [8/50], Val Losses: mse: 62.9784, mae: 6.0910, huber: 5.6139, swd: 26.5158, ept: 50.6390
    Epoch [8/50], Test Losses: mse: 58.6447, mae: 5.7544, huber: 5.2792, swd: 28.3061, ept: 54.2034
      Epoch 8 composite train-obj: 59.476237
            No improvement (62.9784), counter 2/5
    Epoch [9/50], Train Losses: mse: 59.4059, mae: 5.8810, huber: 5.4048, swd: 25.7356, ept: 59.8461
    Epoch [9/50], Val Losses: mse: 62.9479, mae: 6.0714, huber: 5.5943, swd: 25.9737, ept: 50.7942
    Epoch [9/50], Test Losses: mse: 58.3401, mae: 5.7025, huber: 5.2288, swd: 28.3654, ept: 56.3232
      Epoch 9 composite train-obj: 59.405862
            No improvement (62.9479), counter 3/5
    Epoch [10/50], Train Losses: mse: 59.3237, mae: 5.8733, huber: 5.3972, swd: 25.6219, ept: 61.6247
    Epoch [10/50], Val Losses: mse: 63.8060, mae: 6.1434, huber: 5.6653, swd: 26.2450, ept: 49.1073
    Epoch [10/50], Test Losses: mse: 59.5081, mae: 5.8056, huber: 5.3309, swd: 28.1076, ept: 53.8119
      Epoch 10 composite train-obj: 59.323666
            No improvement (63.8060), counter 4/5
    Epoch [11/50], Train Losses: mse: 59.2848, mae: 5.8711, huber: 5.3952, swd: 25.5452, ept: 61.7949
    Epoch [11/50], Val Losses: mse: 63.5436, mae: 6.1073, huber: 5.6300, swd: 25.2415, ept: 53.7118
    Epoch [11/50], Test Losses: mse: 58.7441, mae: 5.7311, huber: 5.2566, swd: 27.0648, ept: 59.6574
      Epoch 11 composite train-obj: 59.284849
    Epoch [11/50], Test Losses: mse: 58.1436, mae: 5.7226, huber: 5.2475, swd: 28.1615, ept: 54.9298
    Best round's Test MSE: 58.1436, MAE: 5.7226, SWD: 28.1615
    Best round's Validation MSE: 62.5878, MAE: 6.0654, SWD: 26.4928
    Best round's Test verification MSE : 58.1436, MAE: 5.7226, SWD: 28.1615
    Time taken: 11.42 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 71.3375, mae: 6.4451, huber: 5.9644, swd: 28.8404, ept: 34.5130
    Epoch [1/50], Val Losses: mse: 63.5890, mae: 6.1555, huber: 5.6766, swd: 28.8227, ept: 40.5000
    Epoch [1/50], Test Losses: mse: 59.6165, mae: 5.8379, huber: 5.3602, swd: 30.1023, ept: 45.3557
      Epoch 1 composite train-obj: 71.337480
            Val objective improved inf → 63.5890, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 60.2518, mae: 5.9590, huber: 5.4804, swd: 26.9622, ept: 53.4383
    Epoch [2/50], Val Losses: mse: 63.6025, mae: 6.1334, huber: 5.6553, swd: 27.4813, ept: 46.2522
    Epoch [2/50], Test Losses: mse: 58.2494, mae: 5.7420, huber: 5.2661, swd: 29.4096, ept: 52.5893
      Epoch 2 composite train-obj: 60.251828
            No improvement (63.6025), counter 1/5
    Epoch [3/50], Train Losses: mse: 59.7940, mae: 5.9203, huber: 5.4424, swd: 26.5921, ept: 57.0237
    Epoch [3/50], Val Losses: mse: 63.2631, mae: 6.1017, huber: 5.6235, swd: 26.2967, ept: 47.5942
    Epoch [3/50], Test Losses: mse: 59.1902, mae: 5.7784, huber: 5.3022, swd: 27.7121, ept: 52.3532
      Epoch 3 composite train-obj: 59.793999
            Val objective improved 63.5890 → 63.2631, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 59.6098, mae: 5.9054, huber: 5.4279, swd: 26.2480, ept: 58.1260
    Epoch [4/50], Val Losses: mse: 63.2768, mae: 6.1186, huber: 5.6408, swd: 27.4617, ept: 48.2195
    Epoch [4/50], Test Losses: mse: 58.8213, mae: 5.7788, huber: 5.3029, swd: 29.0806, ept: 53.1379
      Epoch 4 composite train-obj: 59.609757
            No improvement (63.2768), counter 1/5
    Epoch [5/50], Train Losses: mse: 59.5434, mae: 5.8974, huber: 5.4202, swd: 26.0756, ept: 58.3011
    Epoch [5/50], Val Losses: mse: 62.9340, mae: 6.0946, huber: 5.6174, swd: 26.8836, ept: 49.3432
    Epoch [5/50], Test Losses: mse: 58.8080, mae: 5.7589, huber: 5.2834, swd: 28.1601, ept: 55.4220
      Epoch 5 composite train-obj: 59.543423
            Val objective improved 63.2631 → 62.9340, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 59.5149, mae: 5.8912, huber: 5.4143, swd: 25.8862, ept: 59.6910
    Epoch [6/50], Val Losses: mse: 63.1485, mae: 6.1186, huber: 5.6401, swd: 26.5386, ept: 47.8153
    Epoch [6/50], Test Losses: mse: 58.5268, mae: 5.7563, huber: 5.2807, swd: 28.3031, ept: 52.2893
      Epoch 6 composite train-obj: 59.514930
            No improvement (63.1485), counter 1/5
    Epoch [7/50], Train Losses: mse: 59.5175, mae: 5.8888, huber: 5.4121, swd: 25.7064, ept: 59.7621
    Epoch [7/50], Val Losses: mse: 63.4509, mae: 6.1108, huber: 5.6332, swd: 26.3551, ept: 51.7547
    Epoch [7/50], Test Losses: mse: 58.1643, mae: 5.7122, huber: 5.2375, swd: 27.9875, ept: 57.6592
      Epoch 7 composite train-obj: 59.517463
            No improvement (63.4509), counter 2/5
    Epoch [8/50], Train Losses: mse: 59.4046, mae: 5.8833, huber: 5.4068, swd: 25.8262, ept: 59.9421
    Epoch [8/50], Val Losses: mse: 63.1363, mae: 6.0975, huber: 5.6201, swd: 25.8479, ept: 49.8200
    Epoch [8/50], Test Losses: mse: 58.5584, mae: 5.7448, huber: 5.2697, swd: 27.7489, ept: 54.4730
      Epoch 8 composite train-obj: 59.404561
            No improvement (63.1363), counter 3/5
    Epoch [9/50], Train Losses: mse: 59.4123, mae: 5.8807, huber: 5.4043, swd: 25.6752, ept: 60.3509
    Epoch [9/50], Val Losses: mse: 62.7381, mae: 6.0723, huber: 5.5939, swd: 26.4091, ept: 50.1916
    Epoch [9/50], Test Losses: mse: 58.8059, mae: 5.7542, huber: 5.2793, swd: 28.0938, ept: 55.3153
      Epoch 9 composite train-obj: 59.412349
            Val objective improved 62.9340 → 62.7381, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 59.2558, mae: 5.8706, huber: 5.3944, swd: 25.7246, ept: 61.6696
    Epoch [10/50], Val Losses: mse: 62.8178, mae: 6.0716, huber: 5.5945, swd: 26.7688, ept: 47.8967
    Epoch [10/50], Test Losses: mse: 58.1203, mae: 5.7161, huber: 5.2417, swd: 28.4255, ept: 51.9265
      Epoch 10 composite train-obj: 59.255806
            No improvement (62.8178), counter 1/5
    Epoch [11/50], Train Losses: mse: 59.4104, mae: 5.8788, huber: 5.4027, swd: 25.6632, ept: 61.1437
    Epoch [11/50], Val Losses: mse: 62.6969, mae: 6.0878, huber: 5.6102, swd: 26.2365, ept: 51.3952
    Epoch [11/50], Test Losses: mse: 58.5363, mae: 5.7496, huber: 5.2746, swd: 28.2548, ept: 57.5614
      Epoch 11 composite train-obj: 59.410381
            Val objective improved 62.7381 → 62.6969, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 59.3472, mae: 5.8750, huber: 5.3991, swd: 25.5821, ept: 61.3600
    Epoch [12/50], Val Losses: mse: 63.2223, mae: 6.0997, huber: 5.6224, swd: 26.1480, ept: 51.7088
    Epoch [12/50], Test Losses: mse: 57.8489, mae: 5.7004, huber: 5.2266, swd: 28.3085, ept: 57.3476
      Epoch 12 composite train-obj: 59.347194
            No improvement (63.2223), counter 1/5
    Epoch [13/50], Train Losses: mse: 59.4024, mae: 5.8778, huber: 5.4021, swd: 25.5402, ept: 61.7852
    Epoch [13/50], Val Losses: mse: 63.5308, mae: 6.1096, huber: 5.6318, swd: 25.6031, ept: 52.2259
    Epoch [13/50], Test Losses: mse: 58.6354, mae: 5.7499, huber: 5.2750, swd: 27.5007, ept: 56.9988
      Epoch 13 composite train-obj: 59.402384
            No improvement (63.5308), counter 2/5
    Epoch [14/50], Train Losses: mse: 59.2439, mae: 5.8666, huber: 5.3910, swd: 25.5165, ept: 62.4696
    Epoch [14/50], Val Losses: mse: 63.7114, mae: 6.1256, huber: 5.6485, swd: 25.0836, ept: 52.4397
    Epoch [14/50], Test Losses: mse: 58.7255, mae: 5.7303, huber: 5.2563, swd: 26.8855, ept: 59.1118
      Epoch 14 composite train-obj: 59.243887
            No improvement (63.7114), counter 3/5
    Epoch [15/50], Train Losses: mse: 59.2401, mae: 5.8634, huber: 5.3878, swd: 25.3850, ept: 62.9106
    Epoch [15/50], Val Losses: mse: 62.4917, mae: 6.0525, huber: 5.5746, swd: 25.5813, ept: 53.5451
    Epoch [15/50], Test Losses: mse: 57.8425, mae: 5.6669, huber: 5.1940, swd: 27.1627, ept: 61.6740
      Epoch 15 composite train-obj: 59.240141
            Val objective improved 62.6969 → 62.4917, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 59.2860, mae: 5.8690, huber: 5.3934, swd: 25.4348, ept: 62.4308
    Epoch [16/50], Val Losses: mse: 62.6366, mae: 6.0650, huber: 5.5877, swd: 26.3870, ept: 53.5206
    Epoch [16/50], Test Losses: mse: 58.5857, mae: 5.7318, huber: 5.2583, swd: 27.9930, ept: 60.0984
      Epoch 16 composite train-obj: 59.286024
            No improvement (62.6366), counter 1/5
    Epoch [17/50], Train Losses: mse: 59.2452, mae: 5.8639, huber: 5.3885, swd: 25.4189, ept: 63.0451
    Epoch [17/50], Val Losses: mse: 63.0394, mae: 6.0972, huber: 5.6202, swd: 25.2034, ept: 55.4364
    Epoch [17/50], Test Losses: mse: 57.7571, mae: 5.7004, huber: 5.2258, swd: 27.2007, ept: 60.8751
      Epoch 17 composite train-obj: 59.245245
            No improvement (63.0394), counter 2/5
    Epoch [18/50], Train Losses: mse: 59.2931, mae: 5.8694, huber: 5.3941, swd: 25.4110, ept: 62.7756
    Epoch [18/50], Val Losses: mse: 62.9622, mae: 6.0932, huber: 5.6160, swd: 25.6087, ept: 52.8321
    Epoch [18/50], Test Losses: mse: 58.5057, mae: 5.7464, huber: 5.2718, swd: 27.6029, ept: 59.3574
      Epoch 18 composite train-obj: 59.293132
            No improvement (62.9622), counter 3/5
    Epoch [19/50], Train Losses: mse: 59.3135, mae: 5.8677, huber: 5.3924, swd: 25.2495, ept: 63.2170
    Epoch [19/50], Val Losses: mse: 64.0574, mae: 6.1547, huber: 5.6771, swd: 25.0273, ept: 55.4909
    Epoch [19/50], Test Losses: mse: 58.6247, mae: 5.7509, huber: 5.2767, swd: 27.2518, ept: 60.8603
      Epoch 19 composite train-obj: 59.313516
            No improvement (64.0574), counter 4/5
    Epoch [20/50], Train Losses: mse: 59.2795, mae: 5.8649, huber: 5.3898, swd: 25.4052, ept: 63.6314
    Epoch [20/50], Val Losses: mse: 63.5063, mae: 6.1216, huber: 5.6442, swd: 25.7468, ept: 52.2662
    Epoch [20/50], Test Losses: mse: 58.5843, mae: 5.7651, huber: 5.2906, swd: 27.7674, ept: 56.9708
      Epoch 20 composite train-obj: 59.279471
    Epoch [20/50], Test Losses: mse: 57.8425, mae: 5.6669, huber: 5.1940, swd: 27.1627, ept: 61.6740
    Best round's Test MSE: 57.8425, MAE: 5.6669, SWD: 27.1627
    Best round's Validation MSE: 62.4917, MAE: 6.0525, SWD: 25.5813
    Best round's Test verification MSE : 57.8425, MAE: 5.6669, SWD: 27.1627
    Time taken: 21.38 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred336_20250514_2139)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 58.0365 ± 0.1374
      mae: 5.7044 ± 0.0265
      huber: 5.2299 ± 0.0254
      swd: 27.8072 ± 0.4565
      ept: 58.1342 ± 2.7635
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 62.5134 ± 0.0541
      mae: 6.0592 ± 0.0053
      huber: 5.5815 ± 0.0054
      swd: 26.1838 ± 0.4260
      ept: 51.4820 ± 1.8038
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 46.28 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred336_20250514_2139
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### MSE + 0.5 SWD


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
    loss_backward_weights = [1.0, 0.0, 0.0, 0.5, 0.0],
    loss_validate_weights = [1.0, 0.0, 0.0, 0.5, 0.0],
)
exp = execute_model_evaluation('lorenz', cfg, data_mgr, scale=False)
```

    Reloading modules...
      Reloaded: utils
    Module reload complete.
    Shape of training data: torch.Size([17500, 3])
    Shape of validation data: torch.Size([2500, 3])
    Shape of testing data: torch.Size([5000, 3])
    global_std.shape: torch.Size([3])
    Global Std for lorenz: tensor([7.9152, 9.0134, 8.6069], device='cuda:0')
    Train set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Validation set sample shapes: torch.Size([336, 3]), torch.Size([336, 3])
    Test set data shapes: torch.Size([5000, 3]), torch.Size([5000, 3])
    Number of batches in train_loader: 132
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
    Training Batches: 132
    Validation Batches: 15
    Test Batches: 34
    ==================================================
    
    ==================================================
     Running experiment with seed 1955 (1/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.4739, mae: 6.6581, huber: 6.1759, swd: 9.4787, ept: 26.9658
    Epoch [1/50], Val Losses: mse: 68.6949, mae: 6.4402, huber: 5.9588, swd: 8.8171, ept: 33.9291
    Epoch [1/50], Test Losses: mse: 63.8976, mae: 6.0649, huber: 5.5858, swd: 8.5858, ept: 38.8033
      Epoch 1 composite train-obj: 80.213225
            Val objective improved inf → 73.1034, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.9678, mae: 6.1544, huber: 5.6740, swd: 7.7888, ept: 45.2128
    Epoch [2/50], Val Losses: mse: 69.3565, mae: 6.4486, huber: 5.9677, swd: 7.9662, ept: 39.0040
    Epoch [2/50], Test Losses: mse: 64.8611, mae: 6.0987, huber: 5.6198, swd: 8.0441, ept: 44.0559
      Epoch 2 composite train-obj: 67.862229
            No improvement (73.3396), counter 1/5
    Epoch [3/50], Train Losses: mse: 63.3624, mae: 6.1118, huber: 5.6318, swd: 7.6809, ept: 49.2539
    Epoch [3/50], Val Losses: mse: 68.3575, mae: 6.3882, huber: 5.9077, swd: 8.5482, ept: 42.1300
    Epoch [3/50], Test Losses: mse: 63.6213, mae: 6.0340, huber: 5.5557, swd: 8.6013, ept: 48.1968
      Epoch 3 composite train-obj: 67.202837
            Val objective improved 73.1034 → 72.6316, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 63.1271, mae: 6.0947, huber: 5.6149, swd: 7.6731, ept: 50.8396
    Epoch [4/50], Val Losses: mse: 68.8003, mae: 6.4106, huber: 5.9299, swd: 7.8626, ept: 42.3923
    Epoch [4/50], Test Losses: mse: 64.3339, mae: 6.0692, huber: 5.5903, swd: 7.7313, ept: 48.2216
      Epoch 4 composite train-obj: 66.963669
            No improvement (72.7316), counter 1/5
    Epoch [5/50], Train Losses: mse: 63.0919, mae: 6.0908, huber: 5.6110, swd: 7.6291, ept: 51.1936
    Epoch [5/50], Val Losses: mse: 69.8916, mae: 6.4459, huber: 5.9659, swd: 7.7394, ept: 43.1910
    Epoch [5/50], Test Losses: mse: 65.0673, mae: 6.0878, huber: 5.6094, swd: 7.8610, ept: 48.9095
      Epoch 5 composite train-obj: 66.906473
            No improvement (73.7613), counter 2/5
    Epoch [6/50], Train Losses: mse: 62.9777, mae: 6.0811, huber: 5.6015, swd: 7.6419, ept: 52.2212
    Epoch [6/50], Val Losses: mse: 68.9302, mae: 6.4003, huber: 5.9201, swd: 7.6056, ept: 42.9789
    Epoch [6/50], Test Losses: mse: 64.3420, mae: 6.0599, huber: 5.5819, swd: 7.8698, ept: 47.1047
      Epoch 6 composite train-obj: 66.798690
            No improvement (72.7330), counter 3/5
    Epoch [7/50], Train Losses: mse: 62.9098, mae: 6.0787, huber: 5.5991, swd: 7.6141, ept: 52.5809
    Epoch [7/50], Val Losses: mse: 69.1431, mae: 6.4092, huber: 5.9286, swd: 7.1097, ept: 44.9644
    Epoch [7/50], Test Losses: mse: 64.6070, mae: 6.0669, huber: 5.5882, swd: 7.3931, ept: 50.4290
      Epoch 7 composite train-obj: 66.716841
            No improvement (72.6980), counter 4/5
    Epoch [8/50], Train Losses: mse: 62.9181, mae: 6.0755, huber: 5.5959, swd: 7.6104, ept: 52.6329
    Epoch [8/50], Val Losses: mse: 68.0724, mae: 6.3600, huber: 5.8796, swd: 7.8910, ept: 45.6362
    Epoch [8/50], Test Losses: mse: 63.3814, mae: 6.0191, huber: 5.5404, swd: 8.1000, ept: 50.9760
      Epoch 8 composite train-obj: 66.723267
            Val objective improved 72.6316 → 72.0179, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 62.9853, mae: 6.0804, huber: 5.6010, swd: 7.6023, ept: 52.2876
    Epoch [9/50], Val Losses: mse: 68.6712, mae: 6.3876, huber: 5.9074, swd: 8.0083, ept: 44.0144
    Epoch [9/50], Test Losses: mse: 64.0433, mae: 6.0359, huber: 5.5575, swd: 7.7225, ept: 49.1039
      Epoch 9 composite train-obj: 66.786438
            No improvement (72.6754), counter 1/5
    Epoch [10/50], Train Losses: mse: 62.8488, mae: 6.0744, huber: 5.5949, swd: 7.6038, ept: 52.7312
    Epoch [10/50], Val Losses: mse: 68.6518, mae: 6.3832, huber: 5.9028, swd: 8.1638, ept: 45.3424
    Epoch [10/50], Test Losses: mse: 63.7918, mae: 6.0266, huber: 5.5485, swd: 8.0828, ept: 50.7736
      Epoch 10 composite train-obj: 66.650744
            No improvement (72.7337), counter 2/5
    Epoch [11/50], Train Losses: mse: 62.9646, mae: 6.0776, huber: 5.5982, swd: 7.5843, ept: 53.1192
    Epoch [11/50], Val Losses: mse: 68.6545, mae: 6.3973, huber: 5.9166, swd: 7.7345, ept: 46.2428
    Epoch [11/50], Test Losses: mse: 63.6125, mae: 6.0311, huber: 5.5523, swd: 7.7235, ept: 52.6809
      Epoch 11 composite train-obj: 66.756768
            No improvement (72.5218), counter 3/5
    Epoch [12/50], Train Losses: mse: 62.8434, mae: 6.0712, huber: 5.5918, swd: 7.5837, ept: 53.3338
    Epoch [12/50], Val Losses: mse: 68.0168, mae: 6.3547, huber: 5.8746, swd: 8.5559, ept: 44.7782
    Epoch [12/50], Test Losses: mse: 63.2149, mae: 6.0052, huber: 5.5272, swd: 8.6983, ept: 47.4202
      Epoch 12 composite train-obj: 66.635184
            No improvement (72.2948), counter 4/5
    Epoch [13/50], Train Losses: mse: 62.8646, mae: 6.0727, huber: 5.5932, swd: 7.6109, ept: 53.2178
    Epoch [13/50], Val Losses: mse: 68.1479, mae: 6.3708, huber: 5.8908, swd: 8.1111, ept: 46.6627
    Epoch [13/50], Test Losses: mse: 62.9826, mae: 5.9985, huber: 5.5207, swd: 8.0824, ept: 52.6041
      Epoch 13 composite train-obj: 66.670112
    Epoch [13/50], Test Losses: mse: 63.3814, mae: 6.0191, huber: 5.5404, swd: 8.1000, ept: 50.9760
    Best round's Test MSE: 63.3814, MAE: 6.0191, SWD: 8.1000
    Best round's Validation MSE: 68.0724, MAE: 6.3600, SWD: 7.8910
    Best round's Test verification MSE : 63.3814, MAE: 6.0191, SWD: 8.1000
    Time taken: 13.81 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.2006, mae: 6.6735, huber: 6.1913, swd: 9.6045, ept: 25.4044
    Epoch [1/50], Val Losses: mse: 70.1670, mae: 6.4994, huber: 6.0181, swd: 8.0919, ept: 31.1807
    Epoch [1/50], Test Losses: mse: 65.1177, mae: 6.1243, huber: 5.6451, swd: 8.2660, ept: 33.6482
      Epoch 1 composite train-obj: 81.002834
            Val objective improved inf → 74.2129, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 63.9929, mae: 6.1594, huber: 5.6788, swd: 7.9321, ept: 43.9434
    Epoch [2/50], Val Losses: mse: 69.1391, mae: 6.4255, huber: 5.9445, swd: 8.6594, ept: 38.9493
    Epoch [2/50], Test Losses: mse: 64.7451, mae: 6.0766, huber: 5.5980, swd: 8.6179, ept: 42.8330
      Epoch 2 composite train-obj: 67.958932
            Val objective improved 74.2129 → 73.4688, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 63.4377, mae: 6.1191, huber: 5.6389, swd: 7.8892, ept: 48.4872
    Epoch [3/50], Val Losses: mse: 69.1805, mae: 6.4256, huber: 5.9447, swd: 8.1344, ept: 40.5994
    Epoch [3/50], Test Losses: mse: 64.0088, mae: 6.0633, huber: 5.5847, swd: 8.2364, ept: 45.0800
      Epoch 3 composite train-obj: 67.382306
            Val objective improved 73.4688 → 73.2477, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 63.1760, mae: 6.1024, huber: 5.6223, swd: 7.8470, ept: 50.6324
    Epoch [4/50], Val Losses: mse: 68.6616, mae: 6.3942, huber: 5.9140, swd: 8.0967, ept: 40.4761
    Epoch [4/50], Test Losses: mse: 63.6180, mae: 6.0364, huber: 5.5579, swd: 8.1264, ept: 45.2258
      Epoch 4 composite train-obj: 67.099479
            Val objective improved 73.2477 → 72.7100, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 63.1023, mae: 6.0948, huber: 5.6148, swd: 7.8125, ept: 50.8808
    Epoch [5/50], Val Losses: mse: 68.7242, mae: 6.4028, huber: 5.9220, swd: 8.0058, ept: 42.8996
    Epoch [5/50], Test Losses: mse: 63.9432, mae: 6.0687, huber: 5.5899, swd: 8.5112, ept: 46.5611
      Epoch 5 composite train-obj: 67.008576
            No improvement (72.7271), counter 1/5
    Epoch [6/50], Train Losses: mse: 62.9883, mae: 6.0899, huber: 5.6101, swd: 7.7948, ept: 51.9161
    Epoch [6/50], Val Losses: mse: 67.9475, mae: 6.3615, huber: 5.8809, swd: 8.0824, ept: 43.2052
    Epoch [6/50], Test Losses: mse: 63.1871, mae: 6.0060, huber: 5.5277, swd: 8.2236, ept: 47.6167
      Epoch 6 composite train-obj: 66.885723
            Val objective improved 72.7100 → 71.9887, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 62.9993, mae: 6.0879, huber: 5.6082, swd: 7.7692, ept: 51.8486
    Epoch [7/50], Val Losses: mse: 68.8170, mae: 6.4075, huber: 5.9271, swd: 8.0771, ept: 45.3980
    Epoch [7/50], Test Losses: mse: 63.0908, mae: 6.0027, huber: 5.5245, swd: 8.4800, ept: 49.6422
      Epoch 7 composite train-obj: 66.883937
            No improvement (72.8556), counter 1/5
    Epoch [8/50], Train Losses: mse: 63.0135, mae: 6.0889, huber: 5.6092, swd: 7.7684, ept: 52.4030
    Epoch [8/50], Val Losses: mse: 68.2869, mae: 6.3761, huber: 5.8959, swd: 8.3082, ept: 43.7435
    Epoch [8/50], Test Losses: mse: 63.5939, mae: 6.0277, huber: 5.5495, swd: 8.5477, ept: 46.9892
      Epoch 8 composite train-obj: 66.897731
            No improvement (72.4410), counter 2/5
    Epoch [9/50], Train Losses: mse: 62.9290, mae: 6.0843, huber: 5.6046, swd: 7.7686, ept: 52.0583
    Epoch [9/50], Val Losses: mse: 68.6123, mae: 6.3750, huber: 5.8950, swd: 7.7085, ept: 44.3397
    Epoch [9/50], Test Losses: mse: 63.5821, mae: 6.0066, huber: 5.5286, swd: 8.1684, ept: 49.3437
      Epoch 9 composite train-obj: 66.813287
            No improvement (72.4666), counter 3/5
    Epoch [10/50], Train Losses: mse: 62.8429, mae: 6.0773, huber: 5.5977, swd: 7.7427, ept: 53.2736
    Epoch [10/50], Val Losses: mse: 69.2174, mae: 6.4286, huber: 5.9478, swd: 8.1464, ept: 41.5004
    Epoch [10/50], Test Losses: mse: 64.4900, mae: 6.0937, huber: 5.6150, swd: 8.3934, ept: 45.4905
      Epoch 10 composite train-obj: 66.714231
            No improvement (73.2906), counter 4/5
    Epoch [11/50], Train Losses: mse: 62.8269, mae: 6.0787, huber: 5.5990, swd: 7.7432, ept: 53.2592
    Epoch [11/50], Val Losses: mse: 69.3117, mae: 6.4147, huber: 5.9341, swd: 7.6961, ept: 47.7859
    Epoch [11/50], Test Losses: mse: 64.2374, mae: 6.0427, huber: 5.5644, swd: 8.0152, ept: 52.1064
      Epoch 11 composite train-obj: 66.698542
    Epoch [11/50], Test Losses: mse: 63.1871, mae: 6.0060, huber: 5.5277, swd: 8.2236, ept: 47.6167
    Best round's Test MSE: 63.1871, MAE: 6.0060, SWD: 8.2236
    Best round's Validation MSE: 67.9475, MAE: 6.3615, SWD: 8.0824
    Best round's Test verification MSE : 63.1871, MAE: 6.0060, SWD: 8.2236
    Time taken: 11.69 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.6656, mae: 6.6672, huber: 6.1850, swd: 9.7554, ept: 26.3983
    Epoch [1/50], Val Losses: mse: 69.3950, mae: 6.4650, huber: 5.9837, swd: 8.1166, ept: 33.8864
    Epoch [1/50], Test Losses: mse: 65.0017, mae: 6.1229, huber: 5.6438, swd: 8.3288, ept: 37.3944
      Epoch 1 composite train-obj: 80.543294
            Val objective improved inf → 73.4533, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 64.0513, mae: 6.1589, huber: 5.6784, swd: 7.9128, ept: 45.2639
    Epoch [2/50], Val Losses: mse: 69.0756, mae: 6.4267, huber: 5.9459, swd: 8.3442, ept: 40.5494
    Epoch [2/50], Test Losses: mse: 63.5002, mae: 6.0316, huber: 5.5530, swd: 8.5264, ept: 45.8974
      Epoch 2 composite train-obj: 68.007721
            Val objective improved 73.4533 → 73.2477, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 63.4356, mae: 6.1160, huber: 5.6358, swd: 7.8768, ept: 49.5267
    Epoch [3/50], Val Losses: mse: 68.5733, mae: 6.3787, huber: 5.8984, swd: 8.2785, ept: 41.9164
    Epoch [3/50], Test Losses: mse: 64.3916, mae: 6.0531, huber: 5.5744, swd: 8.0815, ept: 46.5082
      Epoch 3 composite train-obj: 67.374021
            Val objective improved 73.2477 → 72.7126, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 63.2149, mae: 6.0998, huber: 5.6198, swd: 7.8331, ept: 50.9504
    Epoch [4/50], Val Losses: mse: 68.6709, mae: 6.4083, huber: 5.9279, swd: 8.4525, ept: 42.9792
    Epoch [4/50], Test Losses: mse: 63.5896, mae: 6.0359, huber: 5.5573, swd: 8.5603, ept: 47.6360
      Epoch 4 composite train-obj: 67.131470
            No improvement (72.8972), counter 1/5
    Epoch [5/50], Train Losses: mse: 63.1435, mae: 6.0937, huber: 5.6138, swd: 7.7859, ept: 51.3532
    Epoch [5/50], Val Losses: mse: 68.1754, mae: 6.3812, huber: 5.9007, swd: 8.2409, ept: 44.2250
    Epoch [5/50], Test Losses: mse: 63.6851, mae: 6.0193, huber: 5.5407, swd: 8.4474, ept: 50.9722
      Epoch 5 composite train-obj: 67.036449
            Val objective improved 72.7126 → 72.2958, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 63.0932, mae: 6.0884, huber: 5.6087, swd: 7.7917, ept: 52.0500
    Epoch [6/50], Val Losses: mse: 68.5374, mae: 6.4071, huber: 5.9262, swd: 7.9784, ept: 42.1674
    Epoch [6/50], Test Losses: mse: 63.6912, mae: 6.0416, huber: 5.5629, swd: 8.1839, ept: 46.3883
      Epoch 6 composite train-obj: 66.989057
            No improvement (72.5266), counter 1/5
    Epoch [7/50], Train Losses: mse: 63.1046, mae: 6.0869, huber: 5.6072, swd: 7.7797, ept: 51.8068
    Epoch [7/50], Val Losses: mse: 68.8281, mae: 6.4012, huber: 5.9204, swd: 7.9713, ept: 45.4672
    Epoch [7/50], Test Losses: mse: 63.3006, mae: 5.9964, huber: 5.5180, swd: 8.1286, ept: 51.1712
      Epoch 7 composite train-obj: 66.994481
            No improvement (72.8138), counter 2/5
    Epoch [8/50], Train Losses: mse: 62.9948, mae: 6.0840, huber: 5.6043, swd: 7.7572, ept: 52.2141
    Epoch [8/50], Val Losses: mse: 68.5703, mae: 6.3874, huber: 5.9072, swd: 7.8755, ept: 43.3265
    Epoch [8/50], Test Losses: mse: 63.6736, mae: 6.0283, huber: 5.5500, swd: 8.0786, ept: 47.7003
      Epoch 8 composite train-obj: 66.873389
            No improvement (72.5081), counter 3/5
    Epoch [9/50], Train Losses: mse: 62.9920, mae: 6.0803, huber: 5.6007, swd: 7.7411, ept: 52.3354
    Epoch [9/50], Val Losses: mse: 68.0913, mae: 6.3589, huber: 5.8785, swd: 8.2519, ept: 45.5644
    Epoch [9/50], Test Losses: mse: 64.1324, mae: 6.0461, huber: 5.5679, swd: 8.4155, ept: 49.2157
      Epoch 9 composite train-obj: 66.862521
            Val objective improved 72.2958 → 72.2172, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 62.8183, mae: 6.0698, huber: 5.5902, swd: 7.7434, ept: 53.3873
    Epoch [10/50], Val Losses: mse: 68.1447, mae: 6.3669, huber: 5.8866, swd: 8.5189, ept: 42.0310
    Epoch [10/50], Test Losses: mse: 63.0613, mae: 5.9975, huber: 5.5191, swd: 8.4219, ept: 44.4547
      Epoch 10 composite train-obj: 66.689991
            No improvement (72.4041), counter 1/5
    Epoch [11/50], Train Losses: mse: 62.9186, mae: 6.0755, huber: 5.5960, swd: 7.7336, ept: 52.8793
    Epoch [11/50], Val Losses: mse: 67.7704, mae: 6.3630, huber: 5.8822, swd: 8.1791, ept: 46.2062
    Epoch [11/50], Test Losses: mse: 63.5199, mae: 6.0232, huber: 5.5448, swd: 8.3815, ept: 50.7798
      Epoch 11 composite train-obj: 66.785388
            Val objective improved 72.2172 → 71.8600, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 62.8791, mae: 6.0733, huber: 5.5938, swd: 7.7287, ept: 52.7067
    Epoch [12/50], Val Losses: mse: 68.3302, mae: 6.3802, huber: 5.8997, swd: 8.4687, ept: 44.5945
    Epoch [12/50], Test Losses: mse: 62.6228, mae: 5.9726, huber: 5.4946, swd: 8.6688, ept: 49.8392
      Epoch 12 composite train-obj: 66.743474
            No improvement (72.5645), counter 1/5
    Epoch [13/50], Train Losses: mse: 62.9688, mae: 6.0804, huber: 5.6009, swd: 7.7457, ept: 53.1010
    Epoch [13/50], Val Losses: mse: 68.7809, mae: 6.3841, huber: 5.9039, swd: 8.2198, ept: 45.7558
    Epoch [13/50], Test Losses: mse: 63.8694, mae: 6.0332, huber: 5.5550, swd: 8.4526, ept: 50.0525
      Epoch 13 composite train-obj: 66.841622
            No improvement (72.8908), counter 2/5
    Epoch [14/50], Train Losses: mse: 62.7990, mae: 6.0689, huber: 5.5894, swd: 7.7370, ept: 53.3339
    Epoch [14/50], Val Losses: mse: 69.1729, mae: 6.4204, huber: 5.9398, swd: 7.5730, ept: 45.2290
    Epoch [14/50], Test Losses: mse: 63.7029, mae: 6.0113, huber: 5.5328, swd: 7.5847, ept: 51.0110
      Epoch 14 composite train-obj: 66.667497
            No improvement (72.9594), counter 3/5
    Epoch [15/50], Train Losses: mse: 62.8017, mae: 6.0663, huber: 5.5868, swd: 7.7244, ept: 53.7404
    Epoch [15/50], Val Losses: mse: 68.0400, mae: 6.3385, huber: 5.8583, swd: 8.5082, ept: 45.0397
    Epoch [15/50], Test Losses: mse: 63.1802, mae: 5.9488, huber: 5.4721, swd: 8.5757, ept: 50.5334
      Epoch 15 composite train-obj: 66.663934
            No improvement (72.2941), counter 4/5
    Epoch [16/50], Train Losses: mse: 62.8255, mae: 6.0700, huber: 5.5905, swd: 7.7200, ept: 53.3619
    Epoch [16/50], Val Losses: mse: 68.2063, mae: 6.3696, huber: 5.8886, swd: 7.6813, ept: 46.1728
    Epoch [16/50], Test Losses: mse: 63.6834, mae: 6.0198, huber: 5.5415, swd: 7.8181, ept: 51.5581
      Epoch 16 composite train-obj: 66.685454
    Epoch [16/50], Test Losses: mse: 63.5199, mae: 6.0232, huber: 5.5448, swd: 8.3815, ept: 50.7798
    Best round's Test MSE: 63.5199, MAE: 6.0232, SWD: 8.3815
    Best round's Validation MSE: 67.7704, MAE: 6.3630, SWD: 8.1791
    Best round's Test verification MSE : 63.5199, MAE: 6.0232, SWD: 8.3815
    Time taken: 17.26 seconds
    
    ==================================================
    Experiment Summary (DLinear_lorenz_seq336_pred336_20250514_2140)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 63.3628 ± 0.1365
      mae: 6.0161 ± 0.0073
      huber: 5.5377 ± 0.0073
      swd: 8.2350 ± 0.1152
      ept: 49.7908 ± 1.5394
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 67.9301 ± 0.1239
      mae: 6.3615 ± 0.0012
      huber: 5.8809 ± 0.0011
      swd: 8.0508 ± 0.1197
      ept: 45.0159 ± 1.3013
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 42.80 seconds
    
    Experiment complete: DLinear_lorenz_seq336_pred336_20250514_2140
    Model: DLinear
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    
