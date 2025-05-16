## AB: OT Components

### AB: One Encoding Layer (x->z, not x->z->x)


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_single_encoding_layer=True, ### HERE
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
    
    Epoch [1/50], Train Losses: mse: 74.7254, mae: 6.5307, huber: 6.0506, swd: 40.5665, ept: 32.1463
    Epoch [1/50], Val Losses: mse: 60.5110, mae: 5.9117, huber: 5.4344, swd: 28.4591, ept: 36.3559
    Epoch [1/50], Test Losses: mse: 55.5920, mae: 5.4952, huber: 5.0219, swd: 27.6634, ept: 40.4119
      Epoch 1 composite train-obj: 6.050616
            Val objective improved inf → 5.4344, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 50.3129, mae: 5.0466, huber: 4.5776, swd: 19.0239, ept: 83.6756
    Epoch [2/50], Val Losses: mse: 54.8847, mae: 5.3319, huber: 4.8612, swd: 18.4634, ept: 76.8741
    Epoch [2/50], Test Losses: mse: 49.3584, mae: 4.9152, huber: 4.4485, swd: 18.8353, ept: 75.9148
      Epoch 2 composite train-obj: 4.577607
            Val objective improved 5.4344 → 4.8612, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 44.1919, mae: 4.4788, huber: 4.0184, swd: 12.7215, ept: 120.5075
    Epoch [3/50], Val Losses: mse: 53.2478, mae: 5.0754, huber: 4.6087, swd: 13.2134, ept: 97.8582
    Epoch [3/50], Test Losses: mse: 46.0031, mae: 4.6011, huber: 4.1377, swd: 13.4329, ept: 96.5982
      Epoch 3 composite train-obj: 4.018352
            Val objective improved 4.8612 → 4.6087, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.3577, mae: 4.0784, huber: 3.6238, swd: 8.4042, ept: 140.8430
    Epoch [4/50], Val Losses: mse: 52.3236, mae: 4.9335, huber: 4.4690, swd: 9.8144, ept: 102.0240
    Epoch [4/50], Test Losses: mse: 44.2731, mae: 4.4370, huber: 3.9758, swd: 9.2833, ept: 102.8718
      Epoch 4 composite train-obj: 3.623814
            Val objective improved 4.6087 → 4.4690, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 35.9280, mae: 3.7733, huber: 3.3242, swd: 6.1613, ept: 156.6037
    Epoch [5/50], Val Losses: mse: 52.6511, mae: 4.8370, huber: 4.3753, swd: 8.4473, ept: 115.3838
    Epoch [5/50], Test Losses: mse: 42.9106, mae: 4.2366, huber: 3.7794, swd: 8.2469, ept: 119.4033
      Epoch 5 composite train-obj: 3.324199
            Val objective improved 4.4690 → 4.3753, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 33.0384, mae: 3.5383, huber: 3.0937, swd: 4.9497, ept: 168.6097
    Epoch [6/50], Val Losses: mse: 52.6753, mae: 4.7476, huber: 4.2867, swd: 6.7938, ept: 119.3349
    Epoch [6/50], Test Losses: mse: 42.7379, mae: 4.1787, huber: 3.7223, swd: 6.5782, ept: 125.2062
      Epoch 6 composite train-obj: 3.093690
            Val objective improved 4.3753 → 4.2867, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 30.4622, mae: 3.3222, huber: 2.8824, swd: 4.0717, ept: 179.5192
    Epoch [7/50], Val Losses: mse: 54.5839, mae: 4.7678, huber: 4.3089, swd: 6.1281, ept: 126.4562
    Epoch [7/50], Test Losses: mse: 41.7156, mae: 4.0374, huber: 3.5849, swd: 5.4468, ept: 136.2798
      Epoch 7 composite train-obj: 2.882425
            No improvement (4.3089), counter 1/5
    Epoch [8/50], Train Losses: mse: 28.4389, mae: 3.1675, huber: 2.7305, swd: 3.4295, ept: 186.9289
    Epoch [8/50], Val Losses: mse: 53.3746, mae: 4.6529, huber: 4.1971, swd: 5.8780, ept: 135.8997
    Epoch [8/50], Test Losses: mse: 40.9860, mae: 3.9532, huber: 3.5028, swd: 4.7637, ept: 154.5554
      Epoch 8 composite train-obj: 2.730493
            Val objective improved 4.2867 → 4.1971, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 26.4404, mae: 3.0076, huber: 2.5746, swd: 3.0012, ept: 194.7353
    Epoch [9/50], Val Losses: mse: 54.6834, mae: 4.6892, huber: 4.2338, swd: 5.3713, ept: 135.9778
    Epoch [9/50], Test Losses: mse: 40.6011, mae: 3.8713, huber: 3.4229, swd: 3.9103, ept: 153.0187
      Epoch 9 composite train-obj: 2.574584
            No improvement (4.2338), counter 1/5
    Epoch [10/50], Train Losses: mse: 25.3228, mae: 2.9186, huber: 2.4875, swd: 2.7259, ept: 200.1883
    Epoch [10/50], Val Losses: mse: 57.9289, mae: 4.8082, huber: 4.3521, swd: 4.9446, ept: 138.4892
    Epoch [10/50], Test Losses: mse: 42.8581, mae: 3.9469, huber: 3.4989, swd: 3.5311, ept: 157.8430
      Epoch 10 composite train-obj: 2.487455
            No improvement (4.3521), counter 2/5
    Epoch [11/50], Train Losses: mse: 23.4405, mae: 2.7646, huber: 2.3380, swd: 2.4128, ept: 207.7916
    Epoch [11/50], Val Losses: mse: 54.0831, mae: 4.6091, huber: 4.1541, swd: 4.5631, ept: 139.9486
    Epoch [11/50], Test Losses: mse: 40.0242, mae: 3.8106, huber: 3.3623, swd: 3.3105, ept: 158.3474
      Epoch 11 composite train-obj: 2.338005
            Val objective improved 4.1971 → 4.1541, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 22.6764, mae: 2.6926, huber: 2.2686, swd: 2.2484, ept: 212.9437
    Epoch [12/50], Val Losses: mse: 53.7650, mae: 4.5163, huber: 4.0658, swd: 4.9076, ept: 147.3460
    Epoch [12/50], Test Losses: mse: 38.9463, mae: 3.6250, huber: 3.1845, swd: 3.2312, ept: 169.4531
      Epoch 12 composite train-obj: 2.268580
            Val objective improved 4.1541 → 4.0658, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.9309, mae: 2.5391, huber: 2.1203, swd: 1.9868, ept: 221.6422
    Epoch [13/50], Val Losses: mse: 53.4066, mae: 4.4946, huber: 4.0453, swd: 4.0062, ept: 153.0964
    Epoch [13/50], Test Losses: mse: 39.4967, mae: 3.6350, huber: 3.1965, swd: 3.0273, ept: 172.7724
      Epoch 13 composite train-obj: 2.120336
            Val objective improved 4.0658 → 4.0453, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 20.3745, mae: 2.4929, huber: 2.0752, swd: 1.8970, ept: 225.5563
    Epoch [14/50], Val Losses: mse: 54.1973, mae: 4.5751, huber: 4.1241, swd: 4.5186, ept: 146.3283
    Epoch [14/50], Test Losses: mse: 40.4160, mae: 3.7532, huber: 3.3114, swd: 3.5468, ept: 167.2567
      Epoch 14 composite train-obj: 2.075215
            No improvement (4.1241), counter 1/5
    Epoch [15/50], Train Losses: mse: 20.1211, mae: 2.4686, huber: 2.0519, swd: 1.8265, ept: 227.1655
    Epoch [15/50], Val Losses: mse: 53.7021, mae: 4.4432, huber: 3.9957, swd: 4.2274, ept: 155.3878
    Epoch [15/50], Test Losses: mse: 39.0376, mae: 3.6225, huber: 3.1835, swd: 2.9714, ept: 175.7783
      Epoch 15 composite train-obj: 2.051905
            Val objective improved 4.0453 → 3.9957, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 19.0224, mae: 2.3668, huber: 1.9545, swd: 1.7069, ept: 233.1993
    Epoch [16/50], Val Losses: mse: 54.4366, mae: 4.4735, huber: 4.0249, swd: 4.4555, ept: 157.6234
    Epoch [16/50], Test Losses: mse: 37.2686, mae: 3.4474, huber: 3.0120, swd: 2.8591, ept: 184.7741
      Epoch 16 composite train-obj: 1.954487
            No improvement (4.0249), counter 1/5
    Epoch [17/50], Train Losses: mse: 17.9009, mae: 2.2682, huber: 1.8593, swd: 1.5575, ept: 238.9776
    Epoch [17/50], Val Losses: mse: 50.6939, mae: 4.2735, huber: 3.8291, swd: 4.0806, ept: 162.5491
    Epoch [17/50], Test Losses: mse: 38.6717, mae: 3.4919, huber: 3.0590, swd: 2.9216, ept: 187.1010
      Epoch 17 composite train-obj: 1.859331
            Val objective improved 3.9957 → 3.8291, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 17.5243, mae: 2.2367, huber: 1.8290, swd: 1.5160, ept: 241.2555
    Epoch [18/50], Val Losses: mse: 53.8862, mae: 4.3926, huber: 3.9469, swd: 3.8604, ept: 161.9788
    Epoch [18/50], Test Losses: mse: 37.2401, mae: 3.4472, huber: 3.0134, swd: 2.5373, ept: 188.2016
      Epoch 18 composite train-obj: 1.829027
            No improvement (3.9469), counter 1/5
    Epoch [19/50], Train Losses: mse: 17.1867, mae: 2.1983, huber: 1.7928, swd: 1.4172, ept: 243.1521
    Epoch [19/50], Val Losses: mse: 51.1037, mae: 4.3401, huber: 3.8906, swd: 3.7974, ept: 160.7215
    Epoch [19/50], Test Losses: mse: 39.5327, mae: 3.6134, huber: 3.1729, swd: 2.8321, ept: 182.4015
      Epoch 19 composite train-obj: 1.792847
            No improvement (3.8906), counter 2/5
    Epoch [20/50], Train Losses: mse: 17.2401, mae: 2.1868, huber: 1.7821, swd: 1.4209, ept: 245.3340
    Epoch [20/50], Val Losses: mse: 49.7265, mae: 4.1759, huber: 3.7330, swd: 3.6308, ept: 168.6474
    Epoch [20/50], Test Losses: mse: 37.4699, mae: 3.4230, huber: 2.9912, swd: 2.7966, ept: 190.8848
      Epoch 20 composite train-obj: 1.782077
            Val objective improved 3.8291 → 3.7330, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 16.4941, mae: 2.1248, huber: 1.7225, swd: 1.3447, ept: 248.9908
    Epoch [21/50], Val Losses: mse: 52.2641, mae: 4.3904, huber: 3.9419, swd: 4.1015, ept: 154.3575
    Epoch [21/50], Test Losses: mse: 40.2343, mae: 3.6194, huber: 3.1817, swd: 2.9211, ept: 181.5227
      Epoch 21 composite train-obj: 1.722474
            No improvement (3.9419), counter 1/5
    Epoch [22/50], Train Losses: mse: 16.0623, mae: 2.0849, huber: 1.6841, swd: 1.2886, ept: 250.5347
    Epoch [22/50], Val Losses: mse: 52.7458, mae: 4.3330, huber: 3.8888, swd: 3.7616, ept: 167.0257
    Epoch [22/50], Test Losses: mse: 37.7772, mae: 3.4005, huber: 2.9697, swd: 2.4596, ept: 199.4375
      Epoch 22 composite train-obj: 1.684068
            No improvement (3.8888), counter 2/5
    Epoch [23/50], Train Losses: mse: 15.4320, mae: 2.0345, huber: 1.6360, swd: 1.2295, ept: 253.0848
    Epoch [23/50], Val Losses: mse: 48.0378, mae: 4.1711, huber: 3.7226, swd: 3.6341, ept: 164.9907
    Epoch [23/50], Test Losses: mse: 37.9349, mae: 3.4660, huber: 3.0296, swd: 2.6638, ept: 193.7268
      Epoch 23 composite train-obj: 1.635992
            Val objective improved 3.7330 → 3.7226, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 15.0693, mae: 2.0034, huber: 1.6060, swd: 1.1980, ept: 254.8298
    Epoch [24/50], Val Losses: mse: 54.4630, mae: 4.4129, huber: 3.9666, swd: 3.6000, ept: 159.4810
    Epoch [24/50], Test Losses: mse: 37.7088, mae: 3.4287, huber: 2.9952, swd: 2.2594, ept: 190.6238
      Epoch 24 composite train-obj: 1.606041
            No improvement (3.9666), counter 1/5
    Epoch [25/50], Train Losses: mse: 15.4679, mae: 2.0176, huber: 1.6205, swd: 1.2165, ept: 255.0069
    Epoch [25/50], Val Losses: mse: 49.0989, mae: 4.2177, huber: 3.7722, swd: 3.4225, ept: 163.2915
    Epoch [25/50], Test Losses: mse: 39.1150, mae: 3.5025, huber: 3.0692, swd: 2.3520, ept: 190.1658
      Epoch 25 composite train-obj: 1.620545
            No improvement (3.7722), counter 2/5
    Epoch [26/50], Train Losses: mse: 14.6281, mae: 1.9518, huber: 1.5575, swd: 1.1304, ept: 257.1164
    Epoch [26/50], Val Losses: mse: 49.5622, mae: 4.1538, huber: 3.7143, swd: 3.6091, ept: 168.3783
    Epoch [26/50], Test Losses: mse: 35.2729, mae: 3.2448, huber: 2.8203, swd: 2.3585, ept: 204.8266
      Epoch 26 composite train-obj: 1.557550
            Val objective improved 3.7226 → 3.7143, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 14.1401, mae: 1.9048, huber: 1.5129, swd: 1.0851, ept: 260.2368
    Epoch [27/50], Val Losses: mse: 50.2881, mae: 4.1851, huber: 3.7430, swd: 3.4681, ept: 167.4193
    Epoch [27/50], Test Losses: mse: 38.8730, mae: 3.4501, huber: 3.0194, swd: 2.0992, ept: 196.6081
      Epoch 27 composite train-obj: 1.512903
            No improvement (3.7430), counter 1/5
    Epoch [28/50], Train Losses: mse: 14.5833, mae: 1.9307, huber: 1.5376, swd: 1.1146, ept: 259.9540
    Epoch [28/50], Val Losses: mse: 48.4512, mae: 4.1151, huber: 3.6739, swd: 3.2486, ept: 169.5743
    Epoch [28/50], Test Losses: mse: 36.3742, mae: 3.3194, huber: 2.8918, swd: 2.3308, ept: 202.7940
      Epoch 28 composite train-obj: 1.537601
            Val objective improved 3.7143 → 3.6739, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 13.6613, mae: 1.8495, huber: 1.4606, swd: 1.0605, ept: 263.7612
    Epoch [29/50], Val Losses: mse: 51.9190, mae: 4.2739, huber: 3.8319, swd: 3.3791, ept: 166.7085
    Epoch [29/50], Test Losses: mse: 38.7900, mae: 3.4588, huber: 3.0289, swd: 2.2830, ept: 195.8057
      Epoch 29 composite train-obj: 1.460589
            No improvement (3.8319), counter 1/5
    Epoch [30/50], Train Losses: mse: 14.3381, mae: 1.9104, huber: 1.5181, swd: 1.0906, ept: 261.3128
    Epoch [30/50], Val Losses: mse: 50.5770, mae: 4.1600, huber: 3.7216, swd: 3.3763, ept: 175.9209
    Epoch [30/50], Test Losses: mse: 37.3798, mae: 3.3147, huber: 2.8898, swd: 2.3937, ept: 208.2387
      Epoch 30 composite train-obj: 1.518065
            No improvement (3.7216), counter 2/5
    Epoch [31/50], Train Losses: mse: 14.2549, mae: 1.8878, huber: 1.4978, swd: 1.0597, ept: 262.1899
    Epoch [31/50], Val Losses: mse: 52.0755, mae: 4.2608, huber: 3.8203, swd: 3.2176, ept: 168.7328
    Epoch [31/50], Test Losses: mse: 40.1952, mae: 3.4796, huber: 3.0506, swd: 2.2788, ept: 200.5887
      Epoch 31 composite train-obj: 1.497753
            No improvement (3.8203), counter 3/5
    Epoch [32/50], Train Losses: mse: 13.4330, mae: 1.8110, huber: 1.4259, swd: 0.9824, ept: 265.7864
    Epoch [32/50], Val Losses: mse: 52.9318, mae: 4.3122, huber: 3.8721, swd: 3.4132, ept: 167.6508
    Epoch [32/50], Test Losses: mse: 37.0273, mae: 3.3341, huber: 2.9085, swd: 2.3770, ept: 202.3024
      Epoch 32 composite train-obj: 1.425939
            No improvement (3.8721), counter 4/5
    Epoch [33/50], Train Losses: mse: 12.7143, mae: 1.7595, huber: 1.3751, swd: 0.9353, ept: 268.3660
    Epoch [33/50], Val Losses: mse: 46.5352, mae: 3.9905, huber: 3.5545, swd: 3.1856, ept: 169.5838
    Epoch [33/50], Test Losses: mse: 34.2053, mae: 3.1712, huber: 2.7495, swd: 1.9865, ept: 206.3938
      Epoch 33 composite train-obj: 1.375115
            Val objective improved 3.6739 → 3.5545, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 13.0045, mae: 1.7624, huber: 1.3795, swd: 0.9284, ept: 268.3199
    Epoch [34/50], Val Losses: mse: 49.5225, mae: 4.1219, huber: 3.6830, swd: 3.1201, ept: 172.0855
    Epoch [34/50], Test Losses: mse: 37.0987, mae: 3.3261, huber: 2.9019, swd: 2.1359, ept: 205.1520
      Epoch 34 composite train-obj: 1.379538
            No improvement (3.6830), counter 1/5
    Epoch [35/50], Train Losses: mse: 13.0185, mae: 1.7714, huber: 1.3879, swd: 0.9652, ept: 268.0384
    Epoch [35/50], Val Losses: mse: 50.7322, mae: 4.2094, huber: 3.7659, swd: 2.9152, ept: 167.3430
    Epoch [35/50], Test Losses: mse: 34.6220, mae: 3.1946, huber: 2.7682, swd: 2.0490, ept: 208.0175
      Epoch 35 composite train-obj: 1.387910
            No improvement (3.7659), counter 2/5
    Epoch [36/50], Train Losses: mse: 13.0434, mae: 1.7708, huber: 1.3865, swd: 0.9573, ept: 268.9723
    Epoch [36/50], Val Losses: mse: 51.8495, mae: 4.2301, huber: 3.7898, swd: 3.0501, ept: 167.9581
    Epoch [36/50], Test Losses: mse: 37.6236, mae: 3.3030, huber: 2.8786, swd: 2.0990, ept: 209.2408
      Epoch 36 composite train-obj: 1.386532
            No improvement (3.7898), counter 3/5
    Epoch [37/50], Train Losses: mse: 12.2237, mae: 1.7009, huber: 1.3213, swd: 0.9088, ept: 270.7775
    Epoch [37/50], Val Losses: mse: 48.5895, mae: 4.0824, huber: 3.6470, swd: 3.1364, ept: 172.1349
    Epoch [37/50], Test Losses: mse: 35.4375, mae: 3.2639, huber: 2.8417, swd: 2.3657, ept: 202.9394
      Epoch 37 composite train-obj: 1.321312
            No improvement (3.6470), counter 4/5
    Epoch [38/50], Train Losses: mse: 12.1925, mae: 1.6831, huber: 1.3056, swd: 0.8907, ept: 271.9485
    Epoch [38/50], Val Losses: mse: 51.2151, mae: 4.2216, huber: 3.7811, swd: 3.1400, ept: 171.6792
    Epoch [38/50], Test Losses: mse: 33.8991, mae: 3.1690, huber: 2.7445, swd: 2.1694, ept: 208.6064
      Epoch 38 composite train-obj: 1.305619
    Epoch [38/50], Test Losses: mse: 34.1791, mae: 3.1700, huber: 2.7483, swd: 1.9879, ept: 206.3347
    Best round's Test MSE: 34.2053, MAE: 3.1712, SWD: 1.9865
    Best round's Validation MSE: 46.5352, MAE: 3.9905, SWD: 3.1856
    Best round's Test verification MSE : 34.1791, MAE: 3.1700, SWD: 1.9879
    Time taken: 115.85 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 72.4269, mae: 6.3488, huber: 5.8701, swd: 40.0765, ept: 36.6949
    Epoch [1/50], Val Losses: mse: 58.8994, mae: 5.7411, huber: 5.2648, swd: 25.7344, ept: 46.9326
    Epoch [1/50], Test Losses: mse: 53.7553, mae: 5.3198, huber: 4.8477, swd: 26.6260, ept: 47.5495
      Epoch 1 composite train-obj: 5.870078
            Val objective improved inf → 5.2648, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.7912, mae: 4.9066, huber: 4.4396, swd: 17.6111, ept: 89.8845
    Epoch [2/50], Val Losses: mse: 55.0564, mae: 5.2749, huber: 4.8058, swd: 16.4514, ept: 79.2601
    Epoch [2/50], Test Losses: mse: 48.7406, mae: 4.8137, huber: 4.3488, swd: 17.3793, ept: 78.4491
      Epoch 2 composite train-obj: 4.439594
            Val objective improved 5.2648 → 4.8058, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.1672, mae: 4.3114, huber: 3.8532, swd: 10.6137, ept: 123.0730
    Epoch [3/50], Val Losses: mse: 53.4559, mae: 5.0758, huber: 4.6092, swd: 11.6253, ept: 91.7271
    Epoch [3/50], Test Losses: mse: 45.4675, mae: 4.5480, huber: 4.0853, swd: 12.1164, ept: 88.7256
      Epoch 3 composite train-obj: 3.853244
            Val objective improved 4.8058 → 4.6092, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 37.6802, mae: 3.9252, huber: 3.4731, swd: 7.0673, ept: 144.5648
    Epoch [4/50], Val Losses: mse: 55.2830, mae: 4.9797, huber: 4.5173, swd: 8.4331, ept: 106.4142
    Epoch [4/50], Test Losses: mse: 44.6095, mae: 4.3291, huber: 3.8718, swd: 8.3625, ept: 111.3917
      Epoch 4 composite train-obj: 3.473055
            Val objective improved 4.6092 → 4.5173, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.1273, mae: 3.6319, huber: 3.1851, swd: 5.3047, ept: 162.5784
    Epoch [5/50], Val Losses: mse: 55.1328, mae: 4.9010, huber: 4.4403, swd: 7.2687, ept: 119.9394
    Epoch [5/50], Test Losses: mse: 42.9715, mae: 4.1925, huber: 3.7371, swd: 7.4971, ept: 128.4775
      Epoch 5 composite train-obj: 3.185098
            Val objective improved 4.5173 → 4.4403, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.4635, mae: 3.4079, huber: 2.9658, swd: 4.2627, ept: 174.4995
    Epoch [6/50], Val Losses: mse: 54.3438, mae: 4.8200, huber: 4.3590, swd: 7.2324, ept: 126.0033
    Epoch [6/50], Test Losses: mse: 42.1455, mae: 4.1294, huber: 3.6739, swd: 5.7356, ept: 138.2144
      Epoch 6 composite train-obj: 2.965806
            Val objective improved 4.4403 → 4.3590, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.9202, mae: 3.2120, huber: 2.7734, swd: 3.5068, ept: 185.0870
    Epoch [7/50], Val Losses: mse: 54.0198, mae: 4.7309, huber: 4.2737, swd: 6.0380, ept: 131.8548
    Epoch [7/50], Test Losses: mse: 41.9226, mae: 4.0217, huber: 3.5707, swd: 5.5844, ept: 143.9928
      Epoch 7 composite train-obj: 2.773397
            Val objective improved 4.3590 → 4.2737, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.9506, mae: 3.0541, huber: 2.6198, swd: 3.0237, ept: 192.4016
    Epoch [8/50], Val Losses: mse: 54.6307, mae: 4.7370, huber: 4.2797, swd: 5.5242, ept: 132.0203
    Epoch [8/50], Test Losses: mse: 42.6045, mae: 3.9897, huber: 3.5401, swd: 4.4191, ept: 147.3046
      Epoch 8 composite train-obj: 2.619829
            No improvement (4.2797), counter 1/5
    Epoch [9/50], Train Losses: mse: 25.6327, mae: 2.9396, huber: 2.5083, swd: 2.7149, ept: 199.5735
    Epoch [9/50], Val Losses: mse: 55.5403, mae: 4.7206, huber: 4.2663, swd: 5.4121, ept: 136.6493
    Epoch [9/50], Test Losses: mse: 41.4048, mae: 3.8417, huber: 3.3963, swd: 4.1338, ept: 155.6699
      Epoch 9 composite train-obj: 2.508322
            Val objective improved 4.2737 → 4.2663, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 24.0244, mae: 2.8119, huber: 2.3838, swd: 2.4255, ept: 206.6988
    Epoch [10/50], Val Losses: mse: 55.4090, mae: 4.6923, huber: 4.2391, swd: 5.9243, ept: 137.0622
    Epoch [10/50], Test Losses: mse: 44.4370, mae: 3.9633, huber: 3.5182, swd: 4.2867, ept: 161.8619
      Epoch 10 composite train-obj: 2.383767
            Val objective improved 4.2663 → 4.2391, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.7758, mae: 2.6867, huber: 2.2635, swd: 2.2151, ept: 214.3666
    Epoch [11/50], Val Losses: mse: 57.6608, mae: 4.7842, huber: 4.3280, swd: 5.2096, ept: 138.3793
    Epoch [11/50], Test Losses: mse: 41.6295, mae: 3.8343, huber: 3.3883, swd: 4.1773, ept: 166.3927
      Epoch 11 composite train-obj: 2.263506
            No improvement (4.3280), counter 1/5
    Epoch [12/50], Train Losses: mse: 22.3768, mae: 2.6545, huber: 2.2322, swd: 2.1604, ept: 216.7178
    Epoch [12/50], Val Losses: mse: 54.6662, mae: 4.6400, huber: 4.1852, swd: 4.6499, ept: 140.7812
    Epoch [12/50], Test Losses: mse: 39.9707, mae: 3.7379, huber: 3.2948, swd: 4.1467, ept: 164.4100
      Epoch 12 composite train-obj: 2.232231
            Val objective improved 4.2391 → 4.1852, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.9691, mae: 2.5363, huber: 2.1177, swd: 1.9486, ept: 222.9445
    Epoch [13/50], Val Losses: mse: 53.9775, mae: 4.5987, huber: 4.1435, swd: 4.2434, ept: 150.8180
    Epoch [13/50], Test Losses: mse: 39.6394, mae: 3.6929, huber: 3.2492, swd: 3.5892, ept: 175.1272
      Epoch 13 composite train-obj: 2.117722
            Val objective improved 4.1852 → 4.1435, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 21.0479, mae: 2.5355, huber: 2.1175, swd: 1.9309, ept: 224.3056
    Epoch [14/50], Val Losses: mse: 52.3840, mae: 4.4630, huber: 4.0127, swd: 4.7484, ept: 150.0622
    Epoch [14/50], Test Losses: mse: 40.3955, mae: 3.6846, huber: 3.2448, swd: 3.6337, ept: 174.0533
      Epoch 14 composite train-obj: 2.117518
            Val objective improved 4.1435 → 4.0127, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 19.6986, mae: 2.4034, huber: 1.9910, swd: 1.7401, ept: 231.3920
    Epoch [15/50], Val Losses: mse: 54.1421, mae: 4.5289, huber: 4.0803, swd: 4.4349, ept: 149.3583
    Epoch [15/50], Test Losses: mse: 43.0179, mae: 3.7866, huber: 3.3483, swd: 3.2024, ept: 175.8112
      Epoch 15 composite train-obj: 1.990990
            No improvement (4.0803), counter 1/5
    Epoch [16/50], Train Losses: mse: 18.6354, mae: 2.3139, huber: 1.9047, swd: 1.6210, ept: 236.2353
    Epoch [16/50], Val Losses: mse: 53.6194, mae: 4.5006, huber: 4.0485, swd: 4.8030, ept: 148.7709
    Epoch [16/50], Test Losses: mse: 38.8567, mae: 3.6082, huber: 3.1667, swd: 3.0947, ept: 176.3432
      Epoch 16 composite train-obj: 1.904716
            No improvement (4.0485), counter 2/5
    Epoch [17/50], Train Losses: mse: 18.4796, mae: 2.3006, huber: 1.8917, swd: 1.5821, ept: 237.6052
    Epoch [17/50], Val Losses: mse: 52.5254, mae: 4.4054, huber: 3.9540, swd: 3.9292, ept: 157.4030
    Epoch [17/50], Test Losses: mse: 39.2499, mae: 3.6137, huber: 3.1730, swd: 3.0918, ept: 180.4595
      Epoch 17 composite train-obj: 1.891712
            Val objective improved 4.0127 → 3.9540, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 17.5847, mae: 2.2225, huber: 1.8167, swd: 1.4993, ept: 241.4101
    Epoch [18/50], Val Losses: mse: 52.5935, mae: 4.4146, huber: 3.9674, swd: 4.6586, ept: 154.8241
    Epoch [18/50], Test Losses: mse: 37.2442, mae: 3.4658, huber: 3.0325, swd: 2.9584, ept: 184.6154
      Epoch 18 composite train-obj: 1.816725
            No improvement (3.9674), counter 1/5
    Epoch [19/50], Train Losses: mse: 17.6296, mae: 2.2201, huber: 1.8147, swd: 1.4751, ept: 241.9119
    Epoch [19/50], Val Losses: mse: 50.6691, mae: 4.3279, huber: 3.8812, swd: 4.6711, ept: 157.5732
    Epoch [19/50], Test Losses: mse: 38.4453, mae: 3.5198, huber: 3.0849, swd: 2.9780, ept: 186.0155
      Epoch 19 composite train-obj: 1.814691
            Val objective improved 3.9540 → 3.8812, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 17.2492, mae: 2.1789, huber: 1.7750, swd: 1.3923, ept: 244.8680
    Epoch [20/50], Val Losses: mse: 53.2093, mae: 4.4613, huber: 4.0135, swd: 4.4409, ept: 155.1470
    Epoch [20/50], Test Losses: mse: 37.9626, mae: 3.5195, huber: 3.0848, swd: 2.9739, ept: 184.1010
      Epoch 20 composite train-obj: 1.775007
            No improvement (4.0135), counter 1/5
    Epoch [21/50], Train Losses: mse: 17.1044, mae: 2.1568, huber: 1.7547, swd: 1.4107, ept: 246.0524
    Epoch [21/50], Val Losses: mse: 51.2656, mae: 4.2882, huber: 3.8441, swd: 3.7230, ept: 162.0740
    Epoch [21/50], Test Losses: mse: 37.1267, mae: 3.4143, huber: 2.9826, swd: 2.7739, ept: 193.1105
      Epoch 21 composite train-obj: 1.754658
            Val objective improved 3.8812 → 3.8441, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 16.7723, mae: 2.1246, huber: 1.7234, swd: 1.3628, ept: 248.8773
    Epoch [22/50], Val Losses: mse: 51.0170, mae: 4.2674, huber: 3.8238, swd: 3.8601, ept: 165.1172
    Epoch [22/50], Test Losses: mse: 37.5636, mae: 3.4068, huber: 2.9754, swd: 2.5299, ept: 192.5977
      Epoch 22 composite train-obj: 1.723369
            Val objective improved 3.8441 → 3.8238, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 16.2730, mae: 2.0691, huber: 1.6715, swd: 1.2866, ept: 251.4926
    Epoch [23/50], Val Losses: mse: 52.7109, mae: 4.3631, huber: 3.9162, swd: 3.5086, ept: 160.5188
    Epoch [23/50], Test Losses: mse: 35.8200, mae: 3.4069, huber: 2.9728, swd: 2.7677, ept: 188.7690
      Epoch 23 composite train-obj: 1.671528
            No improvement (3.9162), counter 1/5
    Epoch [24/50], Train Losses: mse: 15.0808, mae: 1.9901, huber: 1.5946, swd: 1.1848, ept: 254.2420
    Epoch [24/50], Val Losses: mse: 50.2467, mae: 4.2139, huber: 3.7721, swd: 3.6833, ept: 166.9908
    Epoch [24/50], Test Losses: mse: 35.7539, mae: 3.2925, huber: 2.8655, swd: 2.4234, ept: 197.8469
      Epoch 24 composite train-obj: 1.594551
            Val objective improved 3.8238 → 3.7721, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 15.1705, mae: 1.9778, huber: 1.5839, swd: 1.1769, ept: 255.1980
    Epoch [25/50], Val Losses: mse: 51.4614, mae: 4.2539, huber: 3.8131, swd: 3.8726, ept: 167.9477
    Epoch [25/50], Test Losses: mse: 39.4620, mae: 3.4528, huber: 3.0257, swd: 2.7376, ept: 198.4678
      Epoch 25 composite train-obj: 1.583876
            No improvement (3.8131), counter 1/5
    Epoch [26/50], Train Losses: mse: 15.4759, mae: 1.9872, huber: 1.5939, swd: 1.2066, ept: 256.0381
    Epoch [26/50], Val Losses: mse: 49.8534, mae: 4.2279, huber: 3.7824, swd: 3.8700, ept: 165.6616
    Epoch [26/50], Test Losses: mse: 38.7393, mae: 3.4969, huber: 3.0630, swd: 2.4986, ept: 188.8200
      Epoch 26 composite train-obj: 1.593947
            No improvement (3.7824), counter 2/5
    Epoch [27/50], Train Losses: mse: 14.7115, mae: 1.9354, huber: 1.5440, swd: 1.1531, ept: 257.5764
    Epoch [27/50], Val Losses: mse: 50.3055, mae: 4.2137, huber: 3.7704, swd: 3.8395, ept: 168.7464
    Epoch [27/50], Test Losses: mse: 36.4493, mae: 3.3799, huber: 2.9490, swd: 2.6231, ept: 191.4833
      Epoch 27 composite train-obj: 1.543973
            Val objective improved 3.7721 → 3.7704, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 14.6997, mae: 1.9347, huber: 1.5427, swd: 1.1183, ept: 258.4417
    Epoch [28/50], Val Losses: mse: 49.1883, mae: 4.1468, huber: 3.7069, swd: 3.6540, ept: 166.3270
    Epoch [28/50], Test Losses: mse: 39.0594, mae: 3.4839, huber: 3.0545, swd: 2.5367, ept: 192.0647
      Epoch 28 composite train-obj: 1.542719
            Val objective improved 3.7704 → 3.7069, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 14.2448, mae: 1.8732, huber: 1.4856, swd: 1.0795, ept: 261.9643
    Epoch [29/50], Val Losses: mse: 50.4005, mae: 4.2095, huber: 3.7678, swd: 3.3966, ept: 170.5418
    Epoch [29/50], Test Losses: mse: 39.6935, mae: 3.5032, huber: 3.0742, swd: 2.5636, ept: 196.9054
      Epoch 29 composite train-obj: 1.485598
            No improvement (3.7678), counter 1/5
    Epoch [30/50], Train Losses: mse: 13.8428, mae: 1.8391, huber: 1.4529, swd: 1.0408, ept: 263.7812
    Epoch [30/50], Val Losses: mse: 50.9804, mae: 4.2952, huber: 3.8504, swd: 3.6861, ept: 163.5985
    Epoch [30/50], Test Losses: mse: 39.9042, mae: 3.5179, huber: 3.0877, swd: 2.3341, ept: 190.4080
      Epoch 30 composite train-obj: 1.452893
            No improvement (3.8504), counter 2/5
    Epoch [31/50], Train Losses: mse: 13.8885, mae: 1.8438, huber: 1.4576, swd: 1.0376, ept: 263.2050
    Epoch [31/50], Val Losses: mse: 49.5435, mae: 4.1066, huber: 3.6692, swd: 3.5657, ept: 176.1391
    Epoch [31/50], Test Losses: mse: 36.8780, mae: 3.3122, huber: 2.8887, swd: 2.4944, ept: 202.7041
      Epoch 31 composite train-obj: 1.457622
            Val objective improved 3.7069 → 3.6692, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 13.4354, mae: 1.7968, huber: 1.4136, swd: 1.0048, ept: 266.0448
    Epoch [32/50], Val Losses: mse: 49.5193, mae: 4.1731, huber: 3.7305, swd: 3.5006, ept: 171.9697
    Epoch [32/50], Test Losses: mse: 37.0497, mae: 3.3846, huber: 2.9550, swd: 2.4707, ept: 196.4400
      Epoch 32 composite train-obj: 1.413618
            No improvement (3.7305), counter 1/5
    Epoch [33/50], Train Losses: mse: 12.8450, mae: 1.7592, huber: 1.3770, swd: 0.9478, ept: 267.1066
    Epoch [33/50], Val Losses: mse: 47.9819, mae: 4.0755, huber: 3.6366, swd: 3.6818, ept: 171.2256
    Epoch [33/50], Test Losses: mse: 35.9580, mae: 3.2757, huber: 2.8523, swd: 2.4978, ept: 204.1263
      Epoch 33 composite train-obj: 1.377022
            Val objective improved 3.6692 → 3.6366, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 13.0668, mae: 1.7663, huber: 1.3841, swd: 0.9809, ept: 268.1752
    Epoch [34/50], Val Losses: mse: 49.9079, mae: 4.1741, huber: 3.7355, swd: 3.5228, ept: 170.5850
    Epoch [34/50], Test Losses: mse: 35.5558, mae: 3.2349, huber: 2.8131, swd: 2.4986, ept: 206.6776
      Epoch 34 composite train-obj: 1.384130
            No improvement (3.7355), counter 1/5
    Epoch [35/50], Train Losses: mse: 12.2452, mae: 1.6941, huber: 1.3168, swd: 0.8935, ept: 270.0460
    Epoch [35/50], Val Losses: mse: 50.7560, mae: 4.2488, huber: 3.8059, swd: 3.6360, ept: 164.1635
    Epoch [35/50], Test Losses: mse: 37.0816, mae: 3.3801, huber: 2.9529, swd: 2.3414, ept: 193.2332
      Epoch 35 composite train-obj: 1.316762
            No improvement (3.8059), counter 2/5
    Epoch [36/50], Train Losses: mse: 12.4732, mae: 1.7105, huber: 1.3322, swd: 0.9054, ept: 269.9908
    Epoch [36/50], Val Losses: mse: 51.0599, mae: 4.2441, huber: 3.8014, swd: 4.0113, ept: 165.8550
    Epoch [36/50], Test Losses: mse: 41.6989, mae: 3.5951, huber: 3.1672, swd: 2.5525, ept: 192.4603
      Epoch 36 composite train-obj: 1.332243
            No improvement (3.8014), counter 3/5
    Epoch [37/50], Train Losses: mse: 12.6238, mae: 1.7113, huber: 1.3327, swd: 0.9271, ept: 271.5449
    Epoch [37/50], Val Losses: mse: 46.5209, mae: 3.9659, huber: 3.5331, swd: 3.3665, ept: 177.6330
    Epoch [37/50], Test Losses: mse: 37.8722, mae: 3.3323, huber: 2.9124, swd: 2.3547, ept: 204.1297
      Epoch 37 composite train-obj: 1.332668
            Val objective improved 3.6366 → 3.5331, saving checkpoint.
    Epoch [38/50], Train Losses: mse: 12.2364, mae: 1.6807, huber: 1.3034, swd: 0.8667, ept: 272.2045
    Epoch [38/50], Val Losses: mse: 49.9320, mae: 4.1510, huber: 3.7105, swd: 3.0388, ept: 168.9240
    Epoch [38/50], Test Losses: mse: 37.6256, mae: 3.3560, huber: 2.9307, swd: 2.1390, ept: 201.9191
      Epoch 38 composite train-obj: 1.303417
            No improvement (3.7105), counter 1/5
    Epoch [39/50], Train Losses: mse: 12.1989, mae: 1.6837, huber: 1.3063, swd: 0.8684, ept: 272.1367
    Epoch [39/50], Val Losses: mse: 48.1126, mae: 4.0246, huber: 3.5893, swd: 3.3170, ept: 183.9841
    Epoch [39/50], Test Losses: mse: 36.3617, mae: 3.2671, huber: 2.8464, swd: 2.3567, ept: 210.9936
      Epoch 39 composite train-obj: 1.306251
            No improvement (3.5893), counter 2/5
    Epoch [40/50], Train Losses: mse: 11.9973, mae: 1.6534, huber: 1.2784, swd: 0.8787, ept: 273.9835
    Epoch [40/50], Val Losses: mse: 50.6671, mae: 4.1708, huber: 3.7325, swd: 3.1099, ept: 173.4425
    Epoch [40/50], Test Losses: mse: 36.6200, mae: 3.3332, huber: 2.9097, swd: 2.1979, ept: 198.5817
      Epoch 40 composite train-obj: 1.278439
            No improvement (3.7325), counter 3/5
    Epoch [41/50], Train Losses: mse: 11.5946, mae: 1.6157, huber: 1.2436, swd: 0.8425, ept: 275.9291
    Epoch [41/50], Val Losses: mse: 49.1762, mae: 4.1344, huber: 3.6943, swd: 3.1890, ept: 174.0287
    Epoch [41/50], Test Losses: mse: 40.3947, mae: 3.4857, huber: 3.0594, swd: 2.2845, ept: 200.8060
      Epoch 41 composite train-obj: 1.243590
            No improvement (3.6943), counter 4/5
    Epoch [42/50], Train Losses: mse: 11.8100, mae: 1.6409, huber: 1.2666, swd: 0.8451, ept: 274.3280
    Epoch [42/50], Val Losses: mse: 51.2608, mae: 4.2463, huber: 3.8048, swd: 3.9573, ept: 167.2910
    Epoch [42/50], Test Losses: mse: 40.3774, mae: 3.5056, huber: 3.0788, swd: 2.2049, ept: 196.6618
      Epoch 42 composite train-obj: 1.266632
    Epoch [42/50], Test Losses: mse: 37.8708, mae: 3.3321, huber: 2.9123, swd: 2.3545, ept: 204.1093
    Best round's Test MSE: 37.8722, MAE: 3.3323, SWD: 2.3547
    Best round's Validation MSE: 46.5209, MAE: 3.9659, SWD: 3.3665
    Best round's Test verification MSE : 37.8708, MAE: 3.3321, SWD: 2.3545
    Time taken: 128.12 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.8219, mae: 6.4586, huber: 5.9790, swd: 40.3225, ept: 34.1846
    Epoch [1/50], Val Losses: mse: 60.9102, mae: 5.9033, huber: 5.4264, swd: 26.7714, ept: 34.6040
    Epoch [1/50], Test Losses: mse: 55.6387, mae: 5.4791, huber: 5.0059, swd: 27.3992, ept: 39.6568
      Epoch 1 composite train-obj: 5.979020
            Val objective improved inf → 5.4264, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.2349, mae: 4.9550, huber: 4.4873, swd: 18.3990, ept: 82.8973
    Epoch [2/50], Val Losses: mse: 54.3987, mae: 5.2916, huber: 4.8217, swd: 17.3229, ept: 74.8887
    Epoch [2/50], Test Losses: mse: 48.0474, mae: 4.8441, huber: 4.3776, swd: 19.1229, ept: 68.9949
      Epoch 2 composite train-obj: 4.487276
            Val objective improved 5.4264 → 4.8217, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.6408, mae: 4.3554, huber: 3.8971, swd: 12.0560, ept: 113.0831
    Epoch [3/50], Val Losses: mse: 51.9009, mae: 5.0225, huber: 4.5556, swd: 11.9676, ept: 84.5904
    Epoch [3/50], Test Losses: mse: 45.3050, mae: 4.5876, huber: 4.1238, swd: 12.8914, ept: 81.7302
      Epoch 3 composite train-obj: 3.897146
            Val objective improved 4.8217 → 4.5556, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.3394, mae: 3.9842, huber: 3.5315, swd: 8.2600, ept: 131.8347
    Epoch [4/50], Val Losses: mse: 53.7992, mae: 4.9593, huber: 4.4954, swd: 9.5197, ept: 96.4407
    Epoch [4/50], Test Losses: mse: 45.9948, mae: 4.4562, huber: 3.9964, swd: 9.7951, ept: 95.8588
      Epoch 4 composite train-obj: 3.531470
            Val objective improved 4.5556 → 4.4954, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 34.5423, mae: 3.6649, huber: 3.2177, swd: 6.0647, ept: 153.3330
    Epoch [5/50], Val Losses: mse: 52.9598, mae: 4.8703, huber: 4.4078, swd: 9.0028, ept: 110.3539
    Epoch [5/50], Test Losses: mse: 43.3144, mae: 4.2708, huber: 3.8132, swd: 8.0309, ept: 118.3320
      Epoch 5 composite train-obj: 3.217662
            Val objective improved 4.4954 → 4.4078, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 31.3374, mae: 3.4043, huber: 2.9621, swd: 4.7430, ept: 170.0346
    Epoch [6/50], Val Losses: mse: 53.6465, mae: 4.8303, huber: 4.3692, swd: 7.9363, ept: 120.2108
    Epoch [6/50], Test Losses: mse: 40.4435, mae: 4.0555, huber: 3.6011, swd: 6.5238, ept: 134.1390
      Epoch 6 composite train-obj: 2.962109
            Val objective improved 4.4078 → 4.3692, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.8297, mae: 3.2131, huber: 2.7743, swd: 3.9359, ept: 180.7409
    Epoch [7/50], Val Losses: mse: 55.7269, mae: 4.8809, huber: 4.4210, swd: 7.5152, ept: 123.4964
    Epoch [7/50], Test Losses: mse: 41.8354, mae: 4.0502, huber: 3.5980, swd: 5.8961, ept: 132.8175
      Epoch 7 composite train-obj: 2.774285
            No improvement (4.4210), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.9642, mae: 3.0581, huber: 2.6233, swd: 3.3911, ept: 189.1702
    Epoch [8/50], Val Losses: mse: 59.7405, mae: 4.9916, huber: 4.5316, swd: 6.1943, ept: 127.0228
    Epoch [8/50], Test Losses: mse: 44.0032, mae: 4.0743, huber: 3.6227, swd: 5.2775, ept: 138.9192
      Epoch 8 composite train-obj: 2.623301
            No improvement (4.5316), counter 2/5
    Epoch [9/50], Train Losses: mse: 25.8537, mae: 2.9600, huber: 2.5274, swd: 3.0682, ept: 195.0211
    Epoch [9/50], Val Losses: mse: 56.3266, mae: 4.7509, huber: 4.2940, swd: 5.5973, ept: 140.2791
    Epoch [9/50], Test Losses: mse: 40.4986, mae: 3.8424, huber: 3.3949, swd: 4.6070, ept: 156.5597
      Epoch 9 composite train-obj: 2.527401
            Val objective improved 4.3692 → 4.2940, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.9868, mae: 2.7947, huber: 2.3675, swd: 2.6533, ept: 204.2311
    Epoch [10/50], Val Losses: mse: 57.3103, mae: 4.7385, huber: 4.2838, swd: 5.0076, ept: 143.1311
    Epoch [10/50], Test Losses: mse: 40.7520, mae: 3.8099, huber: 3.3638, swd: 4.3618, ept: 156.9411
      Epoch 10 composite train-obj: 2.367500
            Val objective improved 4.2940 → 4.2838, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 23.0738, mae: 2.7069, huber: 2.2826, swd: 2.4602, ept: 210.3654
    Epoch [11/50], Val Losses: mse: 56.7033, mae: 4.8057, huber: 4.3486, swd: 5.7902, ept: 132.9352
    Epoch [11/50], Test Losses: mse: 40.7034, mae: 3.8224, huber: 3.3756, swd: 4.1765, ept: 157.1385
      Epoch 11 composite train-obj: 2.282598
            No improvement (4.3486), counter 1/5
    Epoch [12/50], Train Losses: mse: 21.9184, mae: 2.6111, huber: 2.1899, swd: 2.2886, ept: 216.4023
    Epoch [12/50], Val Losses: mse: 54.2489, mae: 4.5887, huber: 4.1376, swd: 5.9820, ept: 148.2239
    Epoch [12/50], Test Losses: mse: 38.0749, mae: 3.5952, huber: 3.1550, swd: 3.6894, ept: 170.2895
      Epoch 12 composite train-obj: 2.189864
            Val objective improved 4.2838 → 4.1376, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 21.7055, mae: 2.5930, huber: 2.1722, swd: 2.2035, ept: 218.3346
    Epoch [13/50], Val Losses: mse: 52.0033, mae: 4.4982, huber: 4.0465, swd: 5.9196, ept: 142.4623
    Epoch [13/50], Test Losses: mse: 39.2711, mae: 3.7206, huber: 3.2778, swd: 4.3536, ept: 161.7371
      Epoch 13 composite train-obj: 2.172150
            Val objective improved 4.1376 → 4.0465, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 20.2861, mae: 2.4648, huber: 2.0490, swd: 2.0001, ept: 226.3936
    Epoch [14/50], Val Losses: mse: 54.3179, mae: 4.5486, huber: 4.0986, swd: 4.7553, ept: 146.9953
    Epoch [14/50], Test Losses: mse: 38.9993, mae: 3.6216, huber: 3.1828, swd: 3.5158, ept: 169.4692
      Epoch 14 composite train-obj: 2.048992
            No improvement (4.0986), counter 1/5
    Epoch [15/50], Train Losses: mse: 19.7794, mae: 2.4244, huber: 2.0098, swd: 1.8979, ept: 228.7201
    Epoch [15/50], Val Losses: mse: 53.3530, mae: 4.5586, huber: 4.1083, swd: 5.5077, ept: 146.0323
    Epoch [15/50], Test Losses: mse: 38.3508, mae: 3.6434, huber: 3.2037, swd: 3.5002, ept: 169.2517
      Epoch 15 composite train-obj: 2.009834
            No improvement (4.1083), counter 2/5
    Epoch [16/50], Train Losses: mse: 19.6929, mae: 2.4150, huber: 2.0009, swd: 1.8989, ept: 229.9082
    Epoch [16/50], Val Losses: mse: 54.2956, mae: 4.4858, huber: 4.0386, swd: 5.1795, ept: 156.0281
    Epoch [16/50], Test Losses: mse: 38.1039, mae: 3.5167, huber: 3.0810, swd: 3.4015, ept: 181.8085
      Epoch 16 composite train-obj: 2.000883
            Val objective improved 4.0465 → 4.0386, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 18.6305, mae: 2.3012, huber: 1.8924, swd: 1.7206, ept: 236.5679
    Epoch [17/50], Val Losses: mse: 57.5202, mae: 4.6937, huber: 4.2433, swd: 5.2063, ept: 146.2819
    Epoch [17/50], Test Losses: mse: 39.1972, mae: 3.6173, huber: 3.1790, swd: 3.3922, ept: 171.8839
      Epoch 17 composite train-obj: 1.892373
            No improvement (4.2433), counter 1/5
    Epoch [18/50], Train Losses: mse: 18.1838, mae: 2.2727, huber: 1.8643, swd: 1.6964, ept: 238.6985
    Epoch [18/50], Val Losses: mse: 54.4918, mae: 4.5169, huber: 4.0691, swd: 4.7900, ept: 155.7119
    Epoch [18/50], Test Losses: mse: 40.8172, mae: 3.6088, huber: 3.1726, swd: 3.0839, ept: 186.8146
      Epoch 18 composite train-obj: 1.864253
            No improvement (4.0691), counter 2/5
    Epoch [19/50], Train Losses: mse: 17.5120, mae: 2.2095, huber: 1.8044, swd: 1.5783, ept: 242.2660
    Epoch [19/50], Val Losses: mse: 52.9786, mae: 4.4018, huber: 3.9553, swd: 4.4404, ept: 161.1635
    Epoch [19/50], Test Losses: mse: 35.9987, mae: 3.3654, huber: 2.9322, swd: 2.9258, ept: 190.8889
      Epoch 19 composite train-obj: 1.804420
            Val objective improved 4.0386 → 3.9553, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 16.5869, mae: 2.1265, huber: 1.7252, swd: 1.4652, ept: 246.7874
    Epoch [20/50], Val Losses: mse: 51.9309, mae: 4.3618, huber: 3.9147, swd: 4.5566, ept: 160.3839
    Epoch [20/50], Test Losses: mse: 39.1107, mae: 3.5384, huber: 3.1033, swd: 3.2070, ept: 185.6564
      Epoch 20 composite train-obj: 1.725222
            Val objective improved 3.9553 → 3.9147, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 16.7482, mae: 2.1352, huber: 1.7330, swd: 1.4569, ept: 247.4205
    Epoch [21/50], Val Losses: mse: 53.6729, mae: 4.4570, huber: 4.0092, swd: 4.3952, ept: 152.6789
    Epoch [21/50], Test Losses: mse: 39.0042, mae: 3.6005, huber: 3.1628, swd: 3.0096, ept: 176.3572
      Epoch 21 composite train-obj: 1.733035
            No improvement (4.0092), counter 1/5
    Epoch [22/50], Train Losses: mse: 16.5285, mae: 2.1003, huber: 1.7006, swd: 1.4323, ept: 249.7193
    Epoch [22/50], Val Losses: mse: 51.9844, mae: 4.3940, huber: 3.9460, swd: 4.5798, ept: 158.0759
    Epoch [22/50], Test Losses: mse: 38.7356, mae: 3.5056, huber: 3.0702, swd: 2.7764, ept: 187.9109
      Epoch 22 composite train-obj: 1.700630
            No improvement (3.9460), counter 2/5
    Epoch [23/50], Train Losses: mse: 16.0512, mae: 2.0612, huber: 1.6633, swd: 1.3918, ept: 252.0319
    Epoch [23/50], Val Losses: mse: 52.4626, mae: 4.3924, huber: 3.9448, swd: 4.7053, ept: 159.7502
    Epoch [23/50], Test Losses: mse: 39.3010, mae: 3.5602, huber: 3.1234, swd: 3.0364, ept: 184.7011
      Epoch 23 composite train-obj: 1.663271
            No improvement (3.9448), counter 3/5
    Epoch [24/50], Train Losses: mse: 16.7726, mae: 2.1242, huber: 1.7230, swd: 1.4440, ept: 249.0607
    Epoch [24/50], Val Losses: mse: 53.7000, mae: 4.4072, huber: 3.9616, swd: 4.2111, ept: 160.8577
    Epoch [24/50], Test Losses: mse: 37.0953, mae: 3.3910, huber: 2.9603, swd: 2.7883, ept: 192.8023
      Epoch 24 composite train-obj: 1.723029
            No improvement (3.9616), counter 4/5
    Epoch [25/50], Train Losses: mse: 14.7620, mae: 1.9492, huber: 1.5562, swd: 1.2560, ept: 257.6240
    Epoch [25/50], Val Losses: mse: 57.2466, mae: 4.5587, huber: 4.1130, swd: 4.1069, ept: 159.9982
    Epoch [25/50], Test Losses: mse: 36.0822, mae: 3.4014, huber: 2.9690, swd: 2.7788, ept: 185.3866
      Epoch 25 composite train-obj: 1.556179
    Epoch [25/50], Test Losses: mse: 39.0998, mae: 3.5379, huber: 3.1028, swd: 3.2077, ept: 185.7361
    Best round's Test MSE: 39.1107, MAE: 3.5384, SWD: 3.2070
    Best round's Validation MSE: 51.9309, MAE: 4.3618, SWD: 4.5566
    Best round's Test verification MSE : 39.0998, MAE: 3.5379, SWD: 3.2077
    Time taken: 80.20 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1617)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 37.0627 ± 2.0828
      mae: 3.3473 ± 0.1503
      huber: 2.9217 ± 0.1446
      swd: 2.5161 ± 0.5112
      ept: 198.7266 ± 9.2882
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 48.3290 ± 2.5469
      mae: 4.1061 ± 0.1811
      huber: 3.6674 ± 0.1751
      swd: 3.7029 ± 0.6081
      ept: 169.2002 ± 7.0471
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 324.27 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1617
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: OT dissect: Shift only for OT


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_shift_only=True, ### HERE
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
    
    Epoch [1/50], Train Losses: mse: 74.7644, mae: 6.4882, huber: 6.0089, swd: 42.7680, ept: 43.6549
    Epoch [1/50], Val Losses: mse: 59.9910, mae: 5.7901, huber: 5.3153, swd: 23.2824, ept: 59.5904
    Epoch [1/50], Test Losses: mse: 55.0858, mae: 5.3538, huber: 4.8830, swd: 23.0242, ept: 61.2007
      Epoch 1 composite train-obj: 6.008860
            Val objective improved inf → 5.3153, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.9733, mae: 4.7326, huber: 4.2690, swd: 14.5144, ept: 118.5624
    Epoch [2/50], Val Losses: mse: 51.7065, mae: 5.0963, huber: 4.6286, swd: 12.7607, ept: 107.5776
    Epoch [2/50], Test Losses: mse: 46.4669, mae: 4.6723, huber: 4.2088, swd: 12.8247, ept: 109.4742
      Epoch 2 composite train-obj: 4.269026
            Val objective improved 5.3153 → 4.6286, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.5538, mae: 4.0939, huber: 3.6413, swd: 8.9376, ept: 153.9992
    Epoch [3/50], Val Losses: mse: 50.3020, mae: 4.8646, huber: 4.4038, swd: 9.4053, ept: 121.3309
    Epoch [3/50], Test Losses: mse: 41.5581, mae: 4.2645, huber: 3.8082, swd: 9.5632, ept: 131.3484
      Epoch 3 composite train-obj: 3.641299
            Val objective improved 4.6286 → 4.4038, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.5640, mae: 3.7327, huber: 3.2882, swd: 6.7954, ept: 172.1152
    Epoch [4/50], Val Losses: mse: 49.2712, mae: 4.7265, huber: 4.2678, swd: 7.2289, ept: 129.1223
    Epoch [4/50], Test Losses: mse: 40.5116, mae: 4.1517, huber: 3.6975, swd: 7.5680, ept: 142.2839
      Epoch 4 composite train-obj: 3.288187
            Val objective improved 4.4038 → 4.2678, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.6187, mae: 3.4827, huber: 3.0440, swd: 5.2948, ept: 184.0204
    Epoch [5/50], Val Losses: mse: 47.0015, mae: 4.5294, huber: 4.0744, swd: 6.5591, ept: 142.1938
    Epoch [5/50], Test Losses: mse: 37.7842, mae: 3.9251, huber: 3.4752, swd: 6.1163, ept: 160.0401
      Epoch 5 composite train-obj: 3.043999
            Val objective improved 4.2678 → 4.0744, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.4625, mae: 3.2907, huber: 2.8567, swd: 4.3989, ept: 194.6752
    Epoch [6/50], Val Losses: mse: 47.4517, mae: 4.4268, huber: 3.9778, swd: 5.5032, ept: 147.2007
    Epoch [6/50], Test Losses: mse: 37.0613, mae: 3.7680, huber: 3.3265, swd: 4.9278, ept: 164.9173
      Epoch 6 composite train-obj: 2.856666
            Val objective improved 4.0744 → 3.9778, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.3017, mae: 3.1179, huber: 2.6881, swd: 3.8097, ept: 202.8538
    Epoch [7/50], Val Losses: mse: 51.5978, mae: 4.6210, huber: 4.1683, swd: 5.3371, ept: 152.6390
    Epoch [7/50], Test Losses: mse: 36.1525, mae: 3.7329, huber: 3.2875, swd: 5.1986, ept: 171.0360
      Epoch 7 composite train-obj: 2.688137
            No improvement (4.1683), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.3233, mae: 2.9662, huber: 2.5395, swd: 3.2808, ept: 209.4636
    Epoch [8/50], Val Losses: mse: 45.9050, mae: 4.2710, huber: 3.8223, swd: 4.3497, ept: 158.8128
    Epoch [8/50], Test Losses: mse: 35.5483, mae: 3.6092, huber: 3.1688, swd: 3.9783, ept: 179.0514
      Epoch 8 composite train-obj: 2.539539
            Val objective improved 3.9778 → 3.8223, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.3982, mae: 2.8135, huber: 2.3910, swd: 2.8428, ept: 216.1947
    Epoch [9/50], Val Losses: mse: 45.1121, mae: 4.2182, huber: 3.7723, swd: 4.3649, ept: 166.9502
    Epoch [9/50], Test Losses: mse: 34.0963, mae: 3.4993, huber: 3.0612, swd: 3.7206, ept: 184.1457
      Epoch 9 composite train-obj: 2.390965
            Val objective improved 3.8223 → 3.7723, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 22.7855, mae: 2.6869, huber: 2.2674, swd: 2.5031, ept: 223.0563
    Epoch [10/50], Val Losses: mse: 47.4005, mae: 4.2256, huber: 3.7823, swd: 3.9511, ept: 167.5226
    Epoch [10/50], Test Losses: mse: 33.6598, mae: 3.4289, huber: 2.9941, swd: 3.2398, ept: 185.5732
      Epoch 10 composite train-obj: 2.267438
            No improvement (3.7823), counter 1/5
    Epoch [11/50], Train Losses: mse: 22.0740, mae: 2.6208, huber: 2.2031, swd: 2.3206, ept: 226.6325
    Epoch [11/50], Val Losses: mse: 49.9790, mae: 4.3302, huber: 3.8860, swd: 3.6474, ept: 171.1403
    Epoch [11/50], Test Losses: mse: 36.0799, mae: 3.5440, huber: 3.1060, swd: 2.8319, ept: 190.3299
      Epoch 11 composite train-obj: 2.203058
            No improvement (3.8860), counter 2/5
    Epoch [12/50], Train Losses: mse: 20.8259, mae: 2.5188, huber: 2.1039, swd: 2.1004, ept: 231.9543
    Epoch [12/50], Val Losses: mse: 45.2650, mae: 4.0695, huber: 3.6284, swd: 3.3511, ept: 171.8056
    Epoch [12/50], Test Losses: mse: 34.0395, mae: 3.3988, huber: 2.9671, swd: 2.9250, ept: 193.3212
      Epoch 12 composite train-obj: 2.103923
            Val objective improved 3.7723 → 3.6284, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.6861, mae: 2.4333, huber: 2.0201, swd: 1.9189, ept: 235.8718
    Epoch [13/50], Val Losses: mse: 47.8167, mae: 4.1938, huber: 3.7502, swd: 3.4053, ept: 178.3663
    Epoch [13/50], Test Losses: mse: 35.7861, mae: 3.4782, huber: 3.0427, swd: 2.8580, ept: 199.6212
      Epoch 13 composite train-obj: 2.020141
            No improvement (3.7502), counter 1/5
    Epoch [14/50], Train Losses: mse: 18.5641, mae: 2.3327, huber: 1.9236, swd: 1.7149, ept: 240.5286
    Epoch [14/50], Val Losses: mse: 45.2948, mae: 4.0272, huber: 3.5844, swd: 2.8993, ept: 181.0754
    Epoch [14/50], Test Losses: mse: 32.8581, mae: 3.3465, huber: 2.9126, swd: 2.6529, ept: 198.1186
      Epoch 14 composite train-obj: 1.923614
            Val objective improved 3.6284 → 3.5844, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 18.1893, mae: 2.2851, huber: 1.8783, swd: 1.6302, ept: 243.9544
    Epoch [15/50], Val Losses: mse: 44.9439, mae: 3.9525, huber: 3.5156, swd: 2.8604, ept: 185.0228
    Epoch [15/50], Test Losses: mse: 32.9352, mae: 3.2750, huber: 2.8466, swd: 2.4146, ept: 206.0897
      Epoch 15 composite train-obj: 1.878301
            Val objective improved 3.5844 → 3.5156, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 17.5440, mae: 2.2345, huber: 1.8291, swd: 1.5551, ept: 246.5579
    Epoch [16/50], Val Losses: mse: 45.4725, mae: 3.9719, huber: 3.5339, swd: 2.6497, ept: 186.3319
    Epoch [16/50], Test Losses: mse: 32.5481, mae: 3.2100, huber: 2.7829, swd: 2.1542, ept: 208.9659
      Epoch 16 composite train-obj: 1.829101
            No improvement (3.5339), counter 1/5
    Epoch [17/50], Train Losses: mse: 16.8563, mae: 2.1669, huber: 1.7642, swd: 1.4313, ept: 250.4805
    Epoch [17/50], Val Losses: mse: 50.2588, mae: 4.1869, huber: 3.7483, swd: 2.7251, ept: 186.1553
    Epoch [17/50], Test Losses: mse: 34.5644, mae: 3.3526, huber: 2.9241, swd: 2.3099, ept: 207.1199
      Epoch 17 composite train-obj: 1.764178
            No improvement (3.7483), counter 2/5
    Epoch [18/50], Train Losses: mse: 16.5744, mae: 2.1525, huber: 1.7495, swd: 1.3992, ept: 252.1121
    Epoch [18/50], Val Losses: mse: 47.8740, mae: 4.0811, huber: 3.6456, swd: 2.8200, ept: 190.8723
    Epoch [18/50], Test Losses: mse: 32.2375, mae: 3.2183, huber: 2.7915, swd: 2.3916, ept: 210.8493
      Epoch 18 composite train-obj: 1.749510
            No improvement (3.6456), counter 3/5
    Epoch [19/50], Train Losses: mse: 15.7581, mae: 2.0783, huber: 1.6782, swd: 1.2987, ept: 256.1386
    Epoch [19/50], Val Losses: mse: 44.8714, mae: 3.9039, huber: 3.4678, swd: 2.5959, ept: 188.1420
    Epoch [19/50], Test Losses: mse: 32.6598, mae: 3.2118, huber: 2.7842, swd: 2.1298, ept: 215.1054
      Epoch 19 composite train-obj: 1.678169
            Val objective improved 3.5156 → 3.4678, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 14.5830, mae: 1.9627, huber: 1.5694, swd: 1.1713, ept: 260.8714
    Epoch [20/50], Val Losses: mse: 43.4352, mae: 3.7919, huber: 3.3585, swd: 2.3798, ept: 196.3688
    Epoch [20/50], Test Losses: mse: 31.3573, mae: 3.1280, huber: 2.7036, swd: 1.8435, ept: 218.9474
      Epoch 20 composite train-obj: 1.569424
            Val objective improved 3.4678 → 3.3585, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 14.8633, mae: 1.9906, huber: 1.5949, swd: 1.1788, ept: 260.5453
    Epoch [21/50], Val Losses: mse: 44.1265, mae: 3.8936, huber: 3.4589, swd: 2.5944, ept: 191.3209
    Epoch [21/50], Test Losses: mse: 34.7456, mae: 3.2905, huber: 2.8658, swd: 2.2623, ept: 216.2942
      Epoch 21 composite train-obj: 1.594888
            No improvement (3.4589), counter 1/5
    Epoch [22/50], Train Losses: mse: 14.7793, mae: 1.9785, huber: 1.5831, swd: 1.1718, ept: 262.1055
    Epoch [22/50], Val Losses: mse: 46.4504, mae: 3.8904, huber: 3.4583, swd: 2.3175, ept: 198.7950
    Epoch [22/50], Test Losses: mse: 32.3152, mae: 3.1169, huber: 2.6968, swd: 1.9310, ept: 222.0748
      Epoch 22 composite train-obj: 1.583096
            No improvement (3.4583), counter 2/5
    Epoch [23/50], Train Losses: mse: 14.4593, mae: 1.9485, huber: 1.5550, swd: 1.1322, ept: 263.7949
    Epoch [23/50], Val Losses: mse: 40.9683, mae: 3.6782, huber: 3.2460, swd: 2.3286, ept: 203.5081
    Epoch [23/50], Test Losses: mse: 32.5410, mae: 3.1513, huber: 2.7305, swd: 2.0528, ept: 222.6136
      Epoch 23 composite train-obj: 1.554951
            Val objective improved 3.3585 → 3.2460, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 13.5484, mae: 1.8635, huber: 1.4740, swd: 1.0442, ept: 267.4677
    Epoch [24/50], Val Losses: mse: 45.6367, mae: 3.8804, huber: 3.4475, swd: 2.2078, ept: 197.1577
    Epoch [24/50], Test Losses: mse: 31.5965, mae: 3.1142, huber: 2.6911, swd: 1.7286, ept: 219.7029
      Epoch 24 composite train-obj: 1.473986
            No improvement (3.4475), counter 1/5
    Epoch [25/50], Train Losses: mse: 13.1986, mae: 1.8336, huber: 1.4450, swd: 1.0089, ept: 269.0775
    Epoch [25/50], Val Losses: mse: 43.4953, mae: 3.7690, huber: 3.3369, swd: 2.3378, ept: 198.4996
    Epoch [25/50], Test Losses: mse: 30.8350, mae: 3.0159, huber: 2.5983, swd: 1.7748, ept: 229.2894
      Epoch 25 composite train-obj: 1.445039
            No improvement (3.3369), counter 2/5
    Epoch [26/50], Train Losses: mse: 12.9522, mae: 1.8143, huber: 1.4260, swd: 0.9816, ept: 270.7490
    Epoch [26/50], Val Losses: mse: 41.1075, mae: 3.6476, huber: 3.2163, swd: 2.3282, ept: 204.7472
    Epoch [26/50], Test Losses: mse: 32.3694, mae: 3.1036, huber: 2.6831, swd: 1.7467, ept: 224.8680
      Epoch 26 composite train-obj: 1.426036
            Val objective improved 3.2460 → 3.2163, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 12.7762, mae: 1.7934, huber: 1.4066, swd: 0.9525, ept: 272.4262
    Epoch [27/50], Val Losses: mse: 47.5994, mae: 3.9575, huber: 3.5227, swd: 2.2895, ept: 195.9449
    Epoch [27/50], Test Losses: mse: 33.2704, mae: 3.1853, huber: 2.7617, swd: 1.8096, ept: 218.1594
      Epoch 27 composite train-obj: 1.406648
            No improvement (3.5227), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.4056, mae: 1.7568, huber: 1.3716, swd: 0.9146, ept: 274.8517
    Epoch [28/50], Val Losses: mse: 48.9262, mae: 4.0345, huber: 3.5999, swd: 2.3265, ept: 194.3375
    Epoch [28/50], Test Losses: mse: 31.1989, mae: 3.1092, huber: 2.6871, swd: 1.8186, ept: 217.8752
      Epoch 28 composite train-obj: 1.371581
            No improvement (3.5999), counter 2/5
    Epoch [29/50], Train Losses: mse: 13.3306, mae: 1.8470, huber: 1.4558, swd: 0.9813, ept: 272.4488
    Epoch [29/50], Val Losses: mse: 41.9597, mae: 3.6525, huber: 3.2229, swd: 2.1245, ept: 204.0272
    Epoch [29/50], Test Losses: mse: 33.7817, mae: 3.1526, huber: 2.7317, swd: 1.7459, ept: 224.4791
      Epoch 29 composite train-obj: 1.455774
            No improvement (3.2229), counter 3/5
    Epoch [30/50], Train Losses: mse: 12.2326, mae: 1.7133, huber: 1.3327, swd: 0.8794, ept: 278.2970
    Epoch [30/50], Val Losses: mse: 44.5113, mae: 3.7874, huber: 3.3569, swd: 2.0380, ept: 202.1446
    Epoch [30/50], Test Losses: mse: 33.4230, mae: 3.1257, huber: 2.7063, swd: 1.6868, ept: 230.6702
      Epoch 30 composite train-obj: 1.332674
            No improvement (3.3569), counter 4/5
    Epoch [31/50], Train Losses: mse: 11.9021, mae: 1.7082, huber: 1.3253, swd: 0.8614, ept: 278.2770
    Epoch [31/50], Val Losses: mse: 43.5822, mae: 3.7277, huber: 3.2992, swd: 2.2055, ept: 200.3650
    Epoch [31/50], Test Losses: mse: 32.9844, mae: 3.1323, huber: 2.7128, swd: 1.6677, ept: 222.9855
      Epoch 31 composite train-obj: 1.325267
    Epoch [31/50], Test Losses: mse: 32.3705, mae: 3.1036, huber: 2.6831, swd: 1.7460, ept: 224.9136
    Best round's Test MSE: 32.3694, MAE: 3.1036, SWD: 1.7467
    Best round's Validation MSE: 41.1075, MAE: 3.6476, SWD: 2.3282
    Best round's Test verification MSE : 32.3705, MAE: 3.1036, SWD: 1.7460
    Time taken: 102.35 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.2901, mae: 6.5077, huber: 6.0287, swd: 42.4338, ept: 45.8745
    Epoch [1/50], Val Losses: mse: 57.6812, mae: 5.6571, huber: 5.1829, swd: 19.9870, ept: 68.3300
    Epoch [1/50], Test Losses: mse: 53.0923, mae: 5.2400, huber: 4.7700, swd: 20.3899, ept: 72.9364
      Epoch 1 composite train-obj: 6.028683
            Val objective improved inf → 5.1829, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 45.8111, mae: 4.6948, huber: 4.2317, swd: 13.2113, ept: 117.4723
    Epoch [2/50], Val Losses: mse: 49.6461, mae: 5.0083, huber: 4.5416, swd: 11.7415, ept: 106.9946
    Epoch [2/50], Test Losses: mse: 45.4695, mae: 4.6256, huber: 4.1621, swd: 11.9758, ept: 111.7833
      Epoch 2 composite train-obj: 4.231720
            Val objective improved 5.1829 → 4.5416, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.5193, mae: 4.1170, huber: 3.6646, swd: 8.7832, ept: 150.1128
    Epoch [3/50], Val Losses: mse: 52.5979, mae: 4.9727, huber: 4.5112, swd: 8.8624, ept: 121.5794
    Epoch [3/50], Test Losses: mse: 42.0369, mae: 4.3100, huber: 3.8533, swd: 9.2566, ept: 127.9946
      Epoch 3 composite train-obj: 3.664611
            Val objective improved 4.5416 → 4.5112, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.6421, mae: 3.7738, huber: 3.3279, swd: 6.6406, ept: 168.1555
    Epoch [4/50], Val Losses: mse: 46.8030, mae: 4.5382, huber: 4.0814, swd: 6.8659, ept: 136.7372
    Epoch [4/50], Test Losses: mse: 40.1309, mae: 4.1189, huber: 3.6640, swd: 7.0196, ept: 144.3824
      Epoch 4 composite train-obj: 3.327890
            Val objective improved 4.5112 → 4.0814, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.8117, mae: 3.5153, huber: 3.0754, swd: 5.2460, ept: 181.7451
    Epoch [5/50], Val Losses: mse: 45.2939, mae: 4.4313, huber: 3.9767, swd: 6.3425, ept: 142.6509
    Epoch [5/50], Test Losses: mse: 37.9715, mae: 3.9475, huber: 3.4968, swd: 6.0751, ept: 153.4381
      Epoch 5 composite train-obj: 3.075364
            Val objective improved 4.0814 → 3.9767, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.4186, mae: 3.3095, huber: 2.8743, swd: 4.3475, ept: 192.3577
    Epoch [6/50], Val Losses: mse: 47.0434, mae: 4.4507, huber: 3.9979, swd: 5.4984, ept: 145.4762
    Epoch [6/50], Test Losses: mse: 37.4363, mae: 3.8490, huber: 3.4007, swd: 5.2033, ept: 160.3518
      Epoch 6 composite train-obj: 2.874348
            No improvement (3.9979), counter 1/5
    Epoch [7/50], Train Losses: mse: 28.3800, mae: 3.1481, huber: 2.7169, swd: 3.7932, ept: 200.6211
    Epoch [7/50], Val Losses: mse: 45.6729, mae: 4.3527, huber: 3.9011, swd: 5.1128, ept: 152.2293
    Epoch [7/50], Test Losses: mse: 36.5792, mae: 3.7803, huber: 3.3338, swd: 5.0348, ept: 165.0857
      Epoch 7 composite train-obj: 2.716924
            Val objective improved 3.9767 → 3.9011, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.9204, mae: 3.0298, huber: 2.6006, swd: 3.3473, ept: 206.7044
    Epoch [8/50], Val Losses: mse: 45.4784, mae: 4.3025, huber: 3.8530, swd: 4.6689, ept: 157.4097
    Epoch [8/50], Test Losses: mse: 36.8576, mae: 3.7407, huber: 3.2972, swd: 4.4625, ept: 170.7258
      Epoch 8 composite train-obj: 2.600611
            Val objective improved 3.9011 → 3.8530, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 24.9706, mae: 2.8748, huber: 2.4497, swd: 2.9466, ept: 214.5835
    Epoch [9/50], Val Losses: mse: 45.3505, mae: 4.2558, huber: 3.8072, swd: 4.0784, ept: 160.6706
    Epoch [9/50], Test Losses: mse: 35.6050, mae: 3.6871, huber: 3.2437, swd: 4.2515, ept: 173.3817
      Epoch 9 composite train-obj: 2.449678
            Val objective improved 3.8530 → 3.8072, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.6251, mae: 2.7713, huber: 2.3484, swd: 2.6398, ept: 219.3422
    Epoch [10/50], Val Losses: mse: 46.2613, mae: 4.2607, huber: 3.8113, swd: 3.9570, ept: 164.0593
    Epoch [10/50], Test Losses: mse: 35.8318, mae: 3.6240, huber: 3.1829, swd: 3.6467, ept: 180.5580
      Epoch 10 composite train-obj: 2.348401
            No improvement (3.8113), counter 1/5
    Epoch [11/50], Train Losses: mse: 23.0734, mae: 2.7275, huber: 2.3053, swd: 2.4840, ept: 223.0396
    Epoch [11/50], Val Losses: mse: 46.3879, mae: 4.2064, huber: 3.7590, swd: 3.8737, ept: 168.6217
    Epoch [11/50], Test Losses: mse: 37.6637, mae: 3.6848, huber: 3.2429, swd: 3.4226, ept: 184.7335
      Epoch 11 composite train-obj: 2.305274
            Val objective improved 3.8072 → 3.7590, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 21.2678, mae: 2.5811, huber: 2.1631, swd: 2.2036, ept: 229.6072
    Epoch [12/50], Val Losses: mse: 45.2598, mae: 4.0910, huber: 3.6484, swd: 3.2070, ept: 172.0883
    Epoch [12/50], Test Losses: mse: 35.3759, mae: 3.5015, huber: 3.0664, swd: 3.0277, ept: 193.2164
      Epoch 12 composite train-obj: 2.163078
            Val objective improved 3.7590 → 3.6484, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.5738, mae: 2.5260, huber: 2.1093, swd: 2.0604, ept: 233.3462
    Epoch [13/50], Val Losses: mse: 45.7549, mae: 4.1209, huber: 3.6750, swd: 3.3247, ept: 173.9614
    Epoch [13/50], Test Losses: mse: 35.6130, mae: 3.5154, huber: 3.0765, swd: 2.9927, ept: 187.9839
      Epoch 13 composite train-obj: 2.109284
            No improvement (3.6750), counter 1/5
    Epoch [14/50], Train Losses: mse: 19.1734, mae: 2.3942, huber: 1.9832, swd: 1.8194, ept: 238.8679
    Epoch [14/50], Val Losses: mse: 43.1630, mae: 3.9647, huber: 3.5231, swd: 3.0248, ept: 176.4455
    Epoch [14/50], Test Losses: mse: 34.3834, mae: 3.4302, huber: 2.9950, swd: 2.7269, ept: 196.6316
      Epoch 14 composite train-obj: 1.983174
            Val objective improved 3.6484 → 3.5231, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 18.7026, mae: 2.3534, huber: 1.9431, swd: 1.7270, ept: 242.6563
    Epoch [15/50], Val Losses: mse: 43.8037, mae: 3.9575, huber: 3.5179, swd: 2.9536, ept: 179.8950
    Epoch [15/50], Test Losses: mse: 33.3727, mae: 3.3081, huber: 2.8767, swd: 2.3809, ept: 200.9127
      Epoch 15 composite train-obj: 1.943084
            Val objective improved 3.5231 → 3.5179, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 17.5367, mae: 2.2524, huber: 1.8457, swd: 1.5805, ept: 247.2289
    Epoch [16/50], Val Losses: mse: 41.7360, mae: 3.8306, huber: 3.3928, swd: 2.7091, ept: 184.2098
    Epoch [16/50], Test Losses: mse: 33.3693, mae: 3.3142, huber: 2.8834, swd: 2.3254, ept: 199.5082
      Epoch 16 composite train-obj: 1.845699
            Val objective improved 3.5179 → 3.3928, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 17.5660, mae: 2.2529, huber: 1.8462, swd: 1.5404, ept: 249.1083
    Epoch [17/50], Val Losses: mse: 44.2731, mae: 3.8768, huber: 3.4412, swd: 2.5242, ept: 185.4272
    Epoch [17/50], Test Losses: mse: 33.2023, mae: 3.2869, huber: 2.8581, swd: 2.3164, ept: 202.1931
      Epoch 17 composite train-obj: 1.846173
            No improvement (3.4412), counter 1/5
    Epoch [18/50], Train Losses: mse: 17.0054, mae: 2.1979, huber: 1.7933, swd: 1.4500, ept: 252.1421
    Epoch [18/50], Val Losses: mse: 41.8977, mae: 3.8165, huber: 3.3788, swd: 2.6617, ept: 189.3645
    Epoch [18/50], Test Losses: mse: 32.6546, mae: 3.2504, huber: 2.8206, swd: 2.3259, ept: 212.5008
      Epoch 18 composite train-obj: 1.793285
            Val objective improved 3.3928 → 3.3788, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 16.2896, mae: 2.1394, huber: 1.7366, swd: 1.3669, ept: 254.5068
    Epoch [19/50], Val Losses: mse: 44.6017, mae: 3.9051, huber: 3.4694, swd: 2.6981, ept: 186.7407
    Epoch [19/50], Test Losses: mse: 31.8868, mae: 3.2001, huber: 2.7735, swd: 2.2751, ept: 209.8374
      Epoch 19 composite train-obj: 1.736643
            No improvement (3.4694), counter 1/5
    Epoch [20/50], Train Losses: mse: 15.7423, mae: 2.0877, huber: 1.6869, swd: 1.2989, ept: 257.1152
    Epoch [20/50], Val Losses: mse: 40.6420, mae: 3.7245, huber: 3.2895, swd: 2.5698, ept: 190.1661
    Epoch [20/50], Test Losses: mse: 31.5312, mae: 3.1739, huber: 2.7483, swd: 2.1045, ept: 211.0098
      Epoch 20 composite train-obj: 1.686875
            Val objective improved 3.3788 → 3.2895, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 15.1421, mae: 2.0258, huber: 1.6285, swd: 1.2237, ept: 260.1204
    Epoch [21/50], Val Losses: mse: 42.1382, mae: 3.7501, huber: 3.3148, swd: 2.4431, ept: 191.5484
    Epoch [21/50], Test Losses: mse: 33.8944, mae: 3.2566, huber: 2.8298, swd: 2.0593, ept: 211.6457
      Epoch 21 composite train-obj: 1.628530
            No improvement (3.3148), counter 1/5
    Epoch [22/50], Train Losses: mse: 15.1951, mae: 2.0258, huber: 1.6279, swd: 1.2013, ept: 261.1891
    Epoch [22/50], Val Losses: mse: 43.4542, mae: 3.8282, huber: 3.3904, swd: 2.3606, ept: 192.4957
    Epoch [22/50], Test Losses: mse: 31.9420, mae: 3.2163, huber: 2.7866, swd: 2.0363, ept: 215.4382
      Epoch 22 composite train-obj: 1.627948
            No improvement (3.3904), counter 2/5
    Epoch [23/50], Train Losses: mse: 14.9598, mae: 2.0085, huber: 1.6116, swd: 1.1881, ept: 262.0132
    Epoch [23/50], Val Losses: mse: 43.1284, mae: 3.7936, huber: 3.3574, swd: 2.2708, ept: 200.0049
    Epoch [23/50], Test Losses: mse: 32.2886, mae: 3.1822, huber: 2.7546, swd: 2.0563, ept: 215.7987
      Epoch 23 composite train-obj: 1.611624
            No improvement (3.3574), counter 3/5
    Epoch [24/50], Train Losses: mse: 15.2434, mae: 2.0183, huber: 1.6205, swd: 1.1676, ept: 262.3762
    Epoch [24/50], Val Losses: mse: 39.6241, mae: 3.5685, huber: 3.1371, swd: 2.1999, ept: 204.5867
    Epoch [24/50], Test Losses: mse: 30.8447, mae: 3.0752, huber: 2.6508, swd: 1.8074, ept: 223.8305
      Epoch 24 composite train-obj: 1.620534
            Val objective improved 3.2895 → 3.1371, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 14.1213, mae: 1.9183, huber: 1.5255, swd: 1.0926, ept: 265.9134
    Epoch [25/50], Val Losses: mse: 43.1140, mae: 3.7527, huber: 3.3216, swd: 2.4172, ept: 198.9527
    Epoch [25/50], Test Losses: mse: 33.6492, mae: 3.2046, huber: 2.7803, swd: 1.7634, ept: 218.1193
      Epoch 25 composite train-obj: 1.525460
            No improvement (3.3216), counter 1/5
    Epoch [26/50], Train Losses: mse: 13.8697, mae: 1.8940, huber: 1.5028, swd: 1.0595, ept: 266.8084
    Epoch [26/50], Val Losses: mse: 40.3544, mae: 3.6507, huber: 3.2192, swd: 2.2653, ept: 196.7822
    Epoch [26/50], Test Losses: mse: 30.1175, mae: 3.0446, huber: 2.6217, swd: 1.7836, ept: 220.2159
      Epoch 26 composite train-obj: 1.502808
            No improvement (3.2192), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.9510, mae: 1.8069, huber: 1.4207, swd: 0.9882, ept: 270.4247
    Epoch [27/50], Val Losses: mse: 44.1856, mae: 3.7515, huber: 3.3207, swd: 2.0499, ept: 201.3925
    Epoch [27/50], Test Losses: mse: 32.6462, mae: 3.1785, huber: 2.7536, swd: 1.6819, ept: 216.0637
      Epoch 27 composite train-obj: 1.420664
            No improvement (3.3207), counter 3/5
    Epoch [28/50], Train Losses: mse: 13.1820, mae: 1.8307, huber: 1.4421, swd: 0.9869, ept: 270.5769
    Epoch [28/50], Val Losses: mse: 40.6272, mae: 3.6542, huber: 3.2222, swd: 2.5567, ept: 199.9874
    Epoch [28/50], Test Losses: mse: 31.1305, mae: 3.0646, huber: 2.6423, swd: 1.7391, ept: 220.3491
      Epoch 28 composite train-obj: 1.442093
            No improvement (3.2222), counter 4/5
    Epoch [29/50], Train Losses: mse: 13.4059, mae: 1.8506, huber: 1.4601, swd: 0.9965, ept: 270.8141
    Epoch [29/50], Val Losses: mse: 39.2120, mae: 3.5322, huber: 3.1038, swd: 2.0985, ept: 206.2678
    Epoch [29/50], Test Losses: mse: 33.0172, mae: 3.1418, huber: 2.7206, swd: 1.8369, ept: 220.8103
      Epoch 29 composite train-obj: 1.460072
            Val objective improved 3.1371 → 3.1038, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 12.8179, mae: 1.7819, huber: 1.3972, swd: 0.9360, ept: 272.9837
    Epoch [30/50], Val Losses: mse: 39.4262, mae: 3.5619, huber: 3.1320, swd: 1.9558, ept: 205.2509
    Epoch [30/50], Test Losses: mse: 31.2740, mae: 3.0585, huber: 2.6374, swd: 1.5234, ept: 224.2531
      Epoch 30 composite train-obj: 1.397160
            No improvement (3.1320), counter 1/5
    Epoch [31/50], Train Losses: mse: 11.9008, mae: 1.6994, huber: 1.3193, swd: 0.8678, ept: 276.1923
    Epoch [31/50], Val Losses: mse: 42.4808, mae: 3.6685, huber: 3.2396, swd: 1.9372, ept: 205.5920
    Epoch [31/50], Test Losses: mse: 34.1780, mae: 3.1547, huber: 2.7350, swd: 1.6747, ept: 228.3671
      Epoch 31 composite train-obj: 1.319327
            No improvement (3.2396), counter 2/5
    Epoch [32/50], Train Losses: mse: 11.5780, mae: 1.6739, huber: 1.2948, swd: 0.8414, ept: 277.7069
    Epoch [32/50], Val Losses: mse: 40.3381, mae: 3.6024, huber: 3.1723, swd: 2.0499, ept: 207.1783
    Epoch [32/50], Test Losses: mse: 32.4862, mae: 3.0936, huber: 2.6733, swd: 1.6062, ept: 225.6864
      Epoch 32 composite train-obj: 1.294752
            No improvement (3.1723), counter 3/5
    Epoch [33/50], Train Losses: mse: 12.0721, mae: 1.7129, huber: 1.3305, swd: 0.8617, ept: 277.5213
    Epoch [33/50], Val Losses: mse: 39.0793, mae: 3.5344, huber: 3.1045, swd: 2.0054, ept: 208.6268
    Epoch [33/50], Test Losses: mse: 30.7907, mae: 3.0103, huber: 2.5907, swd: 1.6643, ept: 229.8198
      Epoch 33 composite train-obj: 1.330516
            No improvement (3.1045), counter 4/5
    Epoch [34/50], Train Losses: mse: 12.0161, mae: 1.7049, huber: 1.3233, swd: 0.8418, ept: 278.1220
    Epoch [34/50], Val Losses: mse: 43.0835, mae: 3.7035, huber: 3.2728, swd: 2.0420, ept: 207.5470
    Epoch [34/50], Test Losses: mse: 32.0056, mae: 3.0622, huber: 2.6423, swd: 1.6456, ept: 230.3117
      Epoch 34 composite train-obj: 1.323254
    Epoch [34/50], Test Losses: mse: 33.0205, mae: 3.1419, huber: 2.7207, swd: 1.8368, ept: 220.7944
    Best round's Test MSE: 33.0172, MAE: 3.1418, SWD: 1.8369
    Best round's Validation MSE: 39.2120, MAE: 3.5322, SWD: 2.0985
    Best round's Test verification MSE : 33.0205, MAE: 3.1419, SWD: 1.8368
    Time taken: 113.67 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.1825, mae: 6.5828, huber: 6.1031, swd: 43.2066, ept: 42.1326
    Epoch [1/50], Val Losses: mse: 59.4109, mae: 5.7362, huber: 5.2613, swd: 20.6255, ept: 67.0156
    Epoch [1/50], Test Losses: mse: 54.5948, mae: 5.3257, huber: 4.8541, swd: 21.0911, ept: 67.8233
      Epoch 1 composite train-obj: 6.103082
            Val objective improved inf → 5.2613, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.3898, mae: 4.7166, huber: 4.2529, swd: 14.4853, ept: 116.9686
    Epoch [2/50], Val Losses: mse: 49.6357, mae: 4.9662, huber: 4.5016, swd: 13.1550, ept: 109.9195
    Epoch [2/50], Test Losses: mse: 44.8246, mae: 4.5987, huber: 4.1356, swd: 14.1297, ept: 111.9171
      Epoch 2 composite train-obj: 4.252907
            Val objective improved 5.2613 → 4.5016, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.1022, mae: 4.0780, huber: 3.6255, swd: 9.0946, ept: 152.5013
    Epoch [3/50], Val Losses: mse: 50.7627, mae: 4.8869, huber: 4.4228, swd: 9.6646, ept: 118.2117
    Epoch [3/50], Test Losses: mse: 44.1169, mae: 4.4059, huber: 3.9453, swd: 9.0559, ept: 128.5187
      Epoch 3 composite train-obj: 3.625532
            Val objective improved 4.5016 → 4.4228, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.2884, mae: 3.7394, huber: 3.2937, swd: 6.6958, ept: 171.9621
    Epoch [4/50], Val Losses: mse: 49.5016, mae: 4.7073, huber: 4.2481, swd: 7.3546, ept: 132.2165
    Epoch [4/50], Test Losses: mse: 41.3397, mae: 4.1640, huber: 3.7087, swd: 6.6396, ept: 143.9776
      Epoch 4 composite train-obj: 3.293686
            Val objective improved 4.4228 → 4.2481, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.2125, mae: 3.4676, huber: 3.0281, swd: 5.3386, ept: 186.0756
    Epoch [5/50], Val Losses: mse: 47.8746, mae: 4.5545, huber: 4.0975, swd: 7.0198, ept: 140.4792
    Epoch [5/50], Test Losses: mse: 39.7477, mae: 4.0360, huber: 3.5838, swd: 6.7020, ept: 154.2193
      Epoch 5 composite train-obj: 3.028079
            Val objective improved 4.2481 → 4.0975, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.8394, mae: 3.2666, huber: 2.8319, swd: 4.5396, ept: 195.9046
    Epoch [6/50], Val Losses: mse: 49.2227, mae: 4.5189, huber: 4.0662, swd: 5.6609, ept: 147.3392
    Epoch [6/50], Test Losses: mse: 36.9646, mae: 3.8070, huber: 3.3605, swd: 5.4806, ept: 162.2261
      Epoch 6 composite train-obj: 2.831912
            Val objective improved 4.0975 → 4.0662, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.9543, mae: 3.1073, huber: 2.6762, swd: 3.8371, ept: 204.2043
    Epoch [7/50], Val Losses: mse: 48.5378, mae: 4.4236, huber: 3.9736, swd: 5.0229, ept: 153.2985
    Epoch [7/50], Test Losses: mse: 38.3102, mae: 3.8258, huber: 3.3801, swd: 4.8638, ept: 170.5064
      Epoch 7 composite train-obj: 2.676158
            Val objective improved 4.0662 → 3.9736, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.7411, mae: 2.9314, huber: 2.5048, swd: 3.3256, ept: 213.2213
    Epoch [8/50], Val Losses: mse: 47.7814, mae: 4.4001, huber: 3.9497, swd: 4.9193, ept: 158.5105
    Epoch [8/50], Test Losses: mse: 35.8624, mae: 3.6626, huber: 3.2201, swd: 4.2435, ept: 175.7895
      Epoch 8 composite train-obj: 2.504752
            Val objective improved 3.9736 → 3.9497, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.8747, mae: 2.7859, huber: 2.3627, swd: 2.8802, ept: 220.3413
    Epoch [9/50], Val Losses: mse: 46.5708, mae: 4.2071, huber: 3.7620, swd: 4.2555, ept: 165.9691
    Epoch [9/50], Test Losses: mse: 34.1219, mae: 3.4791, huber: 3.0425, swd: 3.7108, ept: 184.9974
      Epoch 9 composite train-obj: 2.362736
            Val objective improved 3.9497 → 3.7620, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.0653, mae: 2.7219, huber: 2.2995, swd: 2.6398, ept: 224.3100
    Epoch [10/50], Val Losses: mse: 46.1158, mae: 4.1695, huber: 3.7232, swd: 3.8863, ept: 170.7525
    Epoch [10/50], Test Losses: mse: 34.7690, mae: 3.5041, huber: 3.0652, swd: 3.5319, ept: 189.3385
      Epoch 10 composite train-obj: 2.299516
            Val objective improved 3.7620 → 3.7232, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.1173, mae: 2.5470, huber: 2.1312, swd: 2.3105, ept: 231.9594
    Epoch [11/50], Val Losses: mse: 43.5020, mae: 3.9953, huber: 3.5538, swd: 3.5513, ept: 173.8382
    Epoch [11/50], Test Losses: mse: 34.3156, mae: 3.4569, huber: 3.0196, swd: 3.0984, ept: 189.0174
      Epoch 11 composite train-obj: 2.131177
            Val objective improved 3.7232 → 3.5538, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 20.2832, mae: 2.4806, huber: 2.0662, swd: 2.1417, ept: 235.8127
    Epoch [12/50], Val Losses: mse: 45.3045, mae: 4.0717, huber: 3.6288, swd: 3.4125, ept: 173.2932
    Epoch [12/50], Test Losses: mse: 35.2913, mae: 3.4408, huber: 3.0077, swd: 2.7848, ept: 197.7958
      Epoch 12 composite train-obj: 2.066232
            No improvement (3.6288), counter 1/5
    Epoch [13/50], Train Losses: mse: 18.9346, mae: 2.3637, huber: 1.9536, swd: 1.9077, ept: 241.3392
    Epoch [13/50], Val Losses: mse: 42.9904, mae: 3.8885, huber: 3.4501, swd: 3.2333, ept: 181.5662
    Epoch [13/50], Test Losses: mse: 33.1077, mae: 3.3209, huber: 2.8898, swd: 2.8516, ept: 201.2578
      Epoch 13 composite train-obj: 1.953636
            Val objective improved 3.5538 → 3.4501, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.3403, mae: 2.3202, huber: 1.9108, swd: 1.7777, ept: 244.0468
    Epoch [14/50], Val Losses: mse: 42.5356, mae: 3.8731, huber: 3.4362, swd: 3.3768, ept: 178.5732
    Epoch [14/50], Test Losses: mse: 32.6394, mae: 3.2711, huber: 2.8421, swd: 2.6720, ept: 202.3176
      Epoch 14 composite train-obj: 1.910762
            Val objective improved 3.4501 → 3.4362, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.8503, mae: 2.2690, huber: 1.8620, swd: 1.6636, ept: 246.8423
    Epoch [15/50], Val Losses: mse: 43.8864, mae: 3.9311, huber: 3.4895, swd: 3.2153, ept: 184.7206
    Epoch [15/50], Test Losses: mse: 32.4346, mae: 3.2731, huber: 2.8396, swd: 2.4929, ept: 203.9400
      Epoch 15 composite train-obj: 1.861972
            No improvement (3.4895), counter 1/5
    Epoch [16/50], Train Losses: mse: 17.0862, mae: 2.2067, huber: 1.8011, swd: 1.5693, ept: 250.7887
    Epoch [16/50], Val Losses: mse: 42.8135, mae: 3.8612, huber: 3.4205, swd: 2.8272, ept: 187.3548
    Epoch [16/50], Test Losses: mse: 32.1356, mae: 3.2344, huber: 2.8024, swd: 2.4235, ept: 209.9138
      Epoch 16 composite train-obj: 1.801121
            Val objective improved 3.4362 → 3.4205, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 16.8398, mae: 2.1761, huber: 1.7722, swd: 1.5017, ept: 252.3101
    Epoch [17/50], Val Losses: mse: 39.4410, mae: 3.6658, huber: 3.2311, swd: 2.6979, ept: 188.4252
    Epoch [17/50], Test Losses: mse: 33.6724, mae: 3.3052, huber: 2.8752, swd: 2.2944, ept: 208.6708
      Epoch 17 composite train-obj: 1.772195
            Val objective improved 3.4205 → 3.2311, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.9837, mae: 2.0997, huber: 1.6994, swd: 1.3917, ept: 256.4137
    Epoch [18/50], Val Losses: mse: 41.6964, mae: 3.8019, huber: 3.3646, swd: 2.7254, ept: 189.5959
    Epoch [18/50], Test Losses: mse: 33.2788, mae: 3.2909, huber: 2.8606, swd: 2.3308, ept: 207.3654
      Epoch 18 composite train-obj: 1.699378
            No improvement (3.3646), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.8230, mae: 1.9969, huber: 1.6009, swd: 1.2694, ept: 260.2802
    Epoch [19/50], Val Losses: mse: 42.6662, mae: 3.7794, huber: 3.3448, swd: 2.7001, ept: 187.0764
    Epoch [19/50], Test Losses: mse: 33.6049, mae: 3.2279, huber: 2.8020, swd: 2.1720, ept: 211.8333
      Epoch 19 composite train-obj: 1.600900
            No improvement (3.3448), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.5843, mae: 1.9740, huber: 1.5782, swd: 1.2230, ept: 262.5023
    Epoch [20/50], Val Losses: mse: 46.2380, mae: 3.9214, huber: 3.4867, swd: 2.9358, ept: 190.6879
    Epoch [20/50], Test Losses: mse: 33.2682, mae: 3.2330, huber: 2.8071, swd: 2.2408, ept: 212.2749
      Epoch 20 composite train-obj: 1.578171
            No improvement (3.4867), counter 3/5
    Epoch [21/50], Train Losses: mse: 15.0199, mae: 2.0143, huber: 1.6164, swd: 1.2477, ept: 261.2466
    Epoch [21/50], Val Losses: mse: 47.2111, mae: 3.9555, huber: 3.5218, swd: 2.9592, ept: 189.3667
    Epoch [21/50], Test Losses: mse: 35.2898, mae: 3.3274, huber: 2.8993, swd: 2.3289, ept: 211.0705
      Epoch 21 composite train-obj: 1.616381
            No improvement (3.5218), counter 4/5
    Epoch [22/50], Train Losses: mse: 14.4074, mae: 1.9418, huber: 1.5482, swd: 1.1606, ept: 264.2198
    Epoch [22/50], Val Losses: mse: 40.8819, mae: 3.6809, huber: 3.2464, swd: 2.6759, ept: 197.7781
    Epoch [22/50], Test Losses: mse: 31.3187, mae: 3.1271, huber: 2.7014, swd: 2.0429, ept: 219.1647
      Epoch 22 composite train-obj: 1.548236
    Epoch [22/50], Test Losses: mse: 33.6734, mae: 3.3053, huber: 2.8753, swd: 2.2944, ept: 208.6867
    Best round's Test MSE: 33.6724, MAE: 3.3052, SWD: 2.2944
    Best round's Validation MSE: 39.4410, MAE: 3.6658, SWD: 2.6979
    Best round's Test verification MSE : 33.6734, MAE: 3.3053, SWD: 2.2944
    Time taken: 71.92 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1627)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 33.0196 ± 0.5320
      mae: 3.1836 ± 0.0874
      huber: 2.7597 ± 0.0831
      swd: 1.9593 ± 0.2398
      ept: 218.1164 ± 6.8814
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 39.9202 ± 0.8448
      mae: 3.6152 ± 0.0591
      huber: 3.1838 ± 0.0568
      swd: 2.3749 ± 0.2469
      ept: 199.8134 ± 8.0766
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 288.04 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1627
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: OT dissect: Scale only for OT


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
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, 
    ablate_scale_only=True, ### HERE
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
    
    Epoch [1/50], Train Losses: mse: 74.8227, mae: 6.5306, huber: 6.0503, swd: 44.1291, ept: 32.1584
    Epoch [1/50], Val Losses: mse: 61.1451, mae: 5.9539, huber: 5.4761, swd: 27.3112, ept: 37.3888
    Epoch [1/50], Test Losses: mse: 56.3111, mae: 5.5624, huber: 5.0880, swd: 29.2554, ept: 38.5528
      Epoch 1 composite train-obj: 6.050285
            Val objective improved inf → 5.4761, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.4940, mae: 4.9555, huber: 4.4887, swd: 18.0068, ept: 90.2718
    Epoch [2/50], Val Losses: mse: 51.8057, mae: 5.1256, huber: 4.6581, swd: 15.3433, ept: 87.3252
    Epoch [2/50], Test Losses: mse: 46.8706, mae: 4.7644, huber: 4.3008, swd: 16.8792, ept: 80.6044
      Epoch 2 composite train-obj: 4.488705
            Val objective improved 5.4761 → 4.6581, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.1590, mae: 4.2154, huber: 3.7605, swd: 10.0484, ept: 136.6766
    Epoch [3/50], Val Losses: mse: 47.8687, mae: 4.7277, huber: 4.2658, swd: 10.0589, ept: 111.6853
    Epoch [3/50], Test Losses: mse: 42.9353, mae: 4.4012, huber: 3.9412, swd: 10.0575, ept: 113.5190
      Epoch 3 composite train-obj: 3.760531
            Val objective improved 4.6581 → 4.2658, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.2442, mae: 3.8083, huber: 3.3601, swd: 6.5814, ept: 162.3511
    Epoch [4/50], Val Losses: mse: 45.8169, mae: 4.4522, huber: 3.9954, swd: 7.0268, ept: 128.8004
    Epoch [4/50], Test Losses: mse: 40.2490, mae: 4.1315, huber: 3.6761, swd: 7.6279, ept: 130.3408
      Epoch 4 composite train-obj: 3.360082
            Val objective improved 4.2658 → 3.9954, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 33.2315, mae: 3.5198, huber: 3.0790, swd: 5.0850, ept: 179.6830
    Epoch [5/50], Val Losses: mse: 42.8417, mae: 4.2056, huber: 3.7534, swd: 6.6808, ept: 142.9069
    Epoch [5/50], Test Losses: mse: 37.4068, mae: 3.8417, huber: 3.3939, swd: 6.1867, ept: 146.8907
      Epoch 5 composite train-obj: 3.078985
            Val objective improved 3.9954 → 3.7534, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.6731, mae: 3.3059, huber: 2.8704, swd: 4.1568, ept: 189.0126
    Epoch [6/50], Val Losses: mse: 41.1441, mae: 4.1045, huber: 3.6526, swd: 5.5359, ept: 152.5167
    Epoch [6/50], Test Losses: mse: 34.9795, mae: 3.6846, huber: 3.2382, swd: 4.8225, ept: 164.6171
      Epoch 6 composite train-obj: 2.870367
            Val objective improved 3.7534 → 3.6526, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.6584, mae: 3.1352, huber: 2.7039, swd: 3.5394, ept: 195.7068
    Epoch [7/50], Val Losses: mse: 43.9309, mae: 4.1486, huber: 3.7019, swd: 5.3575, ept: 152.9045
    Epoch [7/50], Test Losses: mse: 34.1494, mae: 3.5845, huber: 3.1421, swd: 4.8369, ept: 162.1756
      Epoch 7 composite train-obj: 2.703887
            No improvement (3.7019), counter 1/5
    Epoch [8/50], Train Losses: mse: 27.1939, mae: 3.0234, huber: 2.5943, swd: 3.1593, ept: 200.4113
    Epoch [8/50], Val Losses: mse: 40.3432, mae: 3.9281, huber: 3.4844, swd: 5.0922, ept: 159.0567
    Epoch [8/50], Test Losses: mse: 32.5047, mae: 3.4551, huber: 3.0158, swd: 4.2279, ept: 169.7299
      Epoch 8 composite train-obj: 2.594284
            Val objective improved 3.6526 → 3.4844, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.8861, mae: 2.9320, huber: 2.5041, swd: 2.8113, ept: 204.3682
    Epoch [9/50], Val Losses: mse: 41.4437, mae: 3.9799, huber: 3.5339, swd: 5.0866, ept: 161.7135
    Epoch [9/50], Test Losses: mse: 32.0267, mae: 3.3973, huber: 2.9581, swd: 4.0474, ept: 176.8814
      Epoch 9 composite train-obj: 2.504065
            No improvement (3.5339), counter 1/5
    Epoch [10/50], Train Losses: mse: 24.3746, mae: 2.8085, huber: 2.3844, swd: 2.5154, ept: 210.3831
    Epoch [10/50], Val Losses: mse: 42.5242, mae: 3.9428, huber: 3.5021, swd: 3.9622, ept: 162.6592
    Epoch [10/50], Test Losses: mse: 31.8827, mae: 3.3558, huber: 2.9192, swd: 3.5000, ept: 175.8262
      Epoch 10 composite train-obj: 2.384436
            No improvement (3.5021), counter 2/5
    Epoch [11/50], Train Losses: mse: 23.0079, mae: 2.6971, huber: 2.2761, swd: 2.3222, ept: 216.6024
    Epoch [11/50], Val Losses: mse: 43.1444, mae: 3.9362, huber: 3.4970, swd: 3.7076, ept: 174.2432
    Epoch [11/50], Test Losses: mse: 32.3388, mae: 3.3149, huber: 2.8800, swd: 2.8253, ept: 189.8588
      Epoch 11 composite train-obj: 2.276147
            No improvement (3.4970), counter 3/5
    Epoch [12/50], Train Losses: mse: 21.8053, mae: 2.5909, huber: 2.1741, swd: 2.0857, ept: 222.0686
    Epoch [12/50], Val Losses: mse: 41.6181, mae: 3.8384, huber: 3.4013, swd: 3.5215, ept: 176.4783
    Epoch [12/50], Test Losses: mse: 31.2034, mae: 3.2313, huber: 2.7981, swd: 2.7664, ept: 194.4461
      Epoch 12 composite train-obj: 2.174146
            Val objective improved 3.4844 → 3.4013, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 21.0175, mae: 2.5372, huber: 2.1212, swd: 1.9433, ept: 225.3016
    Epoch [13/50], Val Losses: mse: 44.4694, mae: 3.9433, huber: 3.5061, swd: 4.0826, ept: 173.3544
    Epoch [13/50], Test Losses: mse: 31.2886, mae: 3.2178, huber: 2.7864, swd: 3.1019, ept: 197.8621
      Epoch 13 composite train-obj: 2.121248
            No improvement (3.5061), counter 1/5
    Epoch [14/50], Train Losses: mse: 19.9879, mae: 2.4445, huber: 2.0321, swd: 1.8090, ept: 229.7608
    Epoch [14/50], Val Losses: mse: 45.0188, mae: 3.9954, huber: 3.5537, swd: 4.2556, ept: 175.1203
    Epoch [14/50], Test Losses: mse: 30.2928, mae: 3.2187, huber: 2.7837, swd: 2.9860, ept: 189.7751
      Epoch 14 composite train-obj: 2.032078
            No improvement (3.5537), counter 2/5
    Epoch [15/50], Train Losses: mse: 19.5224, mae: 2.4090, huber: 1.9973, swd: 1.7257, ept: 232.4952
    Epoch [15/50], Val Losses: mse: 38.6080, mae: 3.6583, huber: 3.2231, swd: 3.4323, ept: 182.2260
    Epoch [15/50], Test Losses: mse: 28.7897, mae: 3.0483, huber: 2.6211, swd: 2.6002, ept: 201.0674
      Epoch 15 composite train-obj: 1.997309
            Val objective improved 3.4013 → 3.2231, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 18.4493, mae: 2.3253, huber: 1.9164, swd: 1.5882, ept: 236.3334
    Epoch [16/50], Val Losses: mse: 38.0166, mae: 3.5699, huber: 3.1382, swd: 3.1997, ept: 187.7630
    Epoch [16/50], Test Losses: mse: 28.0329, mae: 2.9758, huber: 2.5505, swd: 2.2050, ept: 208.5695
      Epoch 16 composite train-obj: 1.916424
            Val objective improved 3.2231 → 3.1382, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 17.3848, mae: 2.2222, huber: 1.8180, swd: 1.4527, ept: 241.5897
    Epoch [17/50], Val Losses: mse: 39.9661, mae: 3.6421, huber: 3.2099, swd: 3.3706, ept: 190.4222
    Epoch [17/50], Test Losses: mse: 27.3315, mae: 2.9264, huber: 2.5020, swd: 2.3848, ept: 209.9608
      Epoch 17 composite train-obj: 1.818007
            No improvement (3.2099), counter 1/5
    Epoch [18/50], Train Losses: mse: 16.7484, mae: 2.1763, huber: 1.7726, swd: 1.3905, ept: 244.1784
    Epoch [18/50], Val Losses: mse: 39.5469, mae: 3.5969, huber: 3.1657, swd: 2.9475, ept: 192.2176
    Epoch [18/50], Test Losses: mse: 27.5947, mae: 2.8999, huber: 2.4773, swd: 2.3342, ept: 212.3333
      Epoch 18 composite train-obj: 1.772649
            No improvement (3.1657), counter 2/5
    Epoch [19/50], Train Losses: mse: 15.7912, mae: 2.0846, huber: 1.6858, swd: 1.2719, ept: 248.8754
    Epoch [19/50], Val Losses: mse: 40.5872, mae: 3.5817, huber: 3.1525, swd: 2.6587, ept: 193.9996
    Epoch [19/50], Test Losses: mse: 28.5434, mae: 2.9147, huber: 2.4936, swd: 1.9136, ept: 217.2272
      Epoch 19 composite train-obj: 1.685780
            No improvement (3.1525), counter 3/5
    Epoch [20/50], Train Losses: mse: 16.0383, mae: 2.1098, huber: 1.7086, swd: 1.2853, ept: 249.0949
    Epoch [20/50], Val Losses: mse: 40.0231, mae: 3.6386, huber: 3.2055, swd: 3.0432, ept: 191.6260
    Epoch [20/50], Test Losses: mse: 27.1533, mae: 2.8900, huber: 2.4658, swd: 2.2903, ept: 214.1159
      Epoch 20 composite train-obj: 1.708643
            No improvement (3.2055), counter 4/5
    Epoch [21/50], Train Losses: mse: 15.1207, mae: 2.0294, huber: 1.6325, swd: 1.2131, ept: 252.4206
    Epoch [21/50], Val Losses: mse: 38.5674, mae: 3.5145, huber: 3.0847, swd: 2.8636, ept: 193.3497
    Epoch [21/50], Test Losses: mse: 29.3687, mae: 2.9732, huber: 2.5497, swd: 2.3026, ept: 215.3598
      Epoch 21 composite train-obj: 1.632526
            Val objective improved 3.1382 → 3.0847, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 14.6627, mae: 1.9884, huber: 1.5926, swd: 1.1482, ept: 255.2127
    Epoch [22/50], Val Losses: mse: 38.4350, mae: 3.4829, huber: 3.0570, swd: 2.9312, ept: 199.4916
    Epoch [22/50], Test Losses: mse: 26.5724, mae: 2.7955, huber: 2.3770, swd: 1.9923, ept: 225.8213
      Epoch 22 composite train-obj: 1.592642
            Val objective improved 3.0847 → 3.0570, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 13.9073, mae: 1.9061, huber: 1.5156, swd: 1.0789, ept: 259.7985
    Epoch [23/50], Val Losses: mse: 38.7259, mae: 3.4916, huber: 3.0629, swd: 2.5681, ept: 199.5406
    Epoch [23/50], Test Losses: mse: 28.2462, mae: 2.8689, huber: 2.4483, swd: 1.8227, ept: 221.7323
      Epoch 23 composite train-obj: 1.515581
            No improvement (3.0629), counter 1/5
    Epoch [24/50], Train Losses: mse: 14.5328, mae: 1.9710, huber: 1.5762, swd: 1.1174, ept: 257.0907
    Epoch [24/50], Val Losses: mse: 38.4570, mae: 3.5210, huber: 3.0919, swd: 2.7065, ept: 192.7374
    Epoch [24/50], Test Losses: mse: 27.2780, mae: 2.8481, huber: 2.4284, swd: 1.8412, ept: 219.3806
      Epoch 24 composite train-obj: 1.576157
            No improvement (3.0919), counter 2/5
    Epoch [25/50], Train Losses: mse: 14.2673, mae: 1.9366, huber: 1.5442, swd: 1.1110, ept: 259.2870
    Epoch [25/50], Val Losses: mse: 38.6234, mae: 3.4936, huber: 3.0667, swd: 2.4856, ept: 200.9085
    Epoch [25/50], Test Losses: mse: 30.7996, mae: 3.0085, huber: 2.5866, swd: 1.8237, ept: 220.2821
      Epoch 25 composite train-obj: 1.544246
            No improvement (3.0667), counter 3/5
    Epoch [26/50], Train Losses: mse: 13.2884, mae: 1.8481, huber: 1.4598, swd: 1.0164, ept: 263.7817
    Epoch [26/50], Val Losses: mse: 40.2234, mae: 3.5519, huber: 3.1266, swd: 2.6506, ept: 195.4738
    Epoch [26/50], Test Losses: mse: 26.6779, mae: 2.7504, huber: 2.3347, swd: 1.9287, ept: 230.3991
      Epoch 26 composite train-obj: 1.459828
            No improvement (3.1266), counter 4/5
    Epoch [27/50], Train Losses: mse: 13.5579, mae: 1.8758, huber: 1.4852, swd: 1.0549, ept: 263.4305
    Epoch [27/50], Val Losses: mse: 38.3501, mae: 3.3858, huber: 2.9647, swd: 2.5894, ept: 207.3401
    Epoch [27/50], Test Losses: mse: 26.1400, mae: 2.6770, huber: 2.2643, swd: 1.6688, ept: 235.0113
      Epoch 27 composite train-obj: 1.485154
            Val objective improved 3.0570 → 2.9647, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 12.9625, mae: 1.8132, huber: 1.4270, swd: 0.9649, ept: 266.3099
    Epoch [28/50], Val Losses: mse: 38.3039, mae: 3.4407, huber: 3.0165, swd: 2.8339, ept: 205.5653
    Epoch [28/50], Test Losses: mse: 26.7993, mae: 2.7834, huber: 2.3679, swd: 1.9183, ept: 228.0982
      Epoch 28 composite train-obj: 1.427040
            No improvement (3.0165), counter 1/5
    Epoch [29/50], Train Losses: mse: 13.0556, mae: 1.8219, huber: 1.4348, swd: 0.9743, ept: 266.3767
    Epoch [29/50], Val Losses: mse: 38.7312, mae: 3.4760, huber: 3.0479, swd: 2.7921, ept: 201.1202
    Epoch [29/50], Test Losses: mse: 29.3268, mae: 2.8757, huber: 2.4578, swd: 1.8484, ept: 226.1274
      Epoch 29 composite train-obj: 1.434810
            No improvement (3.0479), counter 2/5
    Epoch [30/50], Train Losses: mse: 12.9533, mae: 1.8121, huber: 1.4251, swd: 0.9795, ept: 267.0916
    Epoch [30/50], Val Losses: mse: 37.8155, mae: 3.4153, huber: 2.9900, swd: 2.1325, ept: 200.4920
    Epoch [30/50], Test Losses: mse: 26.9630, mae: 2.8025, huber: 2.3849, swd: 1.8044, ept: 223.0701
      Epoch 30 composite train-obj: 1.425106
            No improvement (2.9900), counter 3/5
    Epoch [31/50], Train Losses: mse: 12.3645, mae: 1.7546, huber: 1.3717, swd: 0.9139, ept: 269.5279
    Epoch [31/50], Val Losses: mse: 37.2491, mae: 3.3663, huber: 2.9416, swd: 2.3209, ept: 203.7285
    Epoch [31/50], Test Losses: mse: 26.6515, mae: 2.7614, huber: 2.3450, swd: 1.8122, ept: 228.7340
      Epoch 31 composite train-obj: 1.371736
            Val objective improved 2.9647 → 2.9416, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 11.6488, mae: 1.6858, huber: 1.3073, swd: 0.8635, ept: 272.8567
    Epoch [32/50], Val Losses: mse: 39.1282, mae: 3.4270, huber: 3.0049, swd: 2.4699, ept: 205.4071
    Epoch [32/50], Test Losses: mse: 28.0402, mae: 2.7700, huber: 2.3584, swd: 1.7687, ept: 232.9981
      Epoch 32 composite train-obj: 1.307267
            No improvement (3.0049), counter 1/5
    Epoch [33/50], Train Losses: mse: 12.6432, mae: 1.7659, huber: 1.3824, swd: 0.9366, ept: 270.2740
    Epoch [33/50], Val Losses: mse: 39.1962, mae: 3.4262, huber: 3.0046, swd: 2.6608, ept: 206.0854
    Epoch [33/50], Test Losses: mse: 28.8082, mae: 2.8500, huber: 2.4350, swd: 1.8695, ept: 227.3275
      Epoch 33 composite train-obj: 1.382394
            No improvement (3.0046), counter 2/5
    Epoch [34/50], Train Losses: mse: 12.0628, mae: 1.7155, huber: 1.3347, swd: 0.8947, ept: 272.6227
    Epoch [34/50], Val Losses: mse: 35.6707, mae: 3.3492, huber: 2.9266, swd: 2.5204, ept: 204.1141
    Epoch [34/50], Test Losses: mse: 27.5787, mae: 2.7649, huber: 2.3551, swd: 1.8081, ept: 231.2798
      Epoch 34 composite train-obj: 1.334677
            Val objective improved 2.9416 → 2.9266, saving checkpoint.
    Epoch [35/50], Train Losses: mse: 11.6879, mae: 1.6789, huber: 1.3010, swd: 0.8696, ept: 274.3391
    Epoch [35/50], Val Losses: mse: 40.3758, mae: 3.4848, huber: 3.0670, swd: 2.6513, ept: 211.5817
    Epoch [35/50], Test Losses: mse: 28.3136, mae: 2.7524, huber: 2.3449, swd: 1.5656, ept: 234.7313
      Epoch 35 composite train-obj: 1.301019
            No improvement (3.0670), counter 1/5
    Epoch [36/50], Train Losses: mse: 11.5705, mae: 1.6590, huber: 1.2827, swd: 0.8480, ept: 275.3532
    Epoch [36/50], Val Losses: mse: 37.5800, mae: 3.4049, huber: 2.9820, swd: 2.6585, ept: 207.2245
    Epoch [36/50], Test Losses: mse: 25.7650, mae: 2.6987, huber: 2.2857, swd: 1.6839, ept: 233.0501
      Epoch 36 composite train-obj: 1.282717
            No improvement (2.9820), counter 2/5
    Epoch [37/50], Train Losses: mse: 11.5667, mae: 1.6782, huber: 1.2998, swd: 0.8250, ept: 274.0717
    Epoch [37/50], Val Losses: mse: 40.2808, mae: 3.5586, huber: 3.1317, swd: 2.3557, ept: 197.0889
    Epoch [37/50], Test Losses: mse: 32.4445, mae: 3.0498, huber: 2.6298, swd: 1.7009, ept: 220.5476
      Epoch 37 composite train-obj: 1.299757
            No improvement (3.1317), counter 3/5
    Epoch [38/50], Train Losses: mse: 11.0944, mae: 1.6125, huber: 1.2388, swd: 0.8110, ept: 278.3780
    Epoch [38/50], Val Losses: mse: 37.0137, mae: 3.2983, huber: 2.8840, swd: 2.3183, ept: 214.5739
    Epoch [38/50], Test Losses: mse: 26.2103, mae: 2.6318, huber: 2.2280, swd: 1.5828, ept: 240.7177
      Epoch 38 composite train-obj: 1.238830
            Val objective improved 2.9266 → 2.8840, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 10.9047, mae: 1.5972, huber: 1.2249, swd: 0.7913, ept: 278.4213
    Epoch [39/50], Val Losses: mse: 35.7169, mae: 3.2399, huber: 2.8207, swd: 2.0364, ept: 212.7087
    Epoch [39/50], Test Losses: mse: 28.2876, mae: 2.7704, huber: 2.3603, swd: 1.4665, ept: 231.9493
      Epoch 39 composite train-obj: 1.224876
            Val objective improved 2.8840 → 2.8207, saving checkpoint.
    Epoch [40/50], Train Losses: mse: 10.4634, mae: 1.5459, huber: 1.1773, swd: 0.7340, ept: 281.1657
    Epoch [40/50], Val Losses: mse: 37.3129, mae: 3.3220, huber: 2.9032, swd: 2.4870, ept: 210.5399
    Epoch [40/50], Test Losses: mse: 26.2288, mae: 2.6368, huber: 2.2316, swd: 1.6475, ept: 238.4576
      Epoch 40 composite train-obj: 1.177256
            No improvement (2.9032), counter 1/5
    Epoch [41/50], Train Losses: mse: 10.2691, mae: 1.5384, huber: 1.1694, swd: 0.7319, ept: 281.4099
    Epoch [41/50], Val Losses: mse: 39.5898, mae: 3.4781, huber: 3.0556, swd: 2.4117, ept: 204.8900
    Epoch [41/50], Test Losses: mse: 27.7071, mae: 2.7758, huber: 2.3627, swd: 1.5203, ept: 233.1188
      Epoch 41 composite train-obj: 1.169445
            No improvement (3.0556), counter 2/5
    Epoch [42/50], Train Losses: mse: 10.2099, mae: 1.5168, huber: 1.1499, swd: 0.7260, ept: 283.5797
    Epoch [42/50], Val Losses: mse: 36.6823, mae: 3.3266, huber: 2.9097, swd: 2.4893, ept: 209.8304
    Epoch [42/50], Test Losses: mse: 24.7884, mae: 2.5667, huber: 2.1647, swd: 1.5529, ept: 241.1231
      Epoch 42 composite train-obj: 1.149892
            No improvement (2.9097), counter 3/5
    Epoch [43/50], Train Losses: mse: 10.4708, mae: 1.5484, huber: 1.1788, swd: 0.7541, ept: 281.8412
    Epoch [43/50], Val Losses: mse: 37.9857, mae: 3.4160, huber: 2.9938, swd: 2.5479, ept: 207.4274
    Epoch [43/50], Test Losses: mse: 27.6077, mae: 2.7605, huber: 2.3490, swd: 1.5884, ept: 234.5730
      Epoch 43 composite train-obj: 1.178795
            No improvement (2.9938), counter 4/5
    Epoch [44/50], Train Losses: mse: 9.7306, mae: 1.4876, huber: 1.1214, swd: 0.7031, ept: 283.8678
    Epoch [44/50], Val Losses: mse: 40.3274, mae: 3.4510, huber: 3.0333, swd: 2.3810, ept: 209.2958
    Epoch [44/50], Test Losses: mse: 28.5625, mae: 2.7549, huber: 2.3476, swd: 1.4058, ept: 237.2035
      Epoch 44 composite train-obj: 1.121359
    Epoch [44/50], Test Losses: mse: 28.3033, mae: 2.7710, huber: 2.3609, swd: 1.4651, ept: 231.9424
    Best round's Test MSE: 28.2876, MAE: 2.7704, SWD: 1.4665
    Best round's Validation MSE: 35.7169, MAE: 3.2399, SWD: 2.0364
    Best round's Test verification MSE : 28.3033, MAE: 2.7710, SWD: 1.4651
    Time taken: 158.97 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.5289, mae: 6.4694, huber: 5.9901, swd: 42.6836, ept: 35.8280
    Epoch [1/50], Val Losses: mse: 59.0577, mae: 5.7436, huber: 5.2682, swd: 23.3462, ept: 47.3890
    Epoch [1/50], Test Losses: mse: 54.6448, mae: 5.3909, huber: 4.9186, swd: 24.8998, ept: 44.4266
      Epoch 1 composite train-obj: 5.990078
            Val objective improved inf → 5.2682, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.6067, mae: 4.7837, huber: 4.3196, swd: 15.7634, ept: 96.4863
    Epoch [2/50], Val Losses: mse: 49.6750, mae: 4.8950, huber: 4.4309, swd: 13.5923, ept: 89.0115
    Epoch [2/50], Test Losses: mse: 46.6428, mae: 4.6424, huber: 4.1813, swd: 14.4279, ept: 84.4983
      Epoch 2 composite train-obj: 4.319642
            Val objective improved 5.2682 → 4.4309, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.1847, mae: 4.1424, huber: 3.6888, swd: 9.1338, ept: 131.5431
    Epoch [3/50], Val Losses: mse: 45.0508, mae: 4.4842, huber: 4.0262, swd: 8.0346, ept: 110.8931
    Epoch [3/50], Test Losses: mse: 41.1691, mae: 4.2305, huber: 3.7741, swd: 8.8841, ept: 114.5596
      Epoch 3 composite train-obj: 3.688766
            Val objective improved 4.4309 → 4.0262, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.2819, mae: 3.7100, huber: 3.2649, swd: 6.1161, ept: 161.8740
    Epoch [4/50], Val Losses: mse: 41.7529, mae: 4.2306, huber: 3.7776, swd: 6.8501, ept: 133.0215
    Epoch [4/50], Test Losses: mse: 37.9460, mae: 3.9775, huber: 3.5256, swd: 7.0577, ept: 141.1243
      Epoch 4 composite train-obj: 3.264910
            Val objective improved 4.0262 → 3.7776, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.3676, mae: 3.4656, huber: 3.0261, swd: 4.8857, ept: 178.5978
    Epoch [5/50], Val Losses: mse: 39.8833, mae: 4.0672, huber: 3.6173, swd: 5.8938, ept: 143.9758
    Epoch [5/50], Test Losses: mse: 35.3462, mae: 3.7692, huber: 3.3215, swd: 5.9048, ept: 152.8700
      Epoch 5 composite train-obj: 3.026104
            Val objective improved 3.7776 → 3.6173, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.1784, mae: 3.2805, huber: 2.8449, swd: 4.0530, ept: 188.7590
    Epoch [6/50], Val Losses: mse: 39.0456, mae: 3.9189, huber: 3.4723, swd: 4.8294, ept: 151.1732
    Epoch [6/50], Test Losses: mse: 33.4464, mae: 3.5757, huber: 3.1319, swd: 4.5820, ept: 159.9901
      Epoch 6 composite train-obj: 2.844910
            Val objective improved 3.6173 → 3.4723, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.1395, mae: 3.1129, huber: 2.6811, swd: 3.4203, ept: 196.9126
    Epoch [7/50], Val Losses: mse: 42.9137, mae: 4.0813, huber: 3.6323, swd: 4.2618, ept: 152.2942
    Epoch [7/50], Test Losses: mse: 33.7418, mae: 3.5681, huber: 3.1246, swd: 4.3893, ept: 162.7111
      Epoch 7 composite train-obj: 2.681054
            No improvement (3.6323), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.1406, mae: 2.9648, huber: 2.5355, swd: 2.9260, ept: 203.9436
    Epoch [8/50], Val Losses: mse: 41.4779, mae: 3.9350, huber: 3.4901, swd: 3.8058, ept: 163.1545
    Epoch [8/50], Test Losses: mse: 32.1455, mae: 3.4091, huber: 2.9690, swd: 3.5281, ept: 175.4931
      Epoch 8 composite train-obj: 2.535537
            No improvement (3.4901), counter 2/5
    Epoch [9/50], Train Losses: mse: 23.9011, mae: 2.7854, huber: 2.3605, swd: 2.5170, ept: 212.7262
    Epoch [9/50], Val Losses: mse: 44.1820, mae: 3.9896, huber: 3.5465, swd: 3.7973, ept: 162.0090
    Epoch [9/50], Test Losses: mse: 32.0202, mae: 3.3388, huber: 2.9007, swd: 3.0979, ept: 176.8506
      Epoch 9 composite train-obj: 2.360547
            No improvement (3.5465), counter 3/5
    Epoch [10/50], Train Losses: mse: 22.5026, mae: 2.6761, huber: 2.2540, swd: 2.2846, ept: 218.8440
    Epoch [10/50], Val Losses: mse: 38.2373, mae: 3.6900, huber: 3.2492, swd: 3.0963, ept: 167.8325
    Epoch [10/50], Test Losses: mse: 32.6979, mae: 3.3277, huber: 2.8927, swd: 3.1819, ept: 188.7562
      Epoch 10 composite train-obj: 2.253977
            Val objective improved 3.4723 → 3.2492, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.4395, mae: 2.5840, huber: 2.1647, swd: 2.0947, ept: 224.1399
    Epoch [11/50], Val Losses: mse: 44.1944, mae: 3.9091, huber: 3.4691, swd: 2.9739, ept: 174.7300
    Epoch [11/50], Test Losses: mse: 32.4219, mae: 3.2948, huber: 2.8595, swd: 2.9321, ept: 186.7542
      Epoch 11 composite train-obj: 2.164686
            No improvement (3.4691), counter 1/5
    Epoch [12/50], Train Losses: mse: 20.1542, mae: 2.4767, huber: 2.0612, swd: 1.8622, ept: 229.0531
    Epoch [12/50], Val Losses: mse: 42.8244, mae: 3.7788, huber: 3.3419, swd: 2.8325, ept: 181.2859
    Epoch [12/50], Test Losses: mse: 29.5754, mae: 3.1097, huber: 2.6783, swd: 2.8372, ept: 198.0184
      Epoch 12 composite train-obj: 2.061187
            No improvement (3.3419), counter 2/5
    Epoch [13/50], Train Losses: mse: 19.0911, mae: 2.3855, huber: 1.9725, swd: 1.7294, ept: 234.6427
    Epoch [13/50], Val Losses: mse: 38.8838, mae: 3.6284, huber: 3.1927, swd: 3.1390, ept: 181.3936
    Epoch [13/50], Test Losses: mse: 29.5032, mae: 3.0362, huber: 2.6092, swd: 2.6079, ept: 207.6985
      Epoch 13 composite train-obj: 1.972453
            Val objective improved 3.2492 → 3.1927, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.6813, mae: 2.3424, huber: 1.9312, swd: 1.6346, ept: 237.3542
    Epoch [14/50], Val Losses: mse: 40.7724, mae: 3.6677, huber: 3.2333, swd: 2.9047, ept: 187.2356
    Epoch [14/50], Test Losses: mse: 30.3753, mae: 3.0646, huber: 2.6374, swd: 2.3306, ept: 208.1989
      Epoch 14 composite train-obj: 1.931191
            No improvement (3.2333), counter 1/5
    Epoch [15/50], Train Losses: mse: 17.1562, mae: 2.2064, huber: 1.8010, swd: 1.4417, ept: 243.8354
    Epoch [15/50], Val Losses: mse: 39.4145, mae: 3.5964, huber: 3.1620, swd: 2.4999, ept: 184.5716
    Epoch [15/50], Test Losses: mse: 31.2320, mae: 3.1563, huber: 2.7256, swd: 2.3349, ept: 203.0444
      Epoch 15 composite train-obj: 1.801048
            Val objective improved 3.1927 → 3.1620, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 17.8604, mae: 2.2668, huber: 1.8583, swd: 1.4918, ept: 241.6407
    Epoch [16/50], Val Losses: mse: 41.1102, mae: 3.6792, huber: 3.2439, swd: 2.5961, ept: 184.0597
    Epoch [16/50], Test Losses: mse: 29.9995, mae: 3.0550, huber: 2.6261, swd: 2.2180, ept: 202.5787
      Epoch 16 composite train-obj: 1.858333
            No improvement (3.2439), counter 1/5
    Epoch [17/50], Train Losses: mse: 16.2441, mae: 2.1218, huber: 1.7196, swd: 1.3283, ept: 249.0267
    Epoch [17/50], Val Losses: mse: 40.6401, mae: 3.6148, huber: 3.1804, swd: 2.2279, ept: 186.4331
    Epoch [17/50], Test Losses: mse: 30.9550, mae: 3.0576, huber: 2.6304, swd: 2.1237, ept: 211.2481
      Epoch 17 composite train-obj: 1.719595
            No improvement (3.1804), counter 2/5
    Epoch [18/50], Train Losses: mse: 16.1605, mae: 2.1126, huber: 1.7107, swd: 1.3214, ept: 250.1802
    Epoch [18/50], Val Losses: mse: 39.4728, mae: 3.5393, huber: 3.1072, swd: 2.3855, ept: 185.8288
    Epoch [18/50], Test Losses: mse: 27.5277, mae: 2.9097, huber: 2.4851, swd: 2.4136, ept: 208.3956
      Epoch 18 composite train-obj: 1.710683
            Val objective improved 3.1620 → 3.1072, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.1719, mae: 2.0199, huber: 1.6226, swd: 1.2128, ept: 255.0511
    Epoch [19/50], Val Losses: mse: 38.7477, mae: 3.4750, huber: 3.0460, swd: 2.3129, ept: 193.1553
    Epoch [19/50], Test Losses: mse: 29.2123, mae: 2.9024, huber: 2.4825, swd: 2.0046, ept: 219.0225
      Epoch 19 composite train-obj: 1.622591
            Val objective improved 3.1072 → 3.0460, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 15.2687, mae: 2.0049, huber: 1.6094, swd: 1.2078, ept: 257.0352
    Epoch [20/50], Val Losses: mse: 37.9473, mae: 3.4776, huber: 3.0465, swd: 2.5034, ept: 190.4930
    Epoch [20/50], Test Losses: mse: 30.7720, mae: 3.0327, huber: 2.6083, swd: 2.4508, ept: 216.4698
      Epoch 20 composite train-obj: 1.609421
            No improvement (3.0465), counter 1/5
    Epoch [21/50], Train Losses: mse: 15.1941, mae: 2.0058, huber: 1.6096, swd: 1.2020, ept: 256.8104
    Epoch [21/50], Val Losses: mse: 38.5501, mae: 3.4364, huber: 3.0077, swd: 1.9594, ept: 199.8334
    Epoch [21/50], Test Losses: mse: 29.0358, mae: 2.8681, huber: 2.4485, swd: 1.7430, ept: 223.7364
      Epoch 21 composite train-obj: 1.609626
            Val objective improved 3.0460 → 3.0077, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 14.4169, mae: 1.9268, huber: 1.5348, swd: 1.1054, ept: 261.5160
    Epoch [22/50], Val Losses: mse: 39.5617, mae: 3.4775, huber: 3.0518, swd: 2.2434, ept: 202.3399
    Epoch [22/50], Test Losses: mse: 28.5741, mae: 2.8389, huber: 2.4231, swd: 2.0648, ept: 226.2274
      Epoch 22 composite train-obj: 1.534761
            No improvement (3.0518), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.6944, mae: 1.8747, huber: 1.4847, swd: 1.0620, ept: 263.8163
    Epoch [23/50], Val Losses: mse: 36.4423, mae: 3.3551, huber: 2.9269, swd: 2.0140, ept: 199.1112
    Epoch [23/50], Test Losses: mse: 27.2449, mae: 2.8051, huber: 2.3871, swd: 1.8363, ept: 220.4542
      Epoch 23 composite train-obj: 1.484679
            Val objective improved 3.0077 → 2.9269, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 14.0368, mae: 1.8908, huber: 1.5011, swd: 1.0604, ept: 262.6505
    Epoch [24/50], Val Losses: mse: 36.8849, mae: 3.3230, huber: 2.8991, swd: 2.0519, ept: 205.2383
    Epoch [24/50], Test Losses: mse: 26.6080, mae: 2.7265, huber: 2.3119, swd: 1.8409, ept: 229.0242
      Epoch 24 composite train-obj: 1.501090
            Val objective improved 2.9269 → 2.8991, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.8363, mae: 1.7864, huber: 1.4016, swd: 0.9696, ept: 268.0051
    Epoch [25/50], Val Losses: mse: 40.6119, mae: 3.5425, huber: 3.1126, swd: 2.2111, ept: 200.1985
    Epoch [25/50], Test Losses: mse: 26.6057, mae: 2.7563, huber: 2.3379, swd: 1.7611, ept: 224.0517
      Epoch 25 composite train-obj: 1.401614
            No improvement (3.1126), counter 1/5
    Epoch [26/50], Train Losses: mse: 13.7591, mae: 1.8722, huber: 1.4818, swd: 1.0541, ept: 265.4856
    Epoch [26/50], Val Losses: mse: 39.5056, mae: 3.4570, huber: 3.0298, swd: 1.6950, ept: 201.6933
    Epoch [26/50], Test Losses: mse: 27.8702, mae: 2.7897, huber: 2.3723, swd: 1.5847, ept: 228.5089
      Epoch 26 composite train-obj: 1.481824
            No improvement (3.0298), counter 2/5
    Epoch [27/50], Train Losses: mse: 12.3769, mae: 1.7418, huber: 1.3598, swd: 0.9308, ept: 270.4637
    Epoch [27/50], Val Losses: mse: 36.3364, mae: 3.2843, huber: 2.8632, swd: 1.9691, ept: 204.7075
    Epoch [27/50], Test Losses: mse: 27.5384, mae: 2.7384, huber: 2.3279, swd: 1.7376, ept: 230.8358
      Epoch 27 composite train-obj: 1.359758
            Val objective improved 2.8991 → 2.8632, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 13.2678, mae: 1.8210, huber: 1.4335, swd: 0.9992, ept: 267.8108
    Epoch [28/50], Val Losses: mse: 36.0647, mae: 3.3148, huber: 2.8905, swd: 2.0817, ept: 207.8376
    Epoch [28/50], Test Losses: mse: 29.7040, mae: 2.8828, huber: 2.4663, swd: 1.7839, ept: 223.1403
      Epoch 28 composite train-obj: 1.433482
            No improvement (2.8905), counter 1/5
    Epoch [29/50], Train Losses: mse: 12.2688, mae: 1.7227, huber: 1.3416, swd: 0.9018, ept: 272.2781
    Epoch [29/50], Val Losses: mse: 40.0124, mae: 3.4952, huber: 3.0667, swd: 1.9500, ept: 205.2281
    Epoch [29/50], Test Losses: mse: 32.1050, mae: 3.0118, huber: 2.5919, swd: 1.7193, ept: 219.8432
      Epoch 29 composite train-obj: 1.341555
            No improvement (3.0667), counter 2/5
    Epoch [30/50], Train Losses: mse: 11.9185, mae: 1.7011, huber: 1.3204, swd: 0.8969, ept: 272.8709
    Epoch [30/50], Val Losses: mse: 37.6511, mae: 3.3100, huber: 2.8881, swd: 1.7584, ept: 209.8243
    Epoch [30/50], Test Losses: mse: 27.0926, mae: 2.6979, huber: 2.2865, swd: 1.6435, ept: 233.8753
      Epoch 30 composite train-obj: 1.320413
            No improvement (2.8881), counter 3/5
    Epoch [31/50], Train Losses: mse: 12.8283, mae: 1.7630, huber: 1.3800, swd: 0.9586, ept: 271.2549
    Epoch [31/50], Val Losses: mse: 37.0583, mae: 3.2869, huber: 2.8681, swd: 1.9448, ept: 210.8007
    Epoch [31/50], Test Losses: mse: 26.1113, mae: 2.6274, huber: 2.2195, swd: 1.5564, ept: 236.4658
      Epoch 31 composite train-obj: 1.380032
            No improvement (2.8681), counter 4/5
    Epoch [32/50], Train Losses: mse: 12.7548, mae: 1.7538, huber: 1.3720, swd: 0.9466, ept: 271.5721
    Epoch [32/50], Val Losses: mse: 39.0456, mae: 3.3904, huber: 2.9640, swd: 1.7178, ept: 211.8844
    Epoch [32/50], Test Losses: mse: 27.1581, mae: 2.7111, huber: 2.2946, swd: 1.5082, ept: 235.4641
      Epoch 32 composite train-obj: 1.371951
    Epoch [32/50], Test Losses: mse: 27.5409, mae: 2.7386, huber: 2.3280, swd: 1.7364, ept: 230.8194
    Best round's Test MSE: 27.5384, MAE: 2.7384, SWD: 1.7376
    Best round's Validation MSE: 36.3364, MAE: 3.2843, SWD: 1.9691
    Best round's Test verification MSE : 27.5409, MAE: 2.7386, SWD: 1.7364
    Time taken: 114.88 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 75.7161, mae: 6.5336, huber: 6.0538, swd: 44.0697, ept: 34.7316
    Epoch [1/50], Val Losses: mse: 59.6383, mae: 5.8005, huber: 5.3249, swd: 25.6871, ept: 40.8869
    Epoch [1/50], Test Losses: mse: 55.2748, mae: 5.4555, huber: 4.9830, swd: 27.3119, ept: 41.3207
      Epoch 1 composite train-obj: 6.053820
            Val objective improved inf → 5.3249, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.7772, mae: 4.8865, huber: 4.4204, swd: 17.8189, ept: 88.8604
    Epoch [2/50], Val Losses: mse: 51.7706, mae: 5.0629, huber: 4.5963, swd: 15.5079, ept: 78.0877
    Epoch [2/50], Test Losses: mse: 48.1579, mae: 4.8053, huber: 4.3408, swd: 17.2491, ept: 72.9891
      Epoch 2 composite train-obj: 4.420419
            Val objective improved 5.3249 → 4.5963, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 41.2878, mae: 4.2082, huber: 3.7537, swd: 11.0955, ept: 126.8229
    Epoch [3/50], Val Losses: mse: 45.9871, mae: 4.5805, huber: 4.1197, swd: 9.9514, ept: 102.5916
    Epoch [3/50], Test Losses: mse: 42.7633, mae: 4.3453, huber: 3.8864, swd: 11.2973, ept: 101.8708
      Epoch 3 composite train-obj: 3.753728
            Val objective improved 4.5963 → 4.1197, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 36.1574, mae: 3.7842, huber: 3.3363, swd: 6.8669, ept: 153.2396
    Epoch [4/50], Val Losses: mse: 43.2385, mae: 4.3121, huber: 3.8563, swd: 7.3558, ept: 118.7228
    Epoch [4/50], Test Losses: mse: 39.5371, mae: 4.0360, huber: 3.5827, swd: 7.8929, ept: 120.4829
      Epoch 4 composite train-obj: 3.336278
            Val objective improved 4.1197 → 3.8563, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.7204, mae: 3.4738, huber: 3.0333, swd: 5.1444, ept: 174.7856
    Epoch [5/50], Val Losses: mse: 41.7057, mae: 4.1315, huber: 3.6797, swd: 6.2121, ept: 133.5317
    Epoch [5/50], Test Losses: mse: 36.5535, mae: 3.8057, huber: 3.3568, swd: 6.4437, ept: 139.0838
      Epoch 5 composite train-obj: 3.033299
            Val objective improved 3.8563 → 3.6797, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.7458, mae: 3.2299, huber: 2.7955, swd: 4.2670, ept: 188.7566
    Epoch [6/50], Val Losses: mse: 41.6378, mae: 4.0821, huber: 3.6335, swd: 6.0367, ept: 145.6058
    Epoch [6/50], Test Losses: mse: 33.2018, mae: 3.5755, huber: 3.1318, swd: 5.2939, ept: 155.3585
      Epoch 6 composite train-obj: 2.795522
            Val objective improved 3.6797 → 3.6335, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.9520, mae: 3.1141, huber: 2.6810, swd: 3.6674, ept: 195.2118
    Epoch [7/50], Val Losses: mse: 43.8390, mae: 4.1732, huber: 3.7219, swd: 5.6912, ept: 149.8338
    Epoch [7/50], Test Losses: mse: 36.6197, mae: 3.7179, huber: 3.2720, swd: 5.5224, ept: 161.5102
      Epoch 7 composite train-obj: 2.680993
            No improvement (3.7219), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.5710, mae: 2.9184, huber: 2.4906, swd: 3.1168, ept: 203.7579
    Epoch [8/50], Val Losses: mse: 41.0229, mae: 3.9585, huber: 3.5126, swd: 4.7030, ept: 162.1398
    Epoch [8/50], Test Losses: mse: 31.8046, mae: 3.4136, huber: 2.9724, swd: 3.9798, ept: 173.0248
      Epoch 8 composite train-obj: 2.490647
            Val objective improved 3.6335 → 3.5126, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.8175, mae: 2.7810, huber: 2.3564, swd: 2.7008, ept: 209.4787
    Epoch [9/50], Val Losses: mse: 41.5218, mae: 3.9087, huber: 3.4653, swd: 4.1124, ept: 165.2336
    Epoch [9/50], Test Losses: mse: 29.5319, mae: 3.2461, huber: 2.8093, swd: 3.5040, ept: 178.2774
      Epoch 9 composite train-obj: 2.356405
            Val objective improved 3.5126 → 3.4653, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 23.0014, mae: 2.7139, huber: 2.2909, swd: 2.5258, ept: 212.8989
    Epoch [10/50], Val Losses: mse: 40.5713, mae: 3.8100, huber: 3.3685, swd: 3.8991, ept: 169.4411
    Epoch [10/50], Test Losses: mse: 32.3408, mae: 3.3427, huber: 2.9060, swd: 3.6596, ept: 179.5574
      Epoch 10 composite train-obj: 2.290927
            Val objective improved 3.4653 → 3.3685, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.0902, mae: 2.5492, huber: 2.1316, swd: 2.2120, ept: 221.2066
    Epoch [11/50], Val Losses: mse: 41.3233, mae: 3.8020, huber: 3.3629, swd: 3.8682, ept: 173.2007
    Epoch [11/50], Test Losses: mse: 29.1303, mae: 3.1183, huber: 2.6873, swd: 3.3464, ept: 191.2484
      Epoch 11 composite train-obj: 2.131623
            Val objective improved 3.3685 → 3.3629, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 19.7434, mae: 2.4485, huber: 2.0338, swd: 1.9931, ept: 225.8413
    Epoch [12/50], Val Losses: mse: 40.9860, mae: 3.8014, huber: 3.3601, swd: 3.8936, ept: 173.4207
    Epoch [12/50], Test Losses: mse: 33.0039, mae: 3.3263, huber: 2.8906, swd: 3.3438, ept: 188.5013
      Epoch 12 composite train-obj: 2.033773
            Val objective improved 3.3629 → 3.3601, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 19.4610, mae: 2.4186, huber: 2.0052, swd: 1.9253, ept: 227.6032
    Epoch [13/50], Val Losses: mse: 38.0870, mae: 3.6180, huber: 3.1796, swd: 3.5426, ept: 180.8899
    Epoch [13/50], Test Losses: mse: 30.0626, mae: 3.1532, huber: 2.7214, swd: 2.8575, ept: 189.9661
      Epoch 13 composite train-obj: 2.005160
            Val objective improved 3.3601 → 3.1796, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.3189, mae: 2.3231, huber: 1.9131, swd: 1.7594, ept: 231.7047
    Epoch [14/50], Val Losses: mse: 39.7384, mae: 3.6437, huber: 3.2084, swd: 3.4721, ept: 184.7465
    Epoch [14/50], Test Losses: mse: 29.9064, mae: 3.0714, huber: 2.6438, swd: 2.6105, ept: 201.9743
      Epoch 14 composite train-obj: 1.913111
            No improvement (3.2084), counter 1/5
    Epoch [15/50], Train Losses: mse: 17.6808, mae: 2.2616, huber: 1.8544, swd: 1.6589, ept: 235.1383
    Epoch [15/50], Val Losses: mse: 42.9488, mae: 3.7723, huber: 3.3395, swd: 3.2542, ept: 185.9439
    Epoch [15/50], Test Losses: mse: 27.4271, mae: 2.9167, huber: 2.4939, swd: 2.6861, ept: 205.9016
      Epoch 15 composite train-obj: 1.854358
            No improvement (3.3395), counter 2/5
    Epoch [16/50], Train Losses: mse: 16.6088, mae: 2.1509, huber: 1.7490, swd: 1.5408, ept: 241.6534
    Epoch [16/50], Val Losses: mse: 41.2809, mae: 3.7045, huber: 3.2691, swd: 3.5632, ept: 183.7130
    Epoch [16/50], Test Losses: mse: 28.3379, mae: 3.0066, huber: 2.5786, swd: 2.5929, ept: 200.4416
      Epoch 16 composite train-obj: 1.749035
            No improvement (3.2691), counter 3/5
    Epoch [17/50], Train Losses: mse: 16.0270, mae: 2.1093, huber: 1.7080, swd: 1.4340, ept: 243.4986
    Epoch [17/50], Val Losses: mse: 39.3535, mae: 3.5817, huber: 3.1477, swd: 2.8270, ept: 191.3760
    Epoch [17/50], Test Losses: mse: 26.9225, mae: 2.8937, huber: 2.4675, swd: 2.2089, ept: 212.2390
      Epoch 17 composite train-obj: 1.708040
            Val objective improved 3.1796 → 3.1477, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.3908, mae: 2.0567, huber: 1.6571, swd: 1.3913, ept: 246.7147
    Epoch [18/50], Val Losses: mse: 35.5286, mae: 3.3870, huber: 2.9573, swd: 3.1072, ept: 196.3688
    Epoch [18/50], Test Losses: mse: 24.7593, mae: 2.7473, huber: 2.3291, swd: 2.3461, ept: 215.4132
      Epoch 18 composite train-obj: 1.657125
            Val objective improved 3.1477 → 2.9573, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 14.1181, mae: 1.9373, huber: 1.5438, swd: 1.2149, ept: 252.0449
    Epoch [19/50], Val Losses: mse: 39.0245, mae: 3.5202, huber: 3.0897, swd: 3.0321, ept: 195.3917
    Epoch [19/50], Test Losses: mse: 25.8548, mae: 2.8202, huber: 2.3981, swd: 2.2523, ept: 211.6526
      Epoch 19 composite train-obj: 1.543764
            No improvement (3.0897), counter 1/5
    Epoch [20/50], Train Losses: mse: 14.3141, mae: 1.9479, huber: 1.5537, swd: 1.2501, ept: 253.6534
    Epoch [20/50], Val Losses: mse: 40.8530, mae: 3.6311, huber: 3.2018, swd: 3.2242, ept: 195.0099
    Epoch [20/50], Test Losses: mse: 30.9736, mae: 3.0325, huber: 2.6105, swd: 2.4630, ept: 212.3589
      Epoch 20 composite train-obj: 1.553741
            No improvement (3.2018), counter 2/5
    Epoch [21/50], Train Losses: mse: 13.9561, mae: 1.9149, huber: 1.5225, swd: 1.1889, ept: 255.5712
    Epoch [21/50], Val Losses: mse: 37.4931, mae: 3.4075, huber: 2.9818, swd: 2.9370, ept: 199.6944
    Epoch [21/50], Test Losses: mse: 27.0282, mae: 2.8048, huber: 2.3875, swd: 2.1371, ept: 224.0476
      Epoch 21 composite train-obj: 1.522475
            No improvement (2.9818), counter 3/5
    Epoch [22/50], Train Losses: mse: 13.9517, mae: 1.9010, huber: 1.5101, swd: 1.1976, ept: 257.4417
    Epoch [22/50], Val Losses: mse: 38.5251, mae: 3.4544, huber: 3.0245, swd: 2.6099, ept: 204.3925
    Epoch [22/50], Test Losses: mse: 26.5280, mae: 2.7776, huber: 2.3569, swd: 1.9379, ept: 224.5732
      Epoch 22 composite train-obj: 1.510060
            No improvement (3.0245), counter 4/5
    Epoch [23/50], Train Losses: mse: 12.5072, mae: 1.7716, huber: 1.3877, swd: 1.0456, ept: 262.7782
    Epoch [23/50], Val Losses: mse: 40.0909, mae: 3.5393, huber: 3.1113, swd: 3.0118, ept: 200.4266
    Epoch [23/50], Test Losses: mse: 28.2403, mae: 2.8666, huber: 2.4476, swd: 2.0989, ept: 221.9717
      Epoch 23 composite train-obj: 1.387693
    Epoch [23/50], Test Losses: mse: 24.7616, mae: 2.7474, huber: 2.3292, swd: 2.3459, ept: 215.4316
    Best round's Test MSE: 24.7593, MAE: 2.7473, SWD: 2.3461
    Best round's Validation MSE: 35.5286, MAE: 3.3870, SWD: 3.1072
    Best round's Test verification MSE : 24.7616, MAE: 2.7474, SWD: 2.3459
    Time taken: 84.70 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1632)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 26.8617 ± 1.5178
      mae: 2.7520 ± 0.0135
      huber: 2.3391 ± 0.0150
      swd: 1.8501 ± 0.3678
      ept: 226.0661 ± 7.5465
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 35.8606 ± 0.3451
      mae: 3.3038 ± 0.0616
      huber: 2.8804 ± 0.0571
      swd: 2.3709 ± 0.5214
      ept: 204.5950 ± 6.6712
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 358.64 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1632
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: deterministic y0 with dist t(x) 
A(y0+t); note y0 as a single point must be added to a t() to retain spread
A as OT cannot operate on dirac y0. 
Results is far worse, the reason is explained in the paper. 


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
    # ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=True, ### HERE
    ablate_shift_inside_scale=True, ### HERE Note: they must be True together
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
    
    Epoch [1/50], Train Losses: mse: 70.5719, mae: 6.3075, huber: 5.8286, swd: 33.9828, ept: 36.6032
    Epoch [1/50], Val Losses: mse: 61.4516, mae: 5.8700, huber: 5.3934, swd: 26.4553, ept: 47.7560
    Epoch [1/50], Test Losses: mse: 56.0727, mae: 5.4544, huber: 4.9811, swd: 26.5209, ept: 47.6238
      Epoch 1 composite train-obj: 5.828635
            Val objective improved inf → 5.3934, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.3833, mae: 4.9418, huber: 4.4746, swd: 18.7350, ept: 91.9460
    Epoch [2/50], Val Losses: mse: 54.7177, mae: 5.2992, huber: 4.8298, swd: 17.4125, ept: 80.2270
    Epoch [2/50], Test Losses: mse: 49.5875, mae: 4.9252, huber: 4.4580, swd: 17.9492, ept: 78.9272
      Epoch 2 composite train-obj: 4.474647
            Val objective improved 5.3934 → 4.8298, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 43.2254, mae: 4.3790, huber: 3.9209, swd: 12.1333, ept: 130.4029
    Epoch [3/50], Val Losses: mse: 53.6252, mae: 5.1013, huber: 4.6347, swd: 13.6209, ept: 97.7020
    Epoch [3/50], Test Losses: mse: 47.0352, mae: 4.6608, huber: 4.1967, swd: 13.5471, ept: 100.5430
      Epoch 3 composite train-obj: 3.920900
            Val objective improved 4.8298 → 4.6347, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.2437, mae: 4.0321, huber: 3.5803, swd: 9.2693, ept: 150.0911
    Epoch [4/50], Val Losses: mse: 50.5331, mae: 4.8576, huber: 4.3955, swd: 11.3626, ept: 111.0715
    Epoch [4/50], Test Losses: mse: 44.7418, mae: 4.4529, huber: 3.9949, swd: 12.1558, ept: 113.6510
      Epoch 4 composite train-obj: 3.580330
            Val objective improved 4.6347 → 4.3955, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 36.2269, mae: 3.7738, huber: 3.3274, swd: 7.4723, ept: 161.7042
    Epoch [5/50], Val Losses: mse: 47.8616, mae: 4.6075, huber: 4.1482, swd: 8.5800, ept: 118.1049
    Epoch [5/50], Test Losses: mse: 41.8061, mae: 4.2090, huber: 3.7528, swd: 8.9059, ept: 124.0191
      Epoch 5 composite train-obj: 3.327429
            Val objective improved 4.3955 → 4.1482, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 33.8762, mae: 3.5911, huber: 3.1479, swd: 6.1712, ept: 170.0567
    Epoch [6/50], Val Losses: mse: 45.6370, mae: 4.4259, huber: 3.9697, swd: 7.8479, ept: 124.8755
    Epoch [6/50], Test Losses: mse: 39.6960, mae: 4.0364, huber: 3.5850, swd: 8.2531, ept: 132.9011
      Epoch 6 composite train-obj: 3.147850
            Val objective improved 4.1482 → 3.9697, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 31.4785, mae: 3.3857, huber: 2.9476, swd: 5.2158, ept: 179.6068
    Epoch [7/50], Val Losses: mse: 46.5819, mae: 4.3852, huber: 3.9317, swd: 6.4673, ept: 130.2184
    Epoch [7/50], Test Losses: mse: 38.0720, mae: 3.8882, huber: 3.4388, swd: 6.3297, ept: 144.1063
      Epoch 7 composite train-obj: 2.947629
            Val objective improved 3.9697 → 3.9317, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.5954, mae: 3.2411, huber: 2.8058, swd: 4.5368, ept: 185.8102
    Epoch [8/50], Val Losses: mse: 45.8709, mae: 4.3513, huber: 3.8977, swd: 6.6632, ept: 138.8200
    Epoch [8/50], Test Losses: mse: 36.7783, mae: 3.8377, huber: 3.3878, swd: 6.5147, ept: 146.9080
      Epoch 8 composite train-obj: 2.805779
            Val objective improved 3.9317 → 3.8977, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 28.7040, mae: 3.1758, huber: 2.7416, swd: 4.2045, ept: 188.9192
    Epoch [9/50], Val Losses: mse: 42.4291, mae: 4.0920, huber: 3.6439, swd: 5.6613, ept: 142.0294
    Epoch [9/50], Test Losses: mse: 35.3024, mae: 3.6815, huber: 3.2364, swd: 5.7282, ept: 154.3333
      Epoch 9 composite train-obj: 2.741612
            Val objective improved 3.8977 → 3.6439, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 26.8592, mae: 3.0302, huber: 2.5993, swd: 3.7542, ept: 195.5365
    Epoch [10/50], Val Losses: mse: 43.1413, mae: 4.0680, huber: 3.6184, swd: 4.3612, ept: 154.3064
    Epoch [10/50], Test Losses: mse: 35.8541, mae: 3.6857, huber: 3.2379, swd: 4.3374, ept: 163.2169
      Epoch 10 composite train-obj: 2.599261
            Val objective improved 3.6439 → 3.6184, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 25.6990, mae: 2.9320, huber: 2.5039, swd: 3.4301, ept: 200.1038
    Epoch [11/50], Val Losses: mse: 39.6846, mae: 3.8996, huber: 3.4539, swd: 4.6048, ept: 154.1027
    Epoch [11/50], Test Losses: mse: 34.1088, mae: 3.5596, huber: 3.1175, swd: 4.4209, ept: 164.9865
      Epoch 11 composite train-obj: 2.503944
            Val objective improved 3.6184 → 3.4539, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 24.3355, mae: 2.8287, huber: 2.4031, swd: 3.0883, ept: 204.7209
    Epoch [12/50], Val Losses: mse: 37.7444, mae: 3.7434, huber: 3.2989, swd: 4.1773, ept: 161.6360
    Epoch [12/50], Test Losses: mse: 32.1612, mae: 3.3968, huber: 2.9555, swd: 3.8768, ept: 179.3963
      Epoch 12 composite train-obj: 2.403077
            Val objective improved 3.4539 → 3.2989, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 23.3955, mae: 2.7520, huber: 2.3280, swd: 2.8661, ept: 209.2210
    Epoch [13/50], Val Losses: mse: 38.6208, mae: 3.7463, huber: 3.3042, swd: 3.8077, ept: 163.3346
    Epoch [13/50], Test Losses: mse: 33.0358, mae: 3.3816, huber: 2.9442, swd: 3.6717, ept: 181.5138
      Epoch 13 composite train-obj: 2.328025
            No improvement (3.3042), counter 1/5
    Epoch [14/50], Train Losses: mse: 22.0419, mae: 2.6341, huber: 2.2139, swd: 2.5931, ept: 214.8983
    Epoch [14/50], Val Losses: mse: 38.2840, mae: 3.7547, huber: 3.3111, swd: 3.9065, ept: 165.4411
    Epoch [14/50], Test Losses: mse: 31.4707, mae: 3.3404, huber: 2.9019, swd: 4.0057, ept: 181.7006
      Epoch 14 composite train-obj: 2.213872
            No improvement (3.3111), counter 2/5
    Epoch [15/50], Train Losses: mse: 21.6288, mae: 2.6046, huber: 2.1847, swd: 2.4346, ept: 217.3347
    Epoch [15/50], Val Losses: mse: 39.9205, mae: 3.7633, huber: 3.3237, swd: 3.7836, ept: 165.1486
    Epoch [15/50], Test Losses: mse: 32.4615, mae: 3.3349, huber: 2.9002, swd: 3.4011, ept: 186.3380
      Epoch 15 composite train-obj: 2.184730
            No improvement (3.3237), counter 3/5
    Epoch [16/50], Train Losses: mse: 20.3905, mae: 2.4936, huber: 2.0777, swd: 2.2224, ept: 223.5891
    Epoch [16/50], Val Losses: mse: 35.5187, mae: 3.5115, huber: 3.0737, swd: 3.1007, ept: 176.5062
    Epoch [16/50], Test Losses: mse: 31.5373, mae: 3.2587, huber: 2.8236, swd: 3.1166, ept: 192.8127
      Epoch 16 composite train-obj: 2.077737
            Val objective improved 3.2989 → 3.0737, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 20.5066, mae: 2.5171, huber: 2.0989, swd: 2.2237, ept: 223.3363
    Epoch [17/50], Val Losses: mse: 40.4945, mae: 3.7391, huber: 3.3007, swd: 3.3812, ept: 169.7628
    Epoch [17/50], Test Losses: mse: 32.8731, mae: 3.3130, huber: 2.8789, swd: 3.0778, ept: 190.2964
      Epoch 17 composite train-obj: 2.098855
            No improvement (3.3007), counter 1/5
    Epoch [18/50], Train Losses: mse: 19.8434, mae: 2.4593, huber: 2.0432, swd: 2.1102, ept: 225.7442
    Epoch [18/50], Val Losses: mse: 37.3820, mae: 3.5686, huber: 3.1313, swd: 3.1369, ept: 174.6272
    Epoch [18/50], Test Losses: mse: 30.1378, mae: 3.1085, huber: 2.6787, swd: 2.9921, ept: 201.9006
      Epoch 18 composite train-obj: 2.043174
            No improvement (3.1313), counter 2/5
    Epoch [19/50], Train Losses: mse: 18.4130, mae: 2.3347, huber: 1.9234, swd: 1.8636, ept: 232.3077
    Epoch [19/50], Val Losses: mse: 37.1265, mae: 3.5548, huber: 3.1182, swd: 3.0237, ept: 171.4852
    Epoch [19/50], Test Losses: mse: 30.5184, mae: 3.1266, huber: 2.6975, swd: 2.8451, ept: 195.4424
      Epoch 19 composite train-obj: 1.923394
            No improvement (3.1182), counter 3/5
    Epoch [20/50], Train Losses: mse: 17.9640, mae: 2.3046, huber: 1.8930, swd: 1.7851, ept: 234.0437
    Epoch [20/50], Val Losses: mse: 37.0582, mae: 3.5156, huber: 3.0812, swd: 2.7575, ept: 178.8013
    Epoch [20/50], Test Losses: mse: 31.6815, mae: 3.1803, huber: 2.7501, swd: 2.8298, ept: 199.8260
      Epoch 20 composite train-obj: 1.892974
            No improvement (3.0812), counter 4/5
    Epoch [21/50], Train Losses: mse: 17.1511, mae: 2.2357, huber: 1.8266, swd: 1.6811, ept: 238.7341
    Epoch [21/50], Val Losses: mse: 36.9984, mae: 3.4848, huber: 3.0516, swd: 2.8497, ept: 180.4331
    Epoch [21/50], Test Losses: mse: 29.8386, mae: 3.0562, huber: 2.6289, swd: 2.4897, ept: 201.6236
      Epoch 21 composite train-obj: 1.826647
            Val objective improved 3.0737 → 3.0516, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 16.5547, mae: 2.1873, huber: 1.7797, swd: 1.5339, ept: 241.0639
    Epoch [22/50], Val Losses: mse: 35.9545, mae: 3.4004, huber: 2.9697, swd: 2.6797, ept: 185.8498
    Epoch [22/50], Test Losses: mse: 30.8575, mae: 3.0457, huber: 2.6214, swd: 2.4894, ept: 213.9750
      Epoch 22 composite train-obj: 1.779700
            Val objective improved 3.0516 → 2.9697, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 15.5119, mae: 2.0964, huber: 1.6921, swd: 1.4385, ept: 245.7940
    Epoch [23/50], Val Losses: mse: 32.8311, mae: 3.2705, huber: 2.8398, swd: 2.4895, ept: 188.3890
    Epoch [23/50], Test Losses: mse: 29.0966, mae: 2.9652, huber: 2.5411, swd: 2.1833, ept: 209.4325
      Epoch 23 composite train-obj: 1.692097
            Val objective improved 2.9697 → 2.8398, saving checkpoint.
    Epoch [24/50], Train Losses: mse: 15.2161, mae: 2.0769, huber: 1.6729, swd: 1.3763, ept: 247.1656
    Epoch [24/50], Val Losses: mse: 36.9907, mae: 3.4048, huber: 2.9753, swd: 2.3894, ept: 192.6206
    Epoch [24/50], Test Losses: mse: 27.5007, mae: 2.8777, huber: 2.4548, swd: 2.1610, ept: 214.1320
      Epoch 24 composite train-obj: 1.672861
            No improvement (2.9753), counter 1/5
    Epoch [25/50], Train Losses: mse: 15.4853, mae: 2.0769, huber: 1.6742, swd: 1.3685, ept: 248.5447
    Epoch [25/50], Val Losses: mse: 35.5390, mae: 3.3305, huber: 2.9012, swd: 2.3331, ept: 187.9434
    Epoch [25/50], Test Losses: mse: 28.4695, mae: 2.8779, huber: 2.4560, swd: 2.0091, ept: 216.6759
      Epoch 25 composite train-obj: 1.674243
            No improvement (2.9012), counter 2/5
    Epoch [26/50], Train Losses: mse: 14.6863, mae: 2.0027, huber: 1.6036, swd: 1.2766, ept: 252.6062
    Epoch [26/50], Val Losses: mse: 35.5781, mae: 3.3276, huber: 2.8992, swd: 2.1849, ept: 193.9777
    Epoch [26/50], Test Losses: mse: 27.3744, mae: 2.8404, huber: 2.4199, swd: 1.8576, ept: 218.4248
      Epoch 26 composite train-obj: 1.603603
            No improvement (2.8992), counter 3/5
    Epoch [27/50], Train Losses: mse: 14.5936, mae: 2.0044, huber: 1.6042, swd: 1.2608, ept: 252.3104
    Epoch [27/50], Val Losses: mse: 34.9717, mae: 3.2681, huber: 2.8404, swd: 1.9928, ept: 195.4760
    Epoch [27/50], Test Losses: mse: 29.5831, mae: 2.9052, huber: 2.4863, swd: 1.8419, ept: 221.0715
      Epoch 27 composite train-obj: 1.604153
            No improvement (2.8404), counter 4/5
    Epoch [28/50], Train Losses: mse: 14.3477, mae: 1.9709, huber: 1.5727, swd: 1.2263, ept: 255.2082
    Epoch [28/50], Val Losses: mse: 37.9854, mae: 3.4726, huber: 3.0392, swd: 2.5208, ept: 187.4718
    Epoch [28/50], Test Losses: mse: 29.5351, mae: 2.9780, huber: 2.5514, swd: 2.1997, ept: 213.3305
      Epoch 28 composite train-obj: 1.572671
    Epoch [28/50], Test Losses: mse: 29.0971, mae: 2.9652, huber: 2.5411, swd: 2.1834, ept: 209.4214
    Best round's Test MSE: 29.0966, MAE: 2.9652, SWD: 2.1833
    Best round's Validation MSE: 32.8311, MAE: 3.2705, SWD: 2.4895
    Best round's Test verification MSE : 29.0971, MAE: 2.9652, SWD: 2.1834
    Time taken: 106.48 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.4266, mae: 6.2757, huber: 5.7972, swd: 33.8517, ept: 36.8617
    Epoch [1/50], Val Losses: mse: 60.5773, mae: 5.8339, huber: 5.3584, swd: 27.3673, ept: 44.4988
    Epoch [1/50], Test Losses: mse: 55.4781, mae: 5.4403, huber: 4.9680, swd: 29.5950, ept: 41.6699
      Epoch 1 composite train-obj: 5.797181
            Val objective improved inf → 5.3584, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.6740, mae: 4.9566, huber: 4.4895, swd: 18.8362, ept: 88.0128
    Epoch [2/50], Val Losses: mse: 55.9310, mae: 5.3296, huber: 4.8606, swd: 16.5468, ept: 77.8047
    Epoch [2/50], Test Losses: mse: 49.6462, mae: 4.8997, huber: 4.4335, swd: 17.7586, ept: 76.3732
      Epoch 2 composite train-obj: 4.489455
            Val objective improved 5.3584 → 4.8606, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 43.8527, mae: 4.4223, huber: 3.9632, swd: 12.0255, ept: 122.8169
    Epoch [3/50], Val Losses: mse: 53.2294, mae: 4.9946, huber: 4.5312, swd: 11.9639, ept: 95.6495
    Epoch [3/50], Test Losses: mse: 47.3226, mae: 4.6054, huber: 4.1455, swd: 12.5042, ept: 95.5196
      Epoch 3 composite train-obj: 3.963237
            Val objective improved 4.8606 → 4.5312, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 39.9716, mae: 4.0703, huber: 3.6177, swd: 8.7912, ept: 142.9235
    Epoch [4/50], Val Losses: mse: 50.7231, mae: 4.7848, huber: 4.3242, swd: 10.0174, ept: 106.1874
    Epoch [4/50], Test Losses: mse: 44.2632, mae: 4.4076, huber: 3.9494, swd: 10.8160, ept: 109.1062
      Epoch 4 composite train-obj: 3.617746
            Val objective improved 4.5312 → 4.3242, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 37.0518, mae: 3.8268, huber: 3.3789, swd: 7.1899, ept: 154.9601
    Epoch [5/50], Val Losses: mse: 48.8384, mae: 4.6161, huber: 4.1580, swd: 8.0206, ept: 117.3161
    Epoch [5/50], Test Losses: mse: 42.2085, mae: 4.2024, huber: 3.7481, swd: 8.6303, ept: 120.2119
      Epoch 5 composite train-obj: 3.378911
            Val objective improved 4.3242 → 4.1580, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 34.7901, mae: 3.6438, huber: 3.1998, swd: 6.0056, ept: 162.7801
    Epoch [6/50], Val Losses: mse: 47.3091, mae: 4.5141, huber: 4.0568, swd: 6.9696, ept: 123.9255
    Epoch [6/50], Test Losses: mse: 40.5223, mae: 4.1048, huber: 3.6498, swd: 7.6964, ept: 128.2712
      Epoch 6 composite train-obj: 3.199815
            Val objective improved 4.1580 → 4.0568, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 32.8218, mae: 3.4858, huber: 3.0451, swd: 5.3534, ept: 172.1778
    Epoch [7/50], Val Losses: mse: 47.6912, mae: 4.4253, huber: 3.9718, swd: 5.5729, ept: 129.7242
    Epoch [7/50], Test Losses: mse: 39.5005, mae: 3.9602, huber: 3.5104, swd: 6.0946, ept: 138.4742
      Epoch 7 composite train-obj: 3.045084
            Val objective improved 4.0568 → 3.9718, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 31.1370, mae: 3.3427, huber: 2.9057, swd: 4.7356, ept: 179.7808
    Epoch [8/50], Val Losses: mse: 44.1646, mae: 4.2797, huber: 3.8253, swd: 5.8497, ept: 126.3860
    Epoch [8/50], Test Losses: mse: 39.1298, mae: 3.9729, huber: 3.5215, swd: 5.9503, ept: 138.9094
      Epoch 8 composite train-obj: 2.905740
            Val objective improved 3.9718 → 3.8253, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 29.6624, mae: 3.2332, huber: 2.7981, swd: 4.3148, ept: 185.8323
    Epoch [9/50], Val Losses: mse: 43.7354, mae: 4.1721, huber: 3.7194, swd: 5.0857, ept: 142.0117
    Epoch [9/50], Test Losses: mse: 36.1014, mae: 3.7277, huber: 3.2802, swd: 5.6808, ept: 151.4726
      Epoch 9 composite train-obj: 2.798065
            Val objective improved 3.8253 → 3.7194, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 27.7414, mae: 3.0762, huber: 2.6455, swd: 3.8377, ept: 192.8220
    Epoch [10/50], Val Losses: mse: 42.0299, mae: 4.0285, huber: 3.5804, swd: 4.2583, ept: 147.3789
    Epoch [10/50], Test Losses: mse: 35.2658, mae: 3.6326, huber: 3.1895, swd: 4.8030, ept: 156.4703
      Epoch 10 composite train-obj: 2.645521
            Val objective improved 3.7194 → 3.5804, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 26.9400, mae: 3.0317, huber: 2.6003, swd: 3.6018, ept: 195.5134
    Epoch [11/50], Val Losses: mse: 45.2589, mae: 4.1717, huber: 3.7239, swd: 4.2916, ept: 146.1862
    Epoch [11/50], Test Losses: mse: 35.1274, mae: 3.6291, huber: 3.1868, swd: 4.7640, ept: 156.3708
      Epoch 11 composite train-obj: 2.600326
            No improvement (3.7239), counter 1/5
    Epoch [12/50], Train Losses: mse: 25.1338, mae: 2.8893, huber: 2.4619, swd: 3.1547, ept: 201.3155
    Epoch [12/50], Val Losses: mse: 42.0421, mae: 3.9917, huber: 3.5440, swd: 3.5918, ept: 153.8315
    Epoch [12/50], Test Losses: mse: 34.0089, mae: 3.5313, huber: 3.0890, swd: 4.0945, ept: 169.1357
      Epoch 12 composite train-obj: 2.461906
            Val objective improved 3.5804 → 3.5440, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 23.8622, mae: 2.7769, huber: 2.3529, swd: 2.8988, ept: 208.1641
    Epoch [13/50], Val Losses: mse: 45.4794, mae: 4.1003, huber: 3.6550, swd: 3.5836, ept: 152.1610
    Epoch [13/50], Test Losses: mse: 35.7795, mae: 3.6434, huber: 3.2004, swd: 4.1486, ept: 161.8454
      Epoch 13 composite train-obj: 2.352897
            No improvement (3.6550), counter 1/5
    Epoch [14/50], Train Losses: mse: 23.4376, mae: 2.7570, huber: 2.3320, swd: 2.7347, ept: 209.3398
    Epoch [14/50], Val Losses: mse: 41.3476, mae: 3.9009, huber: 3.4555, swd: 3.9543, ept: 158.7659
    Epoch [14/50], Test Losses: mse: 31.7766, mae: 3.3877, huber: 2.9469, swd: 4.2384, ept: 173.0116
      Epoch 14 composite train-obj: 2.332022
            Val objective improved 3.5440 → 3.4555, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 22.1299, mae: 2.6402, huber: 2.2194, swd: 2.5064, ept: 215.3776
    Epoch [15/50], Val Losses: mse: 42.9356, mae: 3.9377, huber: 3.4930, swd: 3.1345, ept: 167.6372
    Epoch [15/50], Test Losses: mse: 32.7465, mae: 3.3756, huber: 2.9368, swd: 3.2559, ept: 184.3919
      Epoch 15 composite train-obj: 2.219438
            No improvement (3.4930), counter 1/5
    Epoch [16/50], Train Losses: mse: 21.0407, mae: 2.5580, huber: 2.1390, swd: 2.3052, ept: 219.2633
    Epoch [16/50], Val Losses: mse: 41.7032, mae: 3.8709, huber: 3.4275, swd: 3.2229, ept: 166.1570
    Epoch [16/50], Test Losses: mse: 34.1143, mae: 3.4550, huber: 3.0155, swd: 3.6311, ept: 175.2838
      Epoch 16 composite train-obj: 2.139042
            Val objective improved 3.4555 → 3.4275, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 20.0404, mae: 2.4953, huber: 2.0767, swd: 2.1551, ept: 222.5256
    Epoch [17/50], Val Losses: mse: 38.9713, mae: 3.6758, huber: 3.2375, swd: 2.6546, ept: 180.4320
    Epoch [17/50], Test Losses: mse: 31.0435, mae: 3.2482, huber: 2.8139, swd: 2.8497, ept: 189.8774
      Epoch 17 composite train-obj: 2.076727
            Val objective improved 3.4275 → 3.2375, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 19.9422, mae: 2.4813, huber: 2.0631, swd: 2.1108, ept: 224.6080
    Epoch [18/50], Val Losses: mse: 39.8226, mae: 3.7484, huber: 3.3093, swd: 3.2383, ept: 171.9963
    Epoch [18/50], Test Losses: mse: 30.5260, mae: 3.1855, huber: 2.7533, swd: 3.2402, ept: 190.1623
      Epoch 18 composite train-obj: 2.063082
            No improvement (3.3093), counter 1/5
    Epoch [19/50], Train Losses: mse: 18.3663, mae: 2.3449, huber: 1.9324, swd: 1.8600, ept: 230.0926
    Epoch [19/50], Val Losses: mse: 41.3065, mae: 3.8113, huber: 3.3699, swd: 3.1756, ept: 170.9225
    Epoch [19/50], Test Losses: mse: 31.9183, mae: 3.2904, huber: 2.8546, swd: 3.2218, ept: 187.2160
      Epoch 19 composite train-obj: 1.932395
            No improvement (3.3699), counter 2/5
    Epoch [20/50], Train Losses: mse: 18.0168, mae: 2.3235, huber: 1.9106, swd: 1.8190, ept: 232.5102
    Epoch [20/50], Val Losses: mse: 42.9121, mae: 3.9136, huber: 3.4675, swd: 3.5819, ept: 168.7148
    Epoch [20/50], Test Losses: mse: 30.6150, mae: 3.2508, huber: 2.8113, swd: 3.5977, ept: 185.1816
      Epoch 20 composite train-obj: 1.910593
            No improvement (3.4675), counter 3/5
    Epoch [21/50], Train Losses: mse: 17.2019, mae: 2.2638, huber: 1.8520, swd: 1.7060, ept: 235.7995
    Epoch [21/50], Val Losses: mse: 41.0343, mae: 3.7078, huber: 3.2712, swd: 2.4948, ept: 178.5354
    Epoch [21/50], Test Losses: mse: 32.2408, mae: 3.2223, huber: 2.7912, swd: 2.6736, ept: 196.1376
      Epoch 21 composite train-obj: 1.852036
            No improvement (3.2712), counter 4/5
    Epoch [22/50], Train Losses: mse: 16.9027, mae: 2.2248, huber: 1.8157, swd: 1.6001, ept: 238.3860
    Epoch [22/50], Val Losses: mse: 40.9661, mae: 3.6959, huber: 3.2601, swd: 2.5506, ept: 180.7508
    Epoch [22/50], Test Losses: mse: 30.4168, mae: 3.1153, huber: 2.6854, swd: 2.5275, ept: 199.3981
      Epoch 22 composite train-obj: 1.815717
    Epoch [22/50], Test Losses: mse: 31.0429, mae: 3.2482, huber: 2.8138, swd: 2.8493, ept: 189.8514
    Best round's Test MSE: 31.0435, MAE: 3.2482, SWD: 2.8497
    Best round's Validation MSE: 38.9713, MAE: 3.6758, SWD: 2.6546
    Best round's Test verification MSE : 31.0429, MAE: 3.2482, SWD: 2.8493
    Time taken: 87.54 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 70.0794, mae: 6.3110, huber: 5.8320, swd: 34.8093, ept: 33.8842
    Epoch [1/50], Val Losses: mse: 60.5656, mae: 5.8145, huber: 5.3393, swd: 27.1189, ept: 44.3432
    Epoch [1/50], Test Losses: mse: 55.7437, mae: 5.4277, huber: 4.9555, swd: 27.6639, ept: 45.3396
      Epoch 1 composite train-obj: 5.832026
            Val objective improved inf → 5.3393, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.8643, mae: 4.9048, huber: 4.4381, swd: 18.2956, ept: 93.0161
    Epoch [2/50], Val Losses: mse: 55.1599, mae: 5.2945, huber: 4.8254, swd: 15.9416, ept: 78.2826
    Epoch [2/50], Test Losses: mse: 49.6277, mae: 4.9183, huber: 4.4510, swd: 16.7922, ept: 73.8839
      Epoch 2 composite train-obj: 4.438117
            Val objective improved 5.3393 → 4.8254, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 42.2544, mae: 4.3228, huber: 3.8648, swd: 11.4608, ept: 127.9241
    Epoch [3/50], Val Losses: mse: 51.7686, mae: 4.9907, huber: 4.5256, swd: 13.4124, ept: 95.1904
    Epoch [3/50], Test Losses: mse: 45.9970, mae: 4.5823, huber: 4.1199, swd: 13.3878, ept: 97.1182
      Epoch 3 composite train-obj: 3.864813
            Val objective improved 4.8254 → 4.5256, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 38.2778, mae: 3.9703, huber: 3.5189, swd: 8.5986, ept: 148.9861
    Epoch [4/50], Val Losses: mse: 49.2556, mae: 4.7047, huber: 4.2445, swd: 9.7884, ept: 110.9265
    Epoch [4/50], Test Losses: mse: 43.0326, mae: 4.3215, huber: 3.8641, swd: 10.1638, ept: 113.6468
      Epoch 4 composite train-obj: 3.518907
            Val objective improved 4.5256 → 4.2445, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 35.4335, mae: 3.7212, huber: 3.2750, swd: 7.0038, ept: 162.7462
    Epoch [5/50], Val Losses: mse: 46.6611, mae: 4.4933, huber: 4.0368, swd: 8.3054, ept: 119.5733
    Epoch [5/50], Test Losses: mse: 40.5058, mae: 4.1150, huber: 3.6619, swd: 8.9494, ept: 122.3982
      Epoch 5 composite train-obj: 3.275048
            Val objective improved 4.2445 → 4.0368, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 33.4803, mae: 3.5640, huber: 3.1209, swd: 6.0479, ept: 170.6239
    Epoch [6/50], Val Losses: mse: 45.3136, mae: 4.3886, huber: 3.9330, swd: 7.1890, ept: 130.5214
    Epoch [6/50], Test Losses: mse: 38.2139, mae: 3.9772, huber: 3.5244, swd: 7.9384, ept: 138.3789
      Epoch 6 composite train-obj: 3.120916
            Val objective improved 4.0368 → 3.9330, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 31.0472, mae: 3.3560, huber: 2.9181, swd: 5.1689, ept: 179.6596
    Epoch [7/50], Val Losses: mse: 43.6611, mae: 4.2280, huber: 3.7766, swd: 6.9814, ept: 130.7569
    Epoch [7/50], Test Losses: mse: 36.6659, mae: 3.8111, huber: 3.3633, swd: 6.6915, ept: 142.6105
      Epoch 7 composite train-obj: 2.918090
            Val objective improved 3.9330 → 3.7766, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 29.4858, mae: 3.2451, huber: 2.8087, swd: 4.6254, ept: 183.8094
    Epoch [8/50], Val Losses: mse: 42.6394, mae: 4.1894, huber: 3.7354, swd: 5.7816, ept: 137.3406
    Epoch [8/50], Test Losses: mse: 35.8766, mae: 3.7725, huber: 3.3225, swd: 5.9122, ept: 147.6188
      Epoch 8 composite train-obj: 2.808679
            Val objective improved 3.7766 → 3.7354, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 27.9703, mae: 3.1348, huber: 2.7004, swd: 4.1856, ept: 187.8482
    Epoch [9/50], Val Losses: mse: 44.4806, mae: 4.2333, huber: 3.7809, swd: 5.7786, ept: 138.9490
    Epoch [9/50], Test Losses: mse: 35.1836, mae: 3.6848, huber: 3.2388, swd: 5.6954, ept: 153.7191
      Epoch 9 composite train-obj: 2.700397
            No improvement (3.7809), counter 1/5
    Epoch [10/50], Train Losses: mse: 26.3952, mae: 3.0033, huber: 2.5726, swd: 3.7548, ept: 193.5835
    Epoch [10/50], Val Losses: mse: 40.2985, mae: 3.9880, huber: 3.5393, swd: 5.3679, ept: 147.5025
    Epoch [10/50], Test Losses: mse: 33.4268, mae: 3.5703, huber: 3.1257, swd: 5.4074, ept: 162.1206
      Epoch 10 composite train-obj: 2.572590
            Val objective improved 3.7354 → 3.5393, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 25.3089, mae: 2.9246, huber: 2.4951, swd: 3.4780, ept: 197.4785
    Epoch [11/50], Val Losses: mse: 44.5734, mae: 4.1101, huber: 3.6613, swd: 4.0331, ept: 148.3415
    Epoch [11/50], Test Losses: mse: 33.5713, mae: 3.5125, huber: 3.0703, swd: 4.0507, ept: 163.4653
      Epoch 11 composite train-obj: 2.495054
            No improvement (3.6613), counter 1/5
    Epoch [12/50], Train Losses: mse: 23.6298, mae: 2.7854, huber: 2.3599, swd: 3.1280, ept: 204.4961
    Epoch [12/50], Val Losses: mse: 39.3726, mae: 3.8791, huber: 3.4324, swd: 4.5439, ept: 153.6235
    Epoch [12/50], Test Losses: mse: 32.1746, mae: 3.4512, huber: 3.0090, swd: 4.4223, ept: 165.6534
      Epoch 12 composite train-obj: 2.359896
            Val objective improved 3.5393 → 3.4324, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 22.6769, mae: 2.7107, huber: 2.2872, swd: 2.9027, ept: 208.5141
    Epoch [13/50], Val Losses: mse: 42.6674, mae: 3.9531, huber: 3.5090, swd: 3.9920, ept: 158.4849
    Epoch [13/50], Test Losses: mse: 32.4679, mae: 3.4070, huber: 2.9678, swd: 4.0894, ept: 172.9191
      Epoch 13 composite train-obj: 2.287181
            No improvement (3.5090), counter 1/5
    Epoch [14/50], Train Losses: mse: 21.7269, mae: 2.6427, huber: 2.2202, swd: 2.7106, ept: 212.3427
    Epoch [14/50], Val Losses: mse: 39.2122, mae: 3.7975, huber: 3.3529, swd: 3.7378, ept: 161.9453
    Epoch [14/50], Test Losses: mse: 31.9317, mae: 3.3625, huber: 2.9235, swd: 3.8156, ept: 174.8344
      Epoch 14 composite train-obj: 2.220155
            Val objective improved 3.4324 → 3.3529, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 20.8017, mae: 2.5782, huber: 2.1570, swd: 2.4942, ept: 214.7911
    Epoch [15/50], Val Losses: mse: 37.7639, mae: 3.6770, huber: 3.2353, swd: 3.2452, ept: 163.7198
    Epoch [15/50], Test Losses: mse: 31.2666, mae: 3.2766, huber: 2.8410, swd: 3.3228, ept: 181.4012
      Epoch 15 composite train-obj: 2.156956
            Val objective improved 3.3529 → 3.2353, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 19.3005, mae: 2.4557, huber: 2.0377, swd: 2.2940, ept: 220.8750
    Epoch [16/50], Val Losses: mse: 37.5301, mae: 3.6430, huber: 3.2050, swd: 3.5513, ept: 172.5527
    Epoch [16/50], Test Losses: mse: 30.9476, mae: 3.2062, huber: 2.7738, swd: 3.2915, ept: 193.0235
      Epoch 16 composite train-obj: 2.037743
            Val objective improved 3.2353 → 3.2050, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 18.5940, mae: 2.3897, huber: 1.9749, swd: 2.0893, ept: 223.4851
    Epoch [17/50], Val Losses: mse: 38.6304, mae: 3.7322, huber: 3.2890, swd: 3.5121, ept: 167.9022
    Epoch [17/50], Test Losses: mse: 32.9463, mae: 3.3950, huber: 2.9561, swd: 3.3387, ept: 180.6925
      Epoch 17 composite train-obj: 1.974898
            No improvement (3.2890), counter 1/5
    Epoch [18/50], Train Losses: mse: 18.7345, mae: 2.4141, huber: 1.9970, swd: 2.0446, ept: 223.8276
    Epoch [18/50], Val Losses: mse: 38.4504, mae: 3.6636, huber: 3.2226, swd: 3.2368, ept: 169.6060
    Epoch [18/50], Test Losses: mse: 31.6720, mae: 3.2670, huber: 2.8311, swd: 2.9538, ept: 187.7499
      Epoch 18 composite train-obj: 1.997027
            No improvement (3.2226), counter 2/5
    Epoch [19/50], Train Losses: mse: 17.5833, mae: 2.3077, huber: 1.8947, swd: 1.8914, ept: 229.6107
    Epoch [19/50], Val Losses: mse: 37.7223, mae: 3.5794, huber: 3.1414, swd: 2.9289, ept: 173.1133
    Epoch [19/50], Test Losses: mse: 29.6954, mae: 3.1235, huber: 2.6925, swd: 2.8383, ept: 194.5743
      Epoch 19 composite train-obj: 1.894704
            Val objective improved 3.2050 → 3.1414, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 16.4827, mae: 2.2097, huber: 1.8005, swd: 1.7163, ept: 234.4254
    Epoch [20/50], Val Losses: mse: 34.2568, mae: 3.4198, huber: 2.9841, swd: 3.1372, ept: 176.4659
    Epoch [20/50], Test Losses: mse: 28.7477, mae: 3.0403, huber: 2.6106, swd: 2.6106, ept: 201.3211
      Epoch 20 composite train-obj: 1.800490
            Val objective improved 3.1414 → 2.9841, saving checkpoint.
    Epoch [21/50], Train Losses: mse: 16.7059, mae: 2.2271, huber: 1.8165, swd: 1.6966, ept: 234.7650
    Epoch [21/50], Val Losses: mse: 39.8491, mae: 3.6552, huber: 3.2181, swd: 3.0240, ept: 174.7502
    Epoch [21/50], Test Losses: mse: 31.4059, mae: 3.1665, huber: 2.7364, swd: 2.7409, ept: 196.4068
      Epoch 21 composite train-obj: 1.816506
            No improvement (3.2181), counter 1/5
    Epoch [22/50], Train Losses: mse: 15.9601, mae: 2.1811, huber: 1.7706, swd: 1.6648, ept: 237.4673
    Epoch [22/50], Val Losses: mse: 37.0544, mae: 3.5354, huber: 3.0960, swd: 2.8453, ept: 183.1594
    Epoch [22/50], Test Losses: mse: 32.1239, mae: 3.2064, huber: 2.7722, swd: 2.5593, ept: 201.3812
      Epoch 22 composite train-obj: 1.770604
            No improvement (3.0960), counter 2/5
    Epoch [23/50], Train Losses: mse: 15.2667, mae: 2.1040, huber: 1.6977, swd: 1.5114, ept: 241.5891
    Epoch [23/50], Val Losses: mse: 35.9831, mae: 3.4219, huber: 2.9900, swd: 2.7292, ept: 185.8998
    Epoch [23/50], Test Losses: mse: 29.8397, mae: 3.0271, huber: 2.6001, swd: 2.3551, ept: 208.0631
      Epoch 23 composite train-obj: 1.697721
            No improvement (2.9900), counter 3/5
    Epoch [24/50], Train Losses: mse: 14.9088, mae: 2.0610, huber: 1.6571, swd: 1.4443, ept: 245.1004
    Epoch [24/50], Val Losses: mse: 35.9535, mae: 3.4072, huber: 2.9751, swd: 2.8742, ept: 188.6558
    Epoch [24/50], Test Losses: mse: 28.3946, mae: 2.9336, huber: 2.5079, swd: 2.4194, ept: 211.0704
      Epoch 24 composite train-obj: 1.657134
            Val objective improved 2.9841 → 2.9751, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 14.2564, mae: 2.0048, huber: 1.6036, swd: 1.3540, ept: 247.1213
    Epoch [25/50], Val Losses: mse: 37.7312, mae: 3.4757, huber: 3.0437, swd: 2.7169, ept: 192.1385
    Epoch [25/50], Test Losses: mse: 29.5029, mae: 3.0104, huber: 2.5837, swd: 2.1955, ept: 208.8755
      Epoch 25 composite train-obj: 1.603622
            No improvement (3.0437), counter 1/5
    Epoch [26/50], Train Losses: mse: 14.0397, mae: 1.9927, huber: 1.5911, swd: 1.3210, ept: 248.5796
    Epoch [26/50], Val Losses: mse: 37.3372, mae: 3.4874, huber: 3.0538, swd: 2.4741, ept: 187.5971
    Epoch [26/50], Test Losses: mse: 27.9937, mae: 2.9352, huber: 2.5105, swd: 2.1564, ept: 209.6285
      Epoch 26 composite train-obj: 1.591068
            No improvement (3.0538), counter 2/5
    Epoch [27/50], Train Losses: mse: 13.6586, mae: 1.9497, huber: 1.5505, swd: 1.2651, ept: 251.0409
    Epoch [27/50], Val Losses: mse: 37.5860, mae: 3.4878, huber: 3.0546, swd: 2.6746, ept: 190.4229
    Epoch [27/50], Test Losses: mse: 28.7636, mae: 2.9801, huber: 2.5534, swd: 2.2089, ept: 208.6388
      Epoch 27 composite train-obj: 1.550547
            No improvement (3.0546), counter 3/5
    Epoch [28/50], Train Losses: mse: 13.7797, mae: 1.9630, huber: 1.5629, swd: 1.2699, ept: 250.8685
    Epoch [28/50], Val Losses: mse: 35.2928, mae: 3.3726, huber: 2.9399, swd: 2.5490, ept: 192.2242
    Epoch [28/50], Test Losses: mse: 28.9863, mae: 2.9606, huber: 2.5348, swd: 2.3484, ept: 211.8403
      Epoch 28 composite train-obj: 1.562921
            Val objective improved 2.9751 → 2.9399, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 13.3468, mae: 1.9344, huber: 1.5341, swd: 1.2274, ept: 253.0174
    Epoch [29/50], Val Losses: mse: 36.3863, mae: 3.4174, huber: 2.9835, swd: 2.5928, ept: 191.8600
    Epoch [29/50], Test Losses: mse: 27.9027, mae: 2.9036, huber: 2.4780, swd: 2.2434, ept: 214.3280
      Epoch 29 composite train-obj: 1.534130
            No improvement (2.9835), counter 1/5
    Epoch [30/50], Train Losses: mse: 12.8657, mae: 1.8802, huber: 1.4833, swd: 1.1757, ept: 256.7030
    Epoch [30/50], Val Losses: mse: 34.9881, mae: 3.2951, huber: 2.8687, swd: 2.2838, ept: 198.2268
    Epoch [30/50], Test Losses: mse: 28.8215, mae: 2.8882, huber: 2.4691, swd: 2.1156, ept: 217.9375
      Epoch 30 composite train-obj: 1.483287
            Val objective improved 2.9399 → 2.8687, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 12.5445, mae: 1.8381, huber: 1.4441, swd: 1.1086, ept: 258.9907
    Epoch [31/50], Val Losses: mse: 37.2470, mae: 3.3529, huber: 2.9272, swd: 2.1634, ept: 205.4472
    Epoch [31/50], Test Losses: mse: 28.1479, mae: 2.8284, huber: 2.4103, swd: 1.8438, ept: 224.3478
      Epoch 31 composite train-obj: 1.444107
            No improvement (2.9272), counter 1/5
    Epoch [32/50], Train Losses: mse: 11.9573, mae: 1.7920, huber: 1.3998, swd: 1.0617, ept: 260.8805
    Epoch [32/50], Val Losses: mse: 36.0296, mae: 3.3804, huber: 2.9498, swd: 2.2474, ept: 198.8019
    Epoch [32/50], Test Losses: mse: 26.0960, mae: 2.8040, huber: 2.3808, swd: 1.8162, ept: 218.0941
      Epoch 32 composite train-obj: 1.399774
            No improvement (2.9498), counter 2/5
    Epoch [33/50], Train Losses: mse: 12.0991, mae: 1.8119, huber: 1.4180, swd: 1.0665, ept: 260.3712
    Epoch [33/50], Val Losses: mse: 37.2255, mae: 3.3644, huber: 2.9366, swd: 2.3705, ept: 202.2055
    Epoch [33/50], Test Losses: mse: 29.3282, mae: 2.8941, huber: 2.4735, swd: 1.9582, ept: 222.1447
      Epoch 33 composite train-obj: 1.417955
            No improvement (2.9366), counter 3/5
    Epoch [34/50], Train Losses: mse: 11.6792, mae: 1.7715, huber: 1.3797, swd: 1.0194, ept: 262.8252
    Epoch [34/50], Val Losses: mse: 37.8736, mae: 3.4478, huber: 3.0168, swd: 2.2045, ept: 196.4713
    Epoch [34/50], Test Losses: mse: 28.9800, mae: 2.9283, huber: 2.5037, swd: 1.9069, ept: 215.0878
      Epoch 34 composite train-obj: 1.379708
            No improvement (3.0168), counter 4/5
    Epoch [35/50], Train Losses: mse: 11.9070, mae: 1.7891, huber: 1.3966, swd: 1.0310, ept: 263.2110
    Epoch [35/50], Val Losses: mse: 35.6678, mae: 3.2570, huber: 2.8348, swd: 1.9717, ept: 207.4763
    Epoch [35/50], Test Losses: mse: 26.4676, mae: 2.6856, huber: 2.2734, swd: 1.6294, ept: 232.6235
      Epoch 35 composite train-obj: 1.396623
            Val objective improved 2.8687 → 2.8348, saving checkpoint.
    Epoch [36/50], Train Losses: mse: 11.5828, mae: 1.7496, huber: 1.3600, swd: 0.9684, ept: 264.8825
    Epoch [36/50], Val Losses: mse: 35.3195, mae: 3.3177, huber: 2.8899, swd: 1.9668, ept: 202.3190
    Epoch [36/50], Test Losses: mse: 27.8433, mae: 2.8304, huber: 2.4107, swd: 1.7438, ept: 221.9068
      Epoch 36 composite train-obj: 1.360001
            No improvement (2.8899), counter 1/5
    Epoch [37/50], Train Losses: mse: 11.1695, mae: 1.6963, huber: 1.3107, swd: 0.9530, ept: 267.6674
    Epoch [37/50], Val Losses: mse: 35.6060, mae: 3.3492, huber: 2.9189, swd: 2.1482, ept: 195.5863
    Epoch [37/50], Test Losses: mse: 26.9526, mae: 2.7773, huber: 2.3594, swd: 1.6591, ept: 227.1685
      Epoch 37 composite train-obj: 1.310712
            No improvement (2.9189), counter 2/5
    Epoch [38/50], Train Losses: mse: 11.4756, mae: 1.7471, huber: 1.3570, swd: 0.9602, ept: 265.4766
    Epoch [38/50], Val Losses: mse: 35.7051, mae: 3.2642, huber: 2.8413, swd: 2.1646, ept: 206.9573
    Epoch [38/50], Test Losses: mse: 29.2459, mae: 2.8529, huber: 2.4380, swd: 1.8810, ept: 226.0106
      Epoch 38 composite train-obj: 1.356983
            No improvement (2.8413), counter 3/5
    Epoch [39/50], Train Losses: mse: 11.6054, mae: 1.7544, huber: 1.3635, swd: 0.9762, ept: 265.7951
    Epoch [39/50], Val Losses: mse: 36.2449, mae: 3.3648, huber: 2.9298, swd: 2.1334, ept: 204.6886
    Epoch [39/50], Test Losses: mse: 29.2264, mae: 2.9047, huber: 2.4788, swd: 1.7844, ept: 225.8078
      Epoch 39 composite train-obj: 1.363520
            No improvement (2.9298), counter 4/5
    Epoch [40/50], Train Losses: mse: 10.9780, mae: 1.6852, huber: 1.2989, swd: 0.9136, ept: 269.8881
    Epoch [40/50], Val Losses: mse: 34.1704, mae: 3.1778, huber: 2.7560, swd: 2.0085, ept: 210.4922
    Epoch [40/50], Test Losses: mse: 25.9910, mae: 2.6378, huber: 2.2272, swd: 1.6286, ept: 238.0950
      Epoch 40 composite train-obj: 1.298944
            Val objective improved 2.8348 → 2.7560, saving checkpoint.
    Epoch [41/50], Train Losses: mse: 11.0332, mae: 1.6864, huber: 1.3004, swd: 0.9144, ept: 269.7549
    Epoch [41/50], Val Losses: mse: 36.6993, mae: 3.3176, huber: 2.8928, swd: 2.0094, ept: 209.7745
    Epoch [41/50], Test Losses: mse: 25.1777, mae: 2.6386, huber: 2.2254, swd: 1.5913, ept: 234.3624
      Epoch 41 composite train-obj: 1.300436
            No improvement (2.8928), counter 1/5
    Epoch [42/50], Train Losses: mse: 9.8378, mae: 1.5749, huber: 1.1956, swd: 0.8208, ept: 274.2474
    Epoch [42/50], Val Losses: mse: 37.6611, mae: 3.3927, huber: 2.9665, swd: 1.9656, ept: 203.7693
    Epoch [42/50], Test Losses: mse: 28.2278, mae: 2.8073, huber: 2.3925, swd: 1.7031, ept: 226.0311
      Epoch 42 composite train-obj: 1.195571
            No improvement (2.9665), counter 2/5
    Epoch [43/50], Train Losses: mse: 10.7005, mae: 1.6537, huber: 1.2695, swd: 0.8988, ept: 271.0843
    Epoch [43/50], Val Losses: mse: 35.3208, mae: 3.2744, huber: 2.8486, swd: 1.9105, ept: 207.6372
    Epoch [43/50], Test Losses: mse: 27.1901, mae: 2.7288, huber: 2.3156, swd: 1.6128, ept: 232.6942
      Epoch 43 composite train-obj: 1.269459
            No improvement (2.8486), counter 3/5
    Epoch [44/50], Train Losses: mse: 10.4767, mae: 1.6393, huber: 1.2548, swd: 0.8548, ept: 272.1911
    Epoch [44/50], Val Losses: mse: 38.5294, mae: 3.4148, huber: 2.9874, swd: 1.9302, ept: 201.3061
    Epoch [44/50], Test Losses: mse: 27.7720, mae: 2.8015, huber: 2.3838, swd: 1.6605, ept: 228.1532
      Epoch 44 composite train-obj: 1.254845
            No improvement (2.9874), counter 4/5
    Epoch [45/50], Train Losses: mse: 10.1975, mae: 1.6077, huber: 1.2253, swd: 0.8136, ept: 273.8696
    Epoch [45/50], Val Losses: mse: 34.8008, mae: 3.2053, huber: 2.7820, swd: 1.8295, ept: 215.6340
    Epoch [45/50], Test Losses: mse: 25.8746, mae: 2.6733, huber: 2.2602, swd: 1.4846, ept: 235.0493
      Epoch 45 composite train-obj: 1.225300
    Epoch [45/50], Test Losses: mse: 25.9915, mae: 2.6378, huber: 2.2272, swd: 1.6292, ept: 238.0831
    Best round's Test MSE: 25.9910, MAE: 2.6378, SWD: 1.6286
    Best round's Validation MSE: 34.1704, MAE: 3.1778, SWD: 2.0085
    Best round's Test verification MSE : 25.9915, MAE: 2.6378, SWD: 1.6292
    Time taken: 180.22 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1824)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 28.7104 ± 2.0807
      mae: 2.9504 ± 0.2494
      huber: 2.5274 ± 0.2397
      swd: 2.2205 ± 0.4992
      ept: 212.4683 ± 19.8014
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 35.3243 ± 2.6361
      mae: 3.3747 ± 0.2163
      huber: 2.9444 ± 0.2100
      swd: 2.3842 ± 0.2741
      ept: 193.1044 ± 12.7169
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 374.36 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1824
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: dist y0: A(y0+t)
Performance is improved; not surprising, since A() alone does better than t().
On datasets 


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
    # ablate_shift_inside_scale=False,
    householder_reflects_latent = 2,
    householder_reflects_data = 4,
    mixing_strategy='delay_only', 
    loss_backward_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    loss_validate_weights = [0.0, 0.0, 1.0, 0.0, 0.0],
    ablate_deterministic_y0=False, ### HERE
    ablate_shift_inside_scale=True, ### HERE Note: they must be True together
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
    
    Epoch [1/50], Train Losses: mse: 76.8604, mae: 6.6733, huber: 6.1923, swd: 44.5487, ept: 29.5391
    Epoch [1/50], Val Losses: mse: 61.6740, mae: 5.9940, huber: 5.5160, swd: 28.9132, ept: 33.5353
    Epoch [1/50], Test Losses: mse: 57.4829, mae: 5.6348, huber: 5.1600, swd: 28.1719, ept: 33.5553
      Epoch 1 composite train-obj: 6.192327
            Val objective improved inf → 5.5160, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 49.1301, mae: 4.9295, huber: 4.4632, swd: 17.2236, ept: 87.6206
    Epoch [2/50], Val Losses: mse: 51.8690, mae: 5.0865, huber: 4.6199, swd: 13.5470, ept: 85.7954
    Epoch [2/50], Test Losses: mse: 47.5056, mae: 4.7452, huber: 4.2817, swd: 14.0896, ept: 79.8105
      Epoch 2 composite train-obj: 4.463202
            Val objective improved 5.5160 → 4.6199, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 40.7062, mae: 4.1729, huber: 3.7189, swd: 9.2586, ept: 128.8894
    Epoch [3/50], Val Losses: mse: 44.0055, mae: 4.4814, huber: 4.0223, swd: 8.4014, ept: 109.3295
    Epoch [3/50], Test Losses: mse: 40.7361, mae: 4.2321, huber: 3.7741, swd: 8.8934, ept: 112.4098
      Epoch 3 composite train-obj: 3.718946
            Val objective improved 4.6199 → 4.0223, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.6687, mae: 3.7502, huber: 3.3036, swd: 5.8231, ept: 159.2186
    Epoch [4/50], Val Losses: mse: 40.0648, mae: 4.1460, huber: 3.6933, swd: 7.1230, ept: 129.7816
    Epoch [4/50], Test Losses: mse: 37.3794, mae: 3.9356, huber: 3.4842, swd: 7.5083, ept: 134.4553
      Epoch 4 composite train-obj: 3.303593
            Val objective improved 4.0223 → 3.6933, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.4291, mae: 3.4549, huber: 3.0159, swd: 4.4832, ept: 180.1118
    Epoch [5/50], Val Losses: mse: 41.6306, mae: 4.1371, huber: 3.6849, swd: 5.4809, ept: 139.2945
    Epoch [5/50], Test Losses: mse: 36.7339, mae: 3.8336, huber: 3.3838, swd: 5.6206, ept: 145.1883
      Epoch 5 composite train-obj: 3.015890
            Val objective improved 3.6933 → 3.6849, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.4888, mae: 3.2895, huber: 2.8552, swd: 3.8929, ept: 189.5743
    Epoch [6/50], Val Losses: mse: 37.8567, mae: 3.8519, huber: 3.4053, swd: 4.3833, ept: 149.8458
    Epoch [6/50], Test Losses: mse: 34.2410, mae: 3.6368, huber: 3.1920, swd: 4.5795, ept: 156.8546
      Epoch 6 composite train-obj: 2.855221
            Val objective improved 3.6849 → 3.4053, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.7996, mae: 3.1441, huber: 2.7135, swd: 3.4075, ept: 196.0277
    Epoch [7/50], Val Losses: mse: 39.3569, mae: 3.8457, huber: 3.4015, swd: 4.1540, ept: 155.7326
    Epoch [7/50], Test Losses: mse: 33.2697, mae: 3.5124, huber: 3.0700, swd: 3.8467, ept: 162.2580
      Epoch 7 composite train-obj: 2.713467
            Val objective improved 3.4053 → 3.4015, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 27.3806, mae: 3.0252, huber: 2.5976, swd: 3.0016, ept: 202.4618
    Epoch [8/50], Val Losses: mse: 36.1319, mae: 3.6708, huber: 3.2303, swd: 4.0245, ept: 166.5853
    Epoch [8/50], Test Losses: mse: 31.0571, mae: 3.3522, huber: 2.9146, swd: 3.7163, ept: 178.7658
      Epoch 8 composite train-obj: 2.597581
            Val objective improved 3.4015 → 3.2303, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 25.6500, mae: 2.8956, huber: 2.4710, swd: 2.6544, ept: 208.1186
    Epoch [9/50], Val Losses: mse: 39.0497, mae: 3.7770, huber: 3.3363, swd: 3.5026, ept: 170.3640
    Epoch [9/50], Test Losses: mse: 29.7524, mae: 3.2576, huber: 2.8214, swd: 3.2792, ept: 184.1696
      Epoch 9 composite train-obj: 2.471003
            No improvement (3.3363), counter 1/5
    Epoch [10/50], Train Losses: mse: 24.7453, mae: 2.8284, huber: 2.4048, swd: 2.4604, ept: 211.9086
    Epoch [10/50], Val Losses: mse: 36.6682, mae: 3.5881, huber: 3.1520, swd: 3.3008, ept: 178.1586
    Epoch [10/50], Test Losses: mse: 27.6300, mae: 3.0715, huber: 2.6410, swd: 2.7333, ept: 196.6357
      Epoch 10 composite train-obj: 2.404848
            Val objective improved 3.2303 → 3.1520, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 23.1456, mae: 2.6863, huber: 2.2675, swd: 2.2155, ept: 218.6605
    Epoch [11/50], Val Losses: mse: 34.7126, mae: 3.4774, huber: 3.0433, swd: 3.0766, ept: 183.1666
    Epoch [11/50], Test Losses: mse: 28.0891, mae: 3.0935, huber: 2.6633, swd: 2.9566, ept: 191.0661
      Epoch 11 composite train-obj: 2.267533
            Val objective improved 3.1520 → 3.0433, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 21.8036, mae: 2.5815, huber: 2.1659, swd: 1.9854, ept: 223.8132
    Epoch [12/50], Val Losses: mse: 35.6900, mae: 3.4242, huber: 2.9927, swd: 2.2819, ept: 187.6161
    Epoch [12/50], Test Losses: mse: 27.5068, mae: 3.0149, huber: 2.5866, swd: 2.3992, ept: 199.7811
      Epoch 12 composite train-obj: 2.165940
            Val objective improved 3.0433 → 2.9927, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 20.9737, mae: 2.5138, huber: 2.1002, swd: 1.8818, ept: 227.0344
    Epoch [13/50], Val Losses: mse: 35.1159, mae: 3.4645, huber: 3.0311, swd: 2.9899, ept: 186.0233
    Epoch [13/50], Test Losses: mse: 28.2369, mae: 3.0681, huber: 2.6383, swd: 2.7814, ept: 200.5188
      Epoch 13 composite train-obj: 2.100157
            No improvement (3.0311), counter 1/5
    Epoch [14/50], Train Losses: mse: 20.2476, mae: 2.4501, huber: 2.0383, swd: 1.7600, ept: 230.8516
    Epoch [14/50], Val Losses: mse: 35.8156, mae: 3.4275, huber: 2.9970, swd: 2.5252, ept: 192.4044
    Epoch [14/50], Test Losses: mse: 26.0908, mae: 2.8998, huber: 2.4752, swd: 2.3056, ept: 210.7741
      Epoch 14 composite train-obj: 2.038295
            No improvement (2.9970), counter 2/5
    Epoch [15/50], Train Losses: mse: 18.5278, mae: 2.3081, huber: 1.9019, swd: 1.5558, ept: 236.9314
    Epoch [15/50], Val Losses: mse: 33.3390, mae: 3.2718, huber: 2.8456, swd: 2.5417, ept: 199.5198
    Epoch [15/50], Test Losses: mse: 25.2661, mae: 2.8043, huber: 2.3833, swd: 2.0142, ept: 215.2215
      Epoch 15 composite train-obj: 1.901911
            Val objective improved 2.9927 → 2.8456, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 17.6308, mae: 2.2372, huber: 1.8333, swd: 1.4518, ept: 240.1018
    Epoch [16/50], Val Losses: mse: 32.6627, mae: 3.2065, huber: 2.7825, swd: 2.5105, ept: 201.7511
    Epoch [16/50], Test Losses: mse: 25.0644, mae: 2.7717, huber: 2.3532, swd: 2.1969, ept: 217.7384
      Epoch 16 composite train-obj: 1.833295
            Val objective improved 2.8456 → 2.7825, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 16.9859, mae: 2.1840, huber: 1.7818, swd: 1.3545, ept: 242.7341
    Epoch [17/50], Val Losses: mse: 33.6653, mae: 3.2358, huber: 2.8076, swd: 2.0993, ept: 204.3323
    Epoch [17/50], Test Losses: mse: 27.3671, mae: 2.9279, huber: 2.5015, swd: 1.9380, ept: 212.1991
      Epoch 17 composite train-obj: 1.781797
            No improvement (2.8076), counter 1/5
    Epoch [18/50], Train Losses: mse: 16.0348, mae: 2.1075, huber: 1.7079, swd: 1.2621, ept: 246.6513
    Epoch [18/50], Val Losses: mse: 30.9936, mae: 3.1053, huber: 2.6819, swd: 2.2249, ept: 206.1227
    Epoch [18/50], Test Losses: mse: 24.6698, mae: 2.7347, huber: 2.3175, swd: 2.0573, ept: 220.0415
      Epoch 18 composite train-obj: 1.707924
            Val objective improved 2.7825 → 2.6819, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 15.4272, mae: 2.0452, huber: 1.6490, swd: 1.1946, ept: 250.7460
    Epoch [19/50], Val Losses: mse: 30.2094, mae: 3.0430, huber: 2.6211, swd: 2.1674, ept: 211.0118
    Epoch [19/50], Test Losses: mse: 24.9683, mae: 2.7116, huber: 2.2946, swd: 1.9091, ept: 223.5141
      Epoch 19 composite train-obj: 1.649013
            Val objective improved 2.6819 → 2.6211, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 15.1457, mae: 2.0140, huber: 1.6195, swd: 1.1571, ept: 253.2546
    Epoch [20/50], Val Losses: mse: 35.2467, mae: 3.2527, huber: 2.8308, swd: 1.9845, ept: 205.7895
    Epoch [20/50], Test Losses: mse: 24.3468, mae: 2.6614, huber: 2.2460, swd: 1.6882, ept: 229.7773
      Epoch 20 composite train-obj: 1.619493
            No improvement (2.8308), counter 1/5
    Epoch [21/50], Train Losses: mse: 14.7956, mae: 1.9913, huber: 1.5970, swd: 1.1240, ept: 254.4352
    Epoch [21/50], Val Losses: mse: 33.4636, mae: 3.1710, huber: 2.7495, swd: 2.2182, ept: 207.1443
    Epoch [21/50], Test Losses: mse: 27.5796, mae: 2.8185, huber: 2.4003, swd: 1.6827, ept: 225.7046
      Epoch 21 composite train-obj: 1.596998
            No improvement (2.7495), counter 2/5
    Epoch [22/50], Train Losses: mse: 14.4743, mae: 1.9617, huber: 1.5684, swd: 1.0975, ept: 256.7096
    Epoch [22/50], Val Losses: mse: 34.1826, mae: 3.1911, huber: 2.7700, swd: 2.2658, ept: 212.4143
    Epoch [22/50], Test Losses: mse: 23.2596, mae: 2.6050, huber: 2.1894, swd: 1.8347, ept: 233.5035
      Epoch 22 composite train-obj: 1.568396
            No improvement (2.7700), counter 3/5
    Epoch [23/50], Train Losses: mse: 13.4231, mae: 1.8586, huber: 1.4711, swd: 1.0039, ept: 262.0098
    Epoch [23/50], Val Losses: mse: 34.1590, mae: 3.1646, huber: 2.7440, swd: 2.2029, ept: 216.9589
    Epoch [23/50], Test Losses: mse: 23.7855, mae: 2.6173, huber: 2.2027, swd: 1.8122, ept: 232.1519
      Epoch 23 composite train-obj: 1.471097
            No improvement (2.7440), counter 4/5
    Epoch [24/50], Train Losses: mse: 13.6559, mae: 1.8820, huber: 1.4927, swd: 1.0022, ept: 261.3671
    Epoch [24/50], Val Losses: mse: 31.6069, mae: 3.0349, huber: 2.6189, swd: 1.9931, ept: 214.2598
    Epoch [24/50], Test Losses: mse: 25.2735, mae: 2.6348, huber: 2.2243, swd: 1.6470, ept: 236.1337
      Epoch 24 composite train-obj: 1.492706
            Val objective improved 2.6211 → 2.6189, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 13.1660, mae: 1.8318, huber: 1.4454, swd: 0.9653, ept: 264.1137
    Epoch [25/50], Val Losses: mse: 30.4515, mae: 2.9666, huber: 2.5524, swd: 1.7107, ept: 221.3038
    Epoch [25/50], Test Losses: mse: 23.4154, mae: 2.5208, huber: 2.1152, swd: 1.5255, ept: 240.7679
      Epoch 25 composite train-obj: 1.445438
            Val objective improved 2.6189 → 2.5524, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.7047, mae: 1.7826, huber: 1.4000, swd: 0.9151, ept: 266.0367
    Epoch [26/50], Val Losses: mse: 31.4961, mae: 3.0289, huber: 2.6105, swd: 1.8489, ept: 217.0879
    Epoch [26/50], Test Losses: mse: 22.3790, mae: 2.5168, huber: 2.1076, swd: 1.5367, ept: 236.6897
      Epoch 26 composite train-obj: 1.399961
            No improvement (2.6105), counter 1/5
    Epoch [27/50], Train Losses: mse: 13.3836, mae: 1.8509, huber: 1.4628, swd: 0.9792, ept: 264.4414
    Epoch [27/50], Val Losses: mse: 30.0161, mae: 2.8925, huber: 2.4815, swd: 1.6454, ept: 223.2633
    Epoch [27/50], Test Losses: mse: 21.4364, mae: 2.4003, huber: 1.9985, swd: 1.3378, ept: 246.1799
      Epoch 27 composite train-obj: 1.462774
            Val objective improved 2.5524 → 2.4815, saving checkpoint.
    Epoch [28/50], Train Losses: mse: 12.1031, mae: 1.7326, huber: 1.3513, swd: 0.8743, ept: 269.0966
    Epoch [28/50], Val Losses: mse: 31.3341, mae: 2.9724, huber: 2.5570, swd: 2.0514, ept: 221.2471
    Epoch [28/50], Test Losses: mse: 20.4064, mae: 2.3825, huber: 1.9751, swd: 1.5587, ept: 241.9361
      Epoch 28 composite train-obj: 1.351331
            No improvement (2.5570), counter 1/5
    Epoch [29/50], Train Losses: mse: 12.3282, mae: 1.7436, huber: 1.3623, swd: 0.8902, ept: 269.3800
    Epoch [29/50], Val Losses: mse: 30.4153, mae: 2.8955, huber: 2.4878, swd: 1.7752, ept: 224.4343
    Epoch [29/50], Test Losses: mse: 25.8633, mae: 2.6101, huber: 2.2052, swd: 1.5728, ept: 240.6696
      Epoch 29 composite train-obj: 1.362279
            No improvement (2.4878), counter 2/5
    Epoch [30/50], Train Losses: mse: 12.3584, mae: 1.7408, huber: 1.3602, swd: 0.8679, ept: 269.4328
    Epoch [30/50], Val Losses: mse: 31.6664, mae: 2.9609, huber: 2.5481, swd: 1.7585, ept: 228.6560
    Epoch [30/50], Test Losses: mse: 21.9265, mae: 2.4364, huber: 2.0293, swd: 1.3723, ept: 245.0897
      Epoch 30 composite train-obj: 1.360238
            No improvement (2.5481), counter 3/5
    Epoch [31/50], Train Losses: mse: 11.5201, mae: 1.6666, huber: 1.2899, swd: 0.8221, ept: 273.4709
    Epoch [31/50], Val Losses: mse: 32.0802, mae: 3.0093, huber: 2.5967, swd: 1.9997, ept: 222.1142
    Epoch [31/50], Test Losses: mse: 25.5353, mae: 2.5936, huber: 2.1874, swd: 1.5523, ept: 240.4522
      Epoch 31 composite train-obj: 1.289872
            No improvement (2.5967), counter 4/5
    Epoch [32/50], Train Losses: mse: 11.8243, mae: 1.6876, huber: 1.3102, swd: 0.8470, ept: 272.8320
    Epoch [32/50], Val Losses: mse: 30.6905, mae: 2.9339, huber: 2.5225, swd: 1.7989, ept: 221.7383
    Epoch [32/50], Test Losses: mse: 22.0096, mae: 2.4321, huber: 2.0272, swd: 1.2492, ept: 244.5689
      Epoch 32 composite train-obj: 1.310208
    Epoch [32/50], Test Losses: mse: 21.4403, mae: 2.4005, huber: 1.9987, swd: 1.3382, ept: 246.1821
    Best round's Test MSE: 21.4364, MAE: 2.4003, SWD: 1.3378
    Best round's Validation MSE: 30.0161, MAE: 2.8925, SWD: 1.6454
    Best round's Test verification MSE : 21.4403, MAE: 2.4005, SWD: 1.3382
    Time taken: 119.50 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.1810, mae: 6.4247, huber: 5.9460, swd: 42.1593, ept: 39.0961
    Epoch [1/50], Val Losses: mse: 56.2456, mae: 5.5513, huber: 5.0779, swd: 23.2181, ept: 58.0432
    Epoch [1/50], Test Losses: mse: 52.1037, mae: 5.2194, huber: 4.7492, swd: 24.8480, ept: 56.0651
      Epoch 1 composite train-obj: 5.945964
            Val objective improved inf → 5.0779, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 45.4550, mae: 4.6514, huber: 4.1889, swd: 13.6903, ept: 107.9848
    Epoch [2/50], Val Losses: mse: 48.2141, mae: 4.8520, huber: 4.3877, swd: 11.3968, ept: 100.7028
    Epoch [2/50], Test Losses: mse: 44.7640, mae: 4.5316, huber: 4.0714, swd: 12.5988, ept: 104.7706
      Epoch 2 composite train-obj: 4.188915
            Val objective improved 5.0779 → 4.3877, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.7243, mae: 4.0403, huber: 3.5883, swd: 8.1127, ept: 148.1442
    Epoch [3/50], Val Losses: mse: 44.9420, mae: 4.5322, huber: 4.0740, swd: 8.8324, ept: 113.5669
    Epoch [3/50], Test Losses: mse: 42.1356, mae: 4.3138, huber: 3.8566, swd: 8.8246, ept: 117.5072
      Epoch 3 composite train-obj: 3.588282
            Val objective improved 4.3877 → 4.0740, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.6468, mae: 3.6762, huber: 3.2318, swd: 5.8271, ept: 170.3274
    Epoch [4/50], Val Losses: mse: 41.6471, mae: 4.1984, huber: 3.7456, swd: 6.6560, ept: 139.2566
    Epoch [4/50], Test Losses: mse: 36.7986, mae: 3.9107, huber: 3.4596, swd: 7.3836, ept: 145.7440
      Epoch 4 composite train-obj: 3.231807
            Val objective improved 4.0740 → 3.7456, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.0041, mae: 3.4424, huber: 3.0033, swd: 4.7217, ept: 183.4179
    Epoch [5/50], Val Losses: mse: 41.3343, mae: 4.1284, huber: 3.6778, swd: 5.7309, ept: 147.7723
    Epoch [5/50], Test Losses: mse: 34.3985, mae: 3.6901, huber: 3.2431, swd: 5.1773, ept: 155.4894
      Epoch 5 composite train-obj: 3.003319
            Val objective improved 3.7456 → 3.6778, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.9539, mae: 3.2710, huber: 2.8355, swd: 3.9328, ept: 190.6448
    Epoch [6/50], Val Losses: mse: 40.3571, mae: 4.0272, huber: 3.5771, swd: 4.9749, ept: 150.5699
    Epoch [6/50], Test Losses: mse: 33.5086, mae: 3.6049, huber: 3.1592, swd: 5.0151, ept: 158.5400
      Epoch 6 composite train-obj: 2.835548
            Val objective improved 3.6778 → 3.5771, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.9226, mae: 3.1015, huber: 2.6701, swd: 3.3743, ept: 197.5159
    Epoch [7/50], Val Losses: mse: 42.9032, mae: 4.0583, huber: 3.6111, swd: 3.9789, ept: 159.4190
    Epoch [7/50], Test Losses: mse: 31.4683, mae: 3.4468, huber: 3.0036, swd: 4.0129, ept: 173.3074
      Epoch 7 composite train-obj: 2.670110
            No improvement (3.6111), counter 1/5
    Epoch [8/50], Train Losses: mse: 26.1502, mae: 2.9591, huber: 2.5308, swd: 2.9404, ept: 204.4813
    Epoch [8/50], Val Losses: mse: 41.7492, mae: 4.0313, huber: 3.5841, swd: 4.2697, ept: 161.6137
    Epoch [8/50], Test Losses: mse: 32.4864, mae: 3.4638, huber: 3.0221, swd: 3.6772, ept: 175.3064
      Epoch 8 composite train-obj: 2.530772
            No improvement (3.5841), counter 2/5
    Epoch [9/50], Train Losses: mse: 24.6151, mae: 2.8335, huber: 2.4086, swd: 2.6035, ept: 210.2750
    Epoch [9/50], Val Losses: mse: 39.8550, mae: 3.8422, huber: 3.4004, swd: 3.9329, ept: 169.3514
    Epoch [9/50], Test Losses: mse: 30.1856, mae: 3.2626, huber: 2.8255, swd: 3.5720, ept: 181.7624
      Epoch 9 composite train-obj: 2.408568
            Val objective improved 3.5771 → 3.4004, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 22.8648, mae: 2.6820, huber: 2.2623, swd: 2.3429, ept: 217.2030
    Epoch [10/50], Val Losses: mse: 38.7227, mae: 3.7099, huber: 3.2734, swd: 3.8868, ept: 176.8331
    Epoch [10/50], Test Losses: mse: 30.0843, mae: 3.2097, huber: 2.7774, swd: 3.4847, ept: 192.8154
      Epoch 10 composite train-obj: 2.262295
            Val objective improved 3.4004 → 3.2734, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.7193, mae: 2.6143, huber: 2.1946, swd: 2.0899, ept: 219.8829
    Epoch [11/50], Val Losses: mse: 42.3497, mae: 3.9392, huber: 3.4996, swd: 4.2288, ept: 173.2888
    Epoch [11/50], Test Losses: mse: 28.4648, mae: 3.1532, huber: 2.7191, swd: 3.4955, ept: 191.4559
      Epoch 11 composite train-obj: 2.194645
            No improvement (3.4996), counter 1/5
    Epoch [12/50], Train Losses: mse: 20.9766, mae: 2.5396, huber: 2.1228, swd: 2.0025, ept: 224.6496
    Epoch [12/50], Val Losses: mse: 43.9569, mae: 3.9489, huber: 3.5087, swd: 3.4163, ept: 176.1854
    Epoch [12/50], Test Losses: mse: 30.2764, mae: 3.1992, huber: 2.7643, swd: 2.5841, ept: 194.2114
      Epoch 12 composite train-obj: 2.122804
            No improvement (3.5087), counter 2/5
    Epoch [13/50], Train Losses: mse: 19.1024, mae: 2.3867, huber: 1.9749, swd: 1.7298, ept: 231.0604
    Epoch [13/50], Val Losses: mse: 41.5489, mae: 3.7388, huber: 3.3063, swd: 2.9365, ept: 186.1339
    Epoch [13/50], Test Losses: mse: 28.0081, mae: 2.9950, huber: 2.5700, swd: 2.5317, ept: 207.3524
      Epoch 13 composite train-obj: 1.974945
            No improvement (3.3063), counter 3/5
    Epoch [14/50], Train Losses: mse: 18.4212, mae: 2.3351, huber: 1.9249, swd: 1.6388, ept: 233.4049
    Epoch [14/50], Val Losses: mse: 39.5184, mae: 3.6285, huber: 3.1962, swd: 2.7528, ept: 189.2023
    Epoch [14/50], Test Losses: mse: 29.5900, mae: 3.1069, huber: 2.6778, swd: 2.5686, ept: 199.4418
      Epoch 14 composite train-obj: 1.924862
            Val objective improved 3.2734 → 3.1962, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.2208, mae: 2.2319, huber: 1.8254, swd: 1.4856, ept: 239.2919
    Epoch [15/50], Val Losses: mse: 37.5443, mae: 3.5149, huber: 3.0862, swd: 2.6430, ept: 192.2992
    Epoch [15/50], Test Losses: mse: 27.6093, mae: 2.9579, huber: 2.5341, swd: 2.2115, ept: 208.2123
      Epoch 15 composite train-obj: 1.825365
            Val objective improved 3.1962 → 3.0862, saving checkpoint.
    Epoch [16/50], Train Losses: mse: 16.5095, mae: 2.1657, huber: 1.7617, swd: 1.3852, ept: 242.8565
    Epoch [16/50], Val Losses: mse: 36.2598, mae: 3.4791, huber: 3.0470, swd: 2.5209, ept: 192.3206
    Epoch [16/50], Test Losses: mse: 30.2939, mae: 3.1251, huber: 2.6970, swd: 2.4207, ept: 201.3313
      Epoch 16 composite train-obj: 1.761744
            Val objective improved 3.0862 → 3.0470, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 16.6514, mae: 2.1844, huber: 1.7788, swd: 1.4024, ept: 243.0670
    Epoch [17/50], Val Losses: mse: 40.9317, mae: 3.6540, huber: 3.2216, swd: 2.6108, ept: 192.6284
    Epoch [17/50], Test Losses: mse: 30.4398, mae: 3.1135, huber: 2.6850, swd: 2.1461, ept: 203.1729
      Epoch 17 composite train-obj: 1.778811
            No improvement (3.2216), counter 1/5
    Epoch [18/50], Train Losses: mse: 15.9516, mae: 2.1177, huber: 1.7155, swd: 1.3124, ept: 246.0395
    Epoch [18/50], Val Losses: mse: 37.2636, mae: 3.3847, huber: 2.9587, swd: 2.0489, ept: 206.5917
    Epoch [18/50], Test Losses: mse: 26.7015, mae: 2.8543, huber: 2.4334, swd: 1.8549, ept: 215.4473
      Epoch 18 composite train-obj: 1.715483
            Val objective improved 3.0470 → 2.9587, saving checkpoint.
    Epoch [19/50], Train Losses: mse: 14.8130, mae: 2.0183, huber: 1.6201, swd: 1.1927, ept: 251.2047
    Epoch [19/50], Val Losses: mse: 36.7242, mae: 3.4003, huber: 2.9757, swd: 2.2598, ept: 200.5083
    Epoch [19/50], Test Losses: mse: 25.1315, mae: 2.7632, huber: 2.3454, swd: 2.0581, ept: 218.3386
      Epoch 19 composite train-obj: 1.620142
            No improvement (2.9757), counter 1/5
    Epoch [20/50], Train Losses: mse: 14.4851, mae: 1.9792, huber: 1.5838, swd: 1.1599, ept: 253.4993
    Epoch [20/50], Val Losses: mse: 38.1077, mae: 3.4803, huber: 3.0537, swd: 2.2319, ept: 197.3901
    Epoch [20/50], Test Losses: mse: 30.1526, mae: 3.0408, huber: 2.6184, swd: 1.9714, ept: 206.9240
      Epoch 20 composite train-obj: 1.583842
            No improvement (3.0537), counter 2/5
    Epoch [21/50], Train Losses: mse: 14.4033, mae: 1.9687, huber: 1.5732, swd: 1.1370, ept: 255.6534
    Epoch [21/50], Val Losses: mse: 37.6629, mae: 3.4007, huber: 2.9739, swd: 2.2241, ept: 204.5387
    Epoch [21/50], Test Losses: mse: 27.1158, mae: 2.8456, huber: 2.4244, swd: 2.2249, ept: 219.1827
      Epoch 21 composite train-obj: 1.573222
            No improvement (2.9739), counter 3/5
    Epoch [22/50], Train Losses: mse: 13.6199, mae: 1.9014, huber: 1.5091, swd: 1.0654, ept: 258.3300
    Epoch [22/50], Val Losses: mse: 33.8628, mae: 3.2152, huber: 2.7920, swd: 2.1237, ept: 211.6136
    Epoch [22/50], Test Losses: mse: 24.3459, mae: 2.6919, huber: 2.2737, swd: 1.7617, ept: 225.6254
      Epoch 22 composite train-obj: 1.509076
            Val objective improved 2.9587 → 2.7920, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 13.4472, mae: 1.8728, huber: 1.4829, swd: 1.0316, ept: 260.9381
    Epoch [23/50], Val Losses: mse: 33.7777, mae: 3.2573, huber: 2.8297, swd: 1.9826, ept: 206.9234
    Epoch [23/50], Test Losses: mse: 26.8801, mae: 2.8114, huber: 2.3919, swd: 1.7449, ept: 222.0231
      Epoch 23 composite train-obj: 1.482911
            No improvement (2.8297), counter 1/5
    Epoch [24/50], Train Losses: mse: 12.7699, mae: 1.8162, huber: 1.4289, swd: 0.9912, ept: 263.1734
    Epoch [24/50], Val Losses: mse: 34.8037, mae: 3.2534, huber: 2.8317, swd: 1.8292, ept: 204.6360
    Epoch [24/50], Test Losses: mse: 27.0385, mae: 2.8260, huber: 2.4100, swd: 1.7716, ept: 216.8425
      Epoch 24 composite train-obj: 1.428874
            No improvement (2.8317), counter 2/5
    Epoch [25/50], Train Losses: mse: 12.5564, mae: 1.7872, huber: 1.4017, swd: 0.9587, ept: 265.8313
    Epoch [25/50], Val Losses: mse: 35.7973, mae: 3.2562, huber: 2.8345, swd: 1.9037, ept: 213.9686
    Epoch [25/50], Test Losses: mse: 28.5611, mae: 2.8510, huber: 2.4335, swd: 1.7132, ept: 226.1360
      Epoch 25 composite train-obj: 1.401702
            No improvement (2.8345), counter 3/5
    Epoch [26/50], Train Losses: mse: 13.0007, mae: 1.8192, huber: 1.4321, swd: 0.9873, ept: 265.1293
    Epoch [26/50], Val Losses: mse: 35.3184, mae: 3.2148, huber: 2.7979, swd: 2.1499, ept: 215.6577
    Epoch [26/50], Test Losses: mse: 27.7428, mae: 2.7700, huber: 2.3591, swd: 1.8402, ept: 230.5626
      Epoch 26 composite train-obj: 1.432131
            No improvement (2.7979), counter 4/5
    Epoch [27/50], Train Losses: mse: 12.2564, mae: 1.7511, huber: 1.3682, swd: 0.9257, ept: 268.1049
    Epoch [27/50], Val Losses: mse: 37.1094, mae: 3.2694, huber: 2.8522, swd: 1.8111, ept: 219.1601
    Epoch [27/50], Test Losses: mse: 27.8162, mae: 2.7495, huber: 2.3394, swd: 1.5743, ept: 231.8547
      Epoch 27 composite train-obj: 1.368193
    Epoch [27/50], Test Losses: mse: 24.3384, mae: 2.6915, huber: 2.2733, swd: 1.7614, ept: 225.6346
    Best round's Test MSE: 24.3459, MAE: 2.6919, SWD: 1.7617
    Best round's Validation MSE: 33.8628, MAE: 3.2152, SWD: 2.1237
    Best round's Test verification MSE : 24.3384, MAE: 2.6915, SWD: 1.7614
    Time taken: 99.96 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 76.4107, mae: 6.5660, huber: 6.0862, swd: 44.7102, ept: 33.5850
    Epoch [1/50], Val Losses: mse: 58.8472, mae: 5.7287, huber: 5.2538, swd: 25.0907, ept: 41.0895
    Epoch [1/50], Test Losses: mse: 54.6767, mae: 5.3944, huber: 4.9225, swd: 26.9514, ept: 39.8708
      Epoch 1 composite train-obj: 6.086175
            Val objective improved inf → 5.2538, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.8258, mae: 4.8116, huber: 4.3469, swd: 16.8209, ept: 90.5868
    Epoch [2/50], Val Losses: mse: 50.9729, mae: 4.9783, huber: 4.5138, swd: 14.0333, ept: 90.1403
    Epoch [2/50], Test Losses: mse: 46.5211, mae: 4.6519, huber: 4.1892, swd: 15.2522, ept: 85.4542
      Epoch 2 composite train-obj: 4.346906
            Val objective improved 5.2538 → 4.5138, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.8847, mae: 4.1021, huber: 3.6491, swd: 9.4573, ept: 135.7343
    Epoch [3/50], Val Losses: mse: 47.4588, mae: 4.5788, huber: 4.1208, swd: 8.5712, ept: 112.1907
    Epoch [3/50], Test Losses: mse: 41.6585, mae: 4.1994, huber: 3.7438, swd: 8.9895, ept: 114.3381
      Epoch 3 composite train-obj: 3.649143
            Val objective improved 4.5138 → 4.1208, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.1033, mae: 3.6980, huber: 3.2525, swd: 6.4034, ept: 162.3321
    Epoch [4/50], Val Losses: mse: 43.6630, mae: 4.2811, huber: 3.8286, swd: 7.3916, ept: 129.6077
    Epoch [4/50], Test Losses: mse: 37.7292, mae: 3.8839, huber: 3.4344, swd: 7.1662, ept: 133.2315
      Epoch 4 composite train-obj: 3.252496
            Val objective improved 4.1208 → 3.8286, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.0182, mae: 3.4227, huber: 2.9840, swd: 5.0605, ept: 177.8255
    Epoch [5/50], Val Losses: mse: 41.9918, mae: 4.1442, huber: 3.6959, swd: 7.2521, ept: 136.2666
    Epoch [5/50], Test Losses: mse: 34.7053, mae: 3.6691, huber: 3.2258, swd: 6.6188, ept: 146.4525
      Epoch 5 composite train-obj: 2.984021
            Val objective improved 3.8286 → 3.6959, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.3418, mae: 3.2002, huber: 2.7670, swd: 4.1994, ept: 187.1712
    Epoch [6/50], Val Losses: mse: 42.2985, mae: 4.0888, huber: 3.6418, swd: 5.2582, ept: 146.6109
    Epoch [6/50], Test Losses: mse: 32.0944, mae: 3.4799, huber: 3.0389, swd: 5.0251, ept: 156.5287
      Epoch 6 composite train-obj: 2.767004
            Val objective improved 3.6959 → 3.6418, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 27.2798, mae: 3.0355, huber: 2.6064, swd: 3.5793, ept: 194.3745
    Epoch [7/50], Val Losses: mse: 43.8572, mae: 4.1283, huber: 3.6809, swd: 4.8672, ept: 149.9348
    Epoch [7/50], Test Losses: mse: 32.7711, mae: 3.4696, huber: 3.0281, swd: 4.5185, ept: 156.8285
      Epoch 7 composite train-obj: 2.606368
            No improvement (3.6809), counter 1/5
    Epoch [8/50], Train Losses: mse: 25.8922, mae: 2.9193, huber: 2.4926, swd: 3.1597, ept: 200.0575
    Epoch [8/50], Val Losses: mse: 36.2215, mae: 3.6921, huber: 3.2493, swd: 4.5402, ept: 159.9346
    Epoch [8/50], Test Losses: mse: 29.7420, mae: 3.2653, huber: 2.8273, swd: 4.1664, ept: 174.8381
      Epoch 8 composite train-obj: 2.492551
            Val objective improved 3.6418 → 3.2493, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.3069, mae: 2.7183, huber: 2.2973, swd: 2.6638, ept: 208.1304
    Epoch [9/50], Val Losses: mse: 41.2846, mae: 3.8196, huber: 3.3794, swd: 3.6323, ept: 169.0437
    Epoch [9/50], Test Losses: mse: 29.1400, mae: 3.1533, huber: 2.7205, swd: 3.4702, ept: 183.3600
      Epoch 9 composite train-obj: 2.297308
            No improvement (3.3794), counter 1/5
    Epoch [10/50], Train Losses: mse: 22.4988, mae: 2.6585, huber: 2.2388, swd: 2.4320, ept: 211.2635
    Epoch [10/50], Val Losses: mse: 41.4358, mae: 3.8816, huber: 3.4404, swd: 4.2196, ept: 164.3649
    Epoch [10/50], Test Losses: mse: 30.3645, mae: 3.2240, huber: 2.7902, swd: 3.7550, ept: 181.7045
      Epoch 10 composite train-obj: 2.238769
            No improvement (3.4404), counter 2/5
    Epoch [11/50], Train Losses: mse: 20.9317, mae: 2.5329, huber: 2.1169, swd: 2.2267, ept: 217.7665
    Epoch [11/50], Val Losses: mse: 35.3624, mae: 3.5044, huber: 3.0689, swd: 3.7790, ept: 174.6323
    Epoch [11/50], Test Losses: mse: 27.4056, mae: 3.0365, huber: 2.6069, swd: 3.1357, ept: 188.5058
      Epoch 11 composite train-obj: 2.116895
            Val objective improved 3.2493 → 3.0689, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 19.8311, mae: 2.4436, huber: 2.0304, swd: 2.0062, ept: 221.8766
    Epoch [12/50], Val Losses: mse: 39.2709, mae: 3.6826, huber: 3.2464, swd: 3.4564, ept: 175.8074
    Epoch [12/50], Test Losses: mse: 30.2335, mae: 3.1238, huber: 2.6943, swd: 2.7616, ept: 194.6825
      Epoch 12 composite train-obj: 2.030420
            No improvement (3.2464), counter 1/5
    Epoch [13/50], Train Losses: mse: 18.5454, mae: 2.3336, huber: 1.9243, swd: 1.8130, ept: 227.1811
    Epoch [13/50], Val Losses: mse: 36.7587, mae: 3.4728, huber: 3.0427, swd: 3.0224, ept: 181.9916
    Epoch [13/50], Test Losses: mse: 25.7483, mae: 2.8557, huber: 2.4327, swd: 2.6852, ept: 200.4475
      Epoch 13 composite train-obj: 1.924281
            Val objective improved 3.0689 → 3.0427, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 17.7626, mae: 2.2615, huber: 1.8548, swd: 1.6787, ept: 231.6331
    Epoch [14/50], Val Losses: mse: 34.8098, mae: 3.3957, huber: 2.9654, swd: 3.4026, ept: 185.5223
    Epoch [14/50], Test Losses: mse: 27.3844, mae: 2.9303, huber: 2.5055, swd: 2.7185, ept: 201.0538
      Epoch 14 composite train-obj: 1.854827
            Val objective improved 3.0427 → 2.9654, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 16.7704, mae: 2.1832, huber: 1.7790, swd: 1.5470, ept: 234.4647
    Epoch [15/50], Val Losses: mse: 37.5179, mae: 3.5546, huber: 3.1217, swd: 3.1939, ept: 181.6371
    Epoch [15/50], Test Losses: mse: 26.8982, mae: 2.9163, huber: 2.4903, swd: 2.2658, ept: 202.5324
      Epoch 15 composite train-obj: 1.779014
            No improvement (3.1217), counter 1/5
    Epoch [16/50], Train Losses: mse: 16.2571, mae: 2.1326, huber: 1.7308, swd: 1.4843, ept: 238.3470
    Epoch [16/50], Val Losses: mse: 35.7565, mae: 3.3887, huber: 2.9606, swd: 2.9937, ept: 189.6885
    Epoch [16/50], Test Losses: mse: 24.9730, mae: 2.8059, huber: 2.3852, swd: 2.3021, ept: 200.7168
      Epoch 16 composite train-obj: 1.730847
            Val objective improved 2.9654 → 2.9606, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.3908, mae: 2.0573, huber: 1.6584, swd: 1.3980, ept: 242.7830
    Epoch [17/50], Val Losses: mse: 34.3357, mae: 3.2724, huber: 2.8467, swd: 2.9542, ept: 196.6471
    Epoch [17/50], Test Losses: mse: 26.6568, mae: 2.8177, huber: 2.3978, swd: 2.2685, ept: 214.5770
      Epoch 17 composite train-obj: 1.658432
            Val objective improved 2.9606 → 2.8467, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.0481, mae: 2.0200, huber: 1.6226, swd: 1.3323, ept: 245.9941
    Epoch [18/50], Val Losses: mse: 38.9831, mae: 3.4973, huber: 3.0717, swd: 2.6642, ept: 193.0945
    Epoch [18/50], Test Losses: mse: 29.6094, mae: 2.9228, huber: 2.5041, swd: 1.9448, ept: 217.9520
      Epoch 18 composite train-obj: 1.622625
            No improvement (3.0717), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.6667, mae: 1.9785, huber: 1.5838, swd: 1.2826, ept: 248.0417
    Epoch [19/50], Val Losses: mse: 34.6131, mae: 3.2975, huber: 2.8707, swd: 2.7834, ept: 191.9284
    Epoch [19/50], Test Losses: mse: 24.3655, mae: 2.7094, huber: 2.2914, swd: 2.1334, ept: 211.0659
      Epoch 19 composite train-obj: 1.583808
            No improvement (2.8707), counter 2/5
    Epoch [20/50], Train Losses: mse: 13.7497, mae: 1.8995, huber: 1.5085, swd: 1.1899, ept: 251.8001
    Epoch [20/50], Val Losses: mse: 37.6874, mae: 3.4331, huber: 3.0094, swd: 2.8405, ept: 193.2721
    Epoch [20/50], Test Losses: mse: 28.4205, mae: 2.8908, huber: 2.4721, swd: 2.3093, ept: 212.0157
      Epoch 20 composite train-obj: 1.508543
            No improvement (3.0094), counter 3/5
    Epoch [21/50], Train Losses: mse: 13.4588, mae: 1.8757, huber: 1.4849, swd: 1.1554, ept: 253.8325
    Epoch [21/50], Val Losses: mse: 36.2991, mae: 3.3593, huber: 2.9344, swd: 2.7386, ept: 197.2749
    Epoch [21/50], Test Losses: mse: 24.0515, mae: 2.6402, huber: 2.2264, swd: 1.8118, ept: 220.3067
      Epoch 21 composite train-obj: 1.484875
            No improvement (2.9344), counter 4/5
    Epoch [22/50], Train Losses: mse: 13.1864, mae: 1.8423, huber: 1.4537, swd: 1.1215, ept: 255.6445
    Epoch [22/50], Val Losses: mse: 33.3586, mae: 3.1540, huber: 2.7347, swd: 2.3883, ept: 207.2582
    Epoch [22/50], Test Losses: mse: 26.2872, mae: 2.7071, huber: 2.2953, swd: 1.9599, ept: 224.3396
      Epoch 22 composite train-obj: 1.453652
            Val objective improved 2.8467 → 2.7347, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 12.3436, mae: 1.7507, huber: 1.3686, swd: 1.0345, ept: 260.0761
    Epoch [23/50], Val Losses: mse: 34.3856, mae: 3.2260, huber: 2.8056, swd: 2.5428, ept: 201.6005
    Epoch [23/50], Test Losses: mse: 26.0170, mae: 2.6840, huber: 2.2722, swd: 1.9498, ept: 225.8656
      Epoch 23 composite train-obj: 1.368637
            No improvement (2.8056), counter 1/5
    Epoch [24/50], Train Losses: mse: 12.2535, mae: 1.7582, huber: 1.3740, swd: 1.0179, ept: 259.8774
    Epoch [24/50], Val Losses: mse: 35.3725, mae: 3.2496, huber: 2.8297, swd: 2.4606, ept: 203.6423
    Epoch [24/50], Test Losses: mse: 27.0452, mae: 2.7589, huber: 2.3448, swd: 1.9344, ept: 223.1744
      Epoch 24 composite train-obj: 1.374028
            No improvement (2.8297), counter 2/5
    Epoch [25/50], Train Losses: mse: 13.0011, mae: 1.8091, huber: 1.4229, swd: 1.0765, ept: 259.4212
    Epoch [25/50], Val Losses: mse: 33.5732, mae: 3.1345, huber: 2.7163, swd: 2.7050, ept: 204.6739
    Epoch [25/50], Test Losses: mse: 25.5805, mae: 2.6736, huber: 2.2614, swd: 2.0452, ept: 224.6333
      Epoch 25 composite train-obj: 1.422897
            Val objective improved 2.7347 → 2.7163, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 11.9056, mae: 1.7212, huber: 1.3388, swd: 0.9875, ept: 262.6317
    Epoch [26/50], Val Losses: mse: 36.4795, mae: 3.2681, huber: 2.8480, swd: 2.2769, ept: 207.3183
    Epoch [26/50], Test Losses: mse: 25.9871, mae: 2.6834, huber: 2.2713, swd: 1.6741, ept: 222.2644
      Epoch 26 composite train-obj: 1.338762
            No improvement (2.8480), counter 1/5
    Epoch [27/50], Train Losses: mse: 11.4655, mae: 1.6739, huber: 1.2951, swd: 0.9255, ept: 264.5311
    Epoch [27/50], Val Losses: mse: 35.0943, mae: 3.2940, huber: 2.8699, swd: 2.2317, ept: 202.2732
    Epoch [27/50], Test Losses: mse: 25.6345, mae: 2.6759, huber: 2.2639, swd: 1.7793, ept: 225.3176
      Epoch 27 composite train-obj: 1.295149
            No improvement (2.8699), counter 2/5
    Epoch [28/50], Train Losses: mse: 11.1664, mae: 1.6465, huber: 1.2686, swd: 0.9109, ept: 266.7425
    Epoch [28/50], Val Losses: mse: 33.3123, mae: 3.0902, huber: 2.6746, swd: 2.1321, ept: 209.7000
    Epoch [28/50], Test Losses: mse: 27.6232, mae: 2.7205, huber: 2.3107, swd: 1.5731, ept: 228.4477
      Epoch 28 composite train-obj: 1.268593
            Val objective improved 2.7163 → 2.6746, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 10.9762, mae: 1.6155, huber: 1.2403, swd: 0.8746, ept: 269.3421
    Epoch [29/50], Val Losses: mse: 35.8962, mae: 3.2957, huber: 2.8766, swd: 2.6709, ept: 206.0935
    Epoch [29/50], Test Losses: mse: 23.5462, mae: 2.5418, huber: 2.1338, swd: 1.7855, ept: 234.8177
      Epoch 29 composite train-obj: 1.240333
            No improvement (2.8766), counter 1/5
    Epoch [30/50], Train Losses: mse: 10.7277, mae: 1.5968, huber: 1.2227, swd: 0.8586, ept: 270.3344
    Epoch [30/50], Val Losses: mse: 30.7208, mae: 3.0081, huber: 2.5927, swd: 2.2873, ept: 214.3659
    Epoch [30/50], Test Losses: mse: 23.9950, mae: 2.5694, huber: 2.1606, swd: 1.6548, ept: 234.9000
      Epoch 30 composite train-obj: 1.222748
            Val objective improved 2.6746 → 2.5927, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 10.4996, mae: 1.5665, huber: 1.1950, swd: 0.8402, ept: 271.7299
    Epoch [31/50], Val Losses: mse: 33.6068, mae: 3.0935, huber: 2.6782, swd: 2.0770, ept: 216.8372
    Epoch [31/50], Test Losses: mse: 23.4897, mae: 2.4969, huber: 2.0927, swd: 1.5077, ept: 238.4016
      Epoch 31 composite train-obj: 1.194976
            No improvement (2.6782), counter 1/5
    Epoch [32/50], Train Losses: mse: 10.8016, mae: 1.5903, huber: 1.2173, swd: 0.8341, ept: 271.2672
    Epoch [32/50], Val Losses: mse: 39.0555, mae: 3.3729, huber: 2.9554, swd: 2.2197, ept: 213.9991
    Epoch [32/50], Test Losses: mse: 24.9974, mae: 2.6300, huber: 2.2198, swd: 1.6004, ept: 232.4799
      Epoch 32 composite train-obj: 1.217287
            No improvement (2.9554), counter 2/5
    Epoch [33/50], Train Losses: mse: 11.5426, mae: 1.6548, huber: 1.2777, swd: 0.9116, ept: 269.6426
    Epoch [33/50], Val Losses: mse: 33.9618, mae: 3.1128, huber: 2.6976, swd: 2.2562, ept: 214.0933
    Epoch [33/50], Test Losses: mse: 26.7125, mae: 2.6592, huber: 2.2511, swd: 1.6912, ept: 234.2132
      Epoch 33 composite train-obj: 1.277695
            No improvement (2.6976), counter 3/5
    Epoch [34/50], Train Losses: mse: 10.1528, mae: 1.5297, huber: 1.1603, swd: 0.8036, ept: 274.9660
    Epoch [34/50], Val Losses: mse: 31.5222, mae: 3.0363, huber: 2.6214, swd: 2.1420, ept: 208.0054
    Epoch [34/50], Test Losses: mse: 28.4419, mae: 2.7385, huber: 2.3313, swd: 1.6408, ept: 229.4226
      Epoch 34 composite train-obj: 1.160254
            No improvement (2.6214), counter 4/5
    Epoch [35/50], Train Losses: mse: 10.0840, mae: 1.5096, huber: 1.1426, swd: 0.7840, ept: 276.6945
    Epoch [35/50], Val Losses: mse: 33.5016, mae: 3.1180, huber: 2.7001, swd: 2.1571, ept: 211.9666
    Epoch [35/50], Test Losses: mse: 23.7680, mae: 2.5384, huber: 2.1320, swd: 1.5685, ept: 234.3125
      Epoch 35 composite train-obj: 1.142609
    Epoch [35/50], Test Losses: mse: 23.9832, mae: 2.5689, huber: 2.1601, swd: 1.6544, ept: 234.9093
    Best round's Test MSE: 23.9950, MAE: 2.5694, SWD: 1.6548
    Best round's Validation MSE: 30.7208, MAE: 3.0081, SWD: 2.2873
    Best round's Test verification MSE : 23.9832, MAE: 2.5689, SWD: 1.6544
    Time taken: 135.85 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1833)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 23.2591 ± 1.2968
      mae: 2.5539 ± 0.1196
      huber: 2.1443 ± 0.1129
      swd: 1.5848 ± 0.1800
      ept: 235.5685 ± 8.4046
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 31.5332 ± 1.6722
      mae: 3.0386 ± 0.1335
      huber: 2.6221 ± 0.1285
      swd: 2.0188 ± 0.2723
      ept: 216.4143 ± 4.9717
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 355.41 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1833
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    

### AB: Cascaded model


```python

```

## AB: Other Components

### AB: two encoders and convex combined
Convex combination was really helpful when there is unsmooth optimizations. 
Offering no additional benefits seems to indicate our model is functioning.


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
    
    Epoch [1/50], Train Losses: mse: 74.2999, mae: 6.4853, huber: 6.0056, swd: 42.3418, ept: 32.1924
    Epoch [1/50], Val Losses: mse: 59.6500, mae: 5.8156, huber: 5.3402, swd: 25.8090, ept: 35.5563
    Epoch [1/50], Test Losses: mse: 55.3164, mae: 5.4569, huber: 4.9840, swd: 27.5737, ept: 35.7190
      Epoch 1 composite train-obj: 6.005582
            Val objective improved inf → 5.3402, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 48.0237, mae: 4.8139, huber: 4.3492, swd: 15.7586, ept: 92.7387
    Epoch [2/50], Val Losses: mse: 51.5800, mae: 5.0744, huber: 4.6071, swd: 12.8090, ept: 84.3388
    Epoch [2/50], Test Losses: mse: 46.8593, mae: 4.7259, huber: 4.2620, swd: 14.5418, ept: 78.4288
      Epoch 2 composite train-obj: 4.349239
            Val objective improved 5.3402 → 4.6071, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.7719, mae: 4.1018, huber: 3.6478, swd: 8.3894, ept: 137.1935
    Epoch [3/50], Val Losses: mse: 44.9534, mae: 4.5072, huber: 4.0468, swd: 7.8069, ept: 115.9292
    Epoch [3/50], Test Losses: mse: 41.4533, mae: 4.2485, huber: 3.7891, swd: 7.7509, ept: 126.1689
      Epoch 3 composite train-obj: 3.647818
            Val objective improved 4.6071 → 4.0468, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.7851, mae: 3.6610, huber: 3.2158, swd: 5.2449, ept: 170.6878
    Epoch [4/50], Val Losses: mse: 45.8948, mae: 4.4496, huber: 3.9926, swd: 7.6730, ept: 135.5339
    Epoch [4/50], Test Losses: mse: 40.0302, mae: 4.0686, huber: 3.6145, swd: 7.3485, ept: 143.5485
      Epoch 4 composite train-obj: 3.215833
            Val objective improved 4.0468 → 3.9926, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 31.8344, mae: 3.4009, huber: 2.9618, swd: 4.2508, ept: 185.0658
    Epoch [5/50], Val Losses: mse: 41.3718, mae: 4.0695, huber: 3.6205, swd: 5.1863, ept: 146.2592
    Epoch [5/50], Test Losses: mse: 38.0281, mae: 3.8637, huber: 3.4150, swd: 5.6206, ept: 151.4064
      Epoch 5 composite train-obj: 2.961785
            Val objective improved 3.9926 → 3.6205, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.7530, mae: 3.2290, huber: 2.7941, swd: 3.6557, ept: 194.2198
    Epoch [6/50], Val Losses: mse: 42.8691, mae: 4.1091, huber: 3.6606, swd: 5.2964, ept: 154.6834
    Epoch [6/50], Test Losses: mse: 35.5305, mae: 3.6544, huber: 3.2099, swd: 4.4891, ept: 162.3177
      Epoch 6 composite train-obj: 2.794071
            No improvement (3.6606), counter 1/5
    Epoch [7/50], Train Losses: mse: 27.4100, mae: 3.0317, huber: 2.6021, swd: 3.0867, ept: 203.2958
    Epoch [7/50], Val Losses: mse: 40.9653, mae: 3.9497, huber: 3.5022, swd: 3.7979, ept: 162.8099
    Epoch [7/50], Test Losses: mse: 33.8614, mae: 3.5347, huber: 3.0909, swd: 3.2962, ept: 172.3799
      Epoch 7 composite train-obj: 2.602127
            Val objective improved 3.6205 → 3.5022, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.4384, mae: 2.8768, huber: 2.4514, swd: 2.6017, ept: 210.4264
    Epoch [8/50], Val Losses: mse: 40.7993, mae: 3.9015, huber: 3.4567, swd: 3.5684, ept: 159.1322
    Epoch [8/50], Test Losses: mse: 33.4327, mae: 3.4534, huber: 3.0139, swd: 3.5234, ept: 175.9755
      Epoch 8 composite train-obj: 2.451399
            Val objective improved 3.5022 → 3.4567, saving checkpoint.
    Epoch [9/50], Train Losses: mse: 23.9864, mae: 2.7702, huber: 2.3470, swd: 2.3312, ept: 215.7592
    Epoch [9/50], Val Losses: mse: 40.6772, mae: 3.9164, huber: 3.4698, swd: 3.9882, ept: 165.2929
    Epoch [9/50], Test Losses: mse: 32.8171, mae: 3.4299, huber: 2.9908, swd: 3.1804, ept: 180.0536
      Epoch 9 composite train-obj: 2.347046
            No improvement (3.4698), counter 1/5
    Epoch [10/50], Train Losses: mse: 23.1877, mae: 2.7180, huber: 2.2953, swd: 2.2238, ept: 218.3230
    Epoch [10/50], Val Losses: mse: 39.9203, mae: 3.7717, huber: 3.3311, swd: 3.1894, ept: 177.0116
    Epoch [10/50], Test Losses: mse: 32.3031, mae: 3.3592, huber: 2.9223, swd: 2.8496, ept: 183.5716
      Epoch 10 composite train-obj: 2.295282
            Val objective improved 3.4567 → 3.3311, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 21.3838, mae: 2.5710, huber: 2.1526, swd: 1.9279, ept: 225.1077
    Epoch [11/50], Val Losses: mse: 41.5912, mae: 3.7779, huber: 3.3374, swd: 2.8588, ept: 182.3201
    Epoch [11/50], Test Losses: mse: 32.7875, mae: 3.3046, huber: 2.8683, swd: 2.6294, ept: 190.8125
      Epoch 11 composite train-obj: 2.152616
            No improvement (3.3374), counter 1/5
    Epoch [12/50], Train Losses: mse: 19.8656, mae: 2.4379, huber: 2.0246, swd: 1.7076, ept: 231.3494
    Epoch [12/50], Val Losses: mse: 39.4748, mae: 3.6466, huber: 3.2103, swd: 2.9409, ept: 183.3701
    Epoch [12/50], Test Losses: mse: 30.1842, mae: 3.1154, huber: 2.6859, swd: 2.3196, ept: 201.0953
      Epoch 12 composite train-obj: 2.024600
            Val objective improved 3.3311 → 3.2103, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.1233, mae: 2.2864, huber: 1.8785, swd: 1.5011, ept: 238.8794
    Epoch [13/50], Val Losses: mse: 39.1580, mae: 3.5813, huber: 3.1470, swd: 2.4039, ept: 188.7346
    Epoch [13/50], Test Losses: mse: 31.8298, mae: 3.1718, huber: 2.7425, swd: 2.3251, ept: 205.6352
      Epoch 13 composite train-obj: 1.878512
            Val objective improved 3.2103 → 3.1470, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.0016, mae: 2.2631, huber: 1.8564, swd: 1.4736, ept: 241.8959
    Epoch [14/50], Val Losses: mse: 41.3191, mae: 3.7152, huber: 3.2754, swd: 3.1436, ept: 183.1804
    Epoch [14/50], Test Losses: mse: 30.4585, mae: 3.1100, huber: 2.6784, swd: 2.6742, ept: 203.5223
      Epoch 14 composite train-obj: 1.856432
            No improvement (3.2754), counter 1/5
    Epoch [15/50], Train Losses: mse: 16.7127, mae: 2.1710, huber: 1.7668, swd: 1.3459, ept: 245.5692
    Epoch [15/50], Val Losses: mse: 44.0991, mae: 3.7698, huber: 3.3373, swd: 2.6995, ept: 187.3360
    Epoch [15/50], Test Losses: mse: 30.4498, mae: 3.0527, huber: 2.6279, swd: 2.0385, ept: 206.2305
      Epoch 15 composite train-obj: 1.766760
            No improvement (3.3373), counter 2/5
    Epoch [16/50], Train Losses: mse: 16.0660, mae: 2.1245, huber: 1.7217, swd: 1.2436, ept: 247.3905
    Epoch [16/50], Val Losses: mse: 40.5984, mae: 3.6420, huber: 3.2067, swd: 2.5718, ept: 185.0206
    Epoch [16/50], Test Losses: mse: 30.3003, mae: 3.0162, huber: 2.5912, swd: 2.0557, ept: 212.8144
      Epoch 16 composite train-obj: 1.721727
            No improvement (3.2067), counter 3/5
    Epoch [17/50], Train Losses: mse: 15.7655, mae: 2.0690, huber: 1.6695, swd: 1.2164, ept: 251.9578
    Epoch [17/50], Val Losses: mse: 39.1555, mae: 3.5092, huber: 3.0797, swd: 2.2846, ept: 193.3901
    Epoch [17/50], Test Losses: mse: 29.6273, mae: 2.9843, huber: 2.5610, swd: 1.9196, ept: 213.4097
      Epoch 17 composite train-obj: 1.669536
            Val objective improved 3.1470 → 3.0797, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 14.9696, mae: 1.9994, huber: 1.6034, swd: 1.1452, ept: 256.4957
    Epoch [18/50], Val Losses: mse: 41.0670, mae: 3.6066, huber: 3.1756, swd: 2.3648, ept: 192.9176
    Epoch [18/50], Test Losses: mse: 31.1363, mae: 3.0026, huber: 2.5811, swd: 2.0502, ept: 221.6182
      Epoch 18 composite train-obj: 1.603402
            No improvement (3.1756), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.8120, mae: 1.9618, huber: 1.5697, swd: 1.1515, ept: 259.2595
    Epoch [19/50], Val Losses: mse: 45.2300, mae: 3.8551, huber: 3.4194, swd: 2.2682, ept: 192.1106
    Epoch [19/50], Test Losses: mse: 29.3407, mae: 2.9699, huber: 2.5447, swd: 1.8446, ept: 218.4743
      Epoch 19 composite train-obj: 1.569682
            No improvement (3.4194), counter 2/5
    Epoch [20/50], Train Losses: mse: 14.6044, mae: 1.9675, huber: 1.5721, swd: 1.0988, ept: 257.9297
    Epoch [20/50], Val Losses: mse: 43.6426, mae: 3.7150, huber: 3.2864, swd: 2.8775, ept: 194.4437
    Epoch [20/50], Test Losses: mse: 30.9542, mae: 2.9618, huber: 2.5429, swd: 1.8196, ept: 221.5563
      Epoch 20 composite train-obj: 1.572119
            No improvement (3.2864), counter 3/5
    Epoch [21/50], Train Losses: mse: 13.6214, mae: 1.8660, huber: 1.4770, swd: 1.0147, ept: 263.4024
    Epoch [21/50], Val Losses: mse: 38.2464, mae: 3.3838, huber: 2.9592, swd: 2.1780, ept: 201.1252
    Epoch [21/50], Test Losses: mse: 29.7296, mae: 2.8614, huber: 2.4473, swd: 1.8584, ept: 228.0649
      Epoch 21 composite train-obj: 1.477037
            Val objective improved 3.0797 → 2.9592, saving checkpoint.
    Epoch [22/50], Train Losses: mse: 13.1992, mae: 1.8282, huber: 1.4409, swd: 0.9746, ept: 265.4958
    Epoch [22/50], Val Losses: mse: 40.6795, mae: 3.4994, huber: 3.0726, swd: 2.2549, ept: 201.1929
    Epoch [22/50], Test Losses: mse: 28.7380, mae: 2.8346, huber: 2.4173, swd: 1.8141, ept: 230.0571
      Epoch 22 composite train-obj: 1.440940
            No improvement (3.0726), counter 1/5
    Epoch [23/50], Train Losses: mse: 13.4926, mae: 1.8649, huber: 1.4749, swd: 1.0024, ept: 263.7448
    Epoch [23/50], Val Losses: mse: 40.5009, mae: 3.5233, huber: 3.0956, swd: 2.2804, ept: 197.9875
    Epoch [23/50], Test Losses: mse: 30.3886, mae: 2.9213, huber: 2.5023, swd: 1.6708, ept: 219.9927
      Epoch 23 composite train-obj: 1.474907
            No improvement (3.0956), counter 2/5
    Epoch [24/50], Train Losses: mse: 11.8000, mae: 1.6998, huber: 1.3197, swd: 0.8592, ept: 271.5458
    Epoch [24/50], Val Losses: mse: 44.0161, mae: 3.6587, huber: 3.2348, swd: 2.1117, ept: 200.3240
    Epoch [24/50], Test Losses: mse: 26.5471, mae: 2.6993, huber: 2.2891, swd: 1.5879, ept: 231.4143
      Epoch 24 composite train-obj: 1.319686
            No improvement (3.2348), counter 3/5
    Epoch [25/50], Train Losses: mse: 12.3600, mae: 1.7400, huber: 1.3581, swd: 0.8794, ept: 270.1408
    Epoch [25/50], Val Losses: mse: 37.7660, mae: 3.3587, huber: 2.9351, swd: 1.8784, ept: 204.9855
    Epoch [25/50], Test Losses: mse: 27.1777, mae: 2.7561, huber: 2.3403, swd: 1.5892, ept: 232.1157
      Epoch 25 composite train-obj: 1.358058
            Val objective improved 2.9592 → 2.9351, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.1577, mae: 1.7183, huber: 1.3383, swd: 0.8713, ept: 271.8060
    Epoch [26/50], Val Losses: mse: 39.7879, mae: 3.4242, huber: 3.0011, swd: 1.8204, ept: 202.3331
    Epoch [26/50], Test Losses: mse: 29.8404, mae: 2.8533, huber: 2.4394, swd: 1.6632, ept: 227.2223
      Epoch 26 composite train-obj: 1.338252
            No improvement (3.0011), counter 1/5
    Epoch [27/50], Train Losses: mse: 12.1761, mae: 1.7327, huber: 1.3499, swd: 0.8770, ept: 270.9283
    Epoch [27/50], Val Losses: mse: 38.6832, mae: 3.4013, huber: 2.9721, swd: 2.0474, ept: 209.6752
    Epoch [27/50], Test Losses: mse: 29.9112, mae: 2.8880, huber: 2.4675, swd: 1.6670, ept: 229.2332
      Epoch 27 composite train-obj: 1.349886
            No improvement (2.9721), counter 2/5
    Epoch [28/50], Train Losses: mse: 11.3142, mae: 1.6515, huber: 1.2739, swd: 0.8067, ept: 274.7094
    Epoch [28/50], Val Losses: mse: 38.6017, mae: 3.3330, huber: 2.9133, swd: 1.8413, ept: 215.7949
    Epoch [28/50], Test Losses: mse: 25.2147, mae: 2.6004, huber: 2.1946, swd: 1.5082, ept: 235.9369
      Epoch 28 composite train-obj: 1.273907
            Val objective improved 2.9351 → 2.9133, saving checkpoint.
    Epoch [29/50], Train Losses: mse: 11.3039, mae: 1.6443, huber: 1.2680, swd: 0.7874, ept: 274.3906
    Epoch [29/50], Val Losses: mse: 37.4112, mae: 3.3160, huber: 2.8953, swd: 2.1217, ept: 208.8719
    Epoch [29/50], Test Losses: mse: 29.0897, mae: 2.8313, huber: 2.4180, swd: 1.6498, ept: 230.0049
      Epoch 29 composite train-obj: 1.268045
            Val objective improved 2.9133 → 2.8953, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 11.6334, mae: 1.6576, huber: 1.2807, swd: 0.8214, ept: 275.4602
    Epoch [30/50], Val Losses: mse: 37.6369, mae: 3.2844, huber: 2.8641, swd: 1.8456, ept: 208.6348
    Epoch [30/50], Test Losses: mse: 26.6568, mae: 2.6714, huber: 2.2624, swd: 1.5183, ept: 235.0424
      Epoch 30 composite train-obj: 1.280717
            Val objective improved 2.8953 → 2.8641, saving checkpoint.
    Epoch [31/50], Train Losses: mse: 11.9523, mae: 1.6811, huber: 1.3037, swd: 0.8448, ept: 274.8995
    Epoch [31/50], Val Losses: mse: 37.6156, mae: 3.2378, huber: 2.8214, swd: 1.7314, ept: 214.7194
    Epoch [31/50], Test Losses: mse: 28.5309, mae: 2.7381, huber: 2.3312, swd: 1.5542, ept: 239.2126
      Epoch 31 composite train-obj: 1.303732
            Val objective improved 2.8641 → 2.8214, saving checkpoint.
    Epoch [32/50], Train Losses: mse: 10.4577, mae: 1.5466, huber: 1.1773, swd: 0.7370, ept: 280.3199
    Epoch [32/50], Val Losses: mse: 36.6409, mae: 3.1590, huber: 2.7452, swd: 1.5622, ept: 224.3697
    Epoch [32/50], Test Losses: mse: 26.0589, mae: 2.5919, huber: 2.1884, swd: 1.3174, ept: 241.6204
      Epoch 32 composite train-obj: 1.177322
            Val objective improved 2.8214 → 2.7452, saving checkpoint.
    Epoch [33/50], Train Losses: mse: 10.6834, mae: 1.5637, huber: 1.1933, swd: 0.7430, ept: 279.9997
    Epoch [33/50], Val Losses: mse: 39.6106, mae: 3.4059, huber: 2.9824, swd: 1.7961, ept: 210.4315
    Epoch [33/50], Test Losses: mse: 28.2938, mae: 2.7891, huber: 2.3756, swd: 1.5850, ept: 231.6500
      Epoch 33 composite train-obj: 1.193308
            No improvement (2.9824), counter 1/5
    Epoch [34/50], Train Losses: mse: 10.2695, mae: 1.5451, huber: 1.1746, swd: 0.7389, ept: 280.9103
    Epoch [34/50], Val Losses: mse: 37.7023, mae: 3.2404, huber: 2.8264, swd: 1.5629, ept: 215.0926
    Epoch [34/50], Test Losses: mse: 28.8277, mae: 2.7086, huber: 2.3055, swd: 1.5309, ept: 240.5351
      Epoch 34 composite train-obj: 1.174635
            No improvement (2.8264), counter 2/5
    Epoch [35/50], Train Losses: mse: 10.3618, mae: 1.5429, huber: 1.1726, swd: 0.7309, ept: 281.7560
    Epoch [35/50], Val Losses: mse: 36.6943, mae: 3.2242, huber: 2.8075, swd: 1.7458, ept: 216.9481
    Epoch [35/50], Test Losses: mse: 27.7550, mae: 2.6827, huber: 2.2768, swd: 1.5038, ept: 239.6299
      Epoch 35 composite train-obj: 1.172555
            No improvement (2.8075), counter 3/5
    Epoch [36/50], Train Losses: mse: 10.2998, mae: 1.5367, huber: 1.1676, swd: 0.7249, ept: 281.6136
    Epoch [36/50], Val Losses: mse: 37.7956, mae: 3.2760, huber: 2.8583, swd: 1.6490, ept: 211.3108
    Epoch [36/50], Test Losses: mse: 27.0929, mae: 2.6611, huber: 2.2562, swd: 1.4295, ept: 236.6207
      Epoch 36 composite train-obj: 1.167623
            No improvement (2.8583), counter 4/5
    Epoch [37/50], Train Losses: mse: 9.6145, mae: 1.4624, huber: 1.0993, swd: 0.6584, ept: 285.1400
    Epoch [37/50], Val Losses: mse: 36.4704, mae: 3.2027, huber: 2.7833, swd: 1.6157, ept: 216.1534
    Epoch [37/50], Test Losses: mse: 25.9255, mae: 2.6080, huber: 2.2009, swd: 1.4248, ept: 242.7168
      Epoch 37 composite train-obj: 1.099306
    Epoch [37/50], Test Losses: mse: 26.0574, mae: 2.5918, huber: 2.1882, swd: 1.3163, ept: 241.6537
    Best round's Test MSE: 26.0589, MAE: 2.5919, SWD: 1.3174
    Best round's Validation MSE: 36.6409, MAE: 3.1590, SWD: 1.5622
    Best round's Test verification MSE : 26.0574, MAE: 2.5918, SWD: 1.3163
    Time taken: 178.62 seconds
    
    ==================================================
     Running experiment with seed 7 (2/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 73.8511, mae: 6.4393, huber: 5.9604, swd: 40.8000, ept: 37.8817
    Epoch [1/50], Val Losses: mse: 58.6656, mae: 5.6985, huber: 5.2243, swd: 23.0527, ept: 55.7980
    Epoch [1/50], Test Losses: mse: 54.0757, mae: 5.3332, huber: 4.8621, swd: 25.2411, ept: 49.3805
      Epoch 1 composite train-obj: 5.960361
            Val objective improved inf → 5.2243, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 46.9887, mae: 4.7496, huber: 4.2858, swd: 14.5444, ept: 103.0870
    Epoch [2/50], Val Losses: mse: 51.6006, mae: 5.0159, huber: 4.5497, swd: 12.9240, ept: 97.7404
    Epoch [2/50], Test Losses: mse: 46.5765, mae: 4.6348, huber: 4.1718, swd: 12.3702, ept: 98.9539
      Epoch 2 composite train-obj: 4.285762
            Val objective improved 5.2243 → 4.5497, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 38.8879, mae: 4.0458, huber: 3.5934, swd: 7.5717, ept: 144.9748
    Epoch [3/50], Val Losses: mse: 50.2060, mae: 4.8592, huber: 4.3958, swd: 10.3344, ept: 112.3019
    Epoch [3/50], Test Losses: mse: 43.4248, mae: 4.3884, huber: 3.9287, swd: 9.0744, ept: 115.4311
      Epoch 3 composite train-obj: 3.593413
            Val objective improved 4.5497 → 4.3958, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 34.8195, mae: 3.6856, huber: 3.2402, swd: 5.5379, ept: 166.1161
    Epoch [4/50], Val Losses: mse: 46.3969, mae: 4.5183, huber: 4.0605, swd: 8.5251, ept: 126.2543
    Epoch [4/50], Test Losses: mse: 38.7856, mae: 4.0089, huber: 3.5561, swd: 7.3943, ept: 132.7051
      Epoch 4 composite train-obj: 3.240197
            Val objective improved 4.3958 → 4.0605, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.0317, mae: 3.4500, huber: 3.0096, swd: 4.4380, ept: 177.9253
    Epoch [5/50], Val Losses: mse: 47.3359, mae: 4.4876, huber: 4.0309, swd: 6.2210, ept: 133.5963
    Epoch [5/50], Test Losses: mse: 36.3060, mae: 3.8241, huber: 3.3731, swd: 5.6395, ept: 144.2994
      Epoch 5 composite train-obj: 3.009606
            Val objective improved 4.0605 → 4.0309, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 29.5342, mae: 3.2364, huber: 2.8012, swd: 3.6399, ept: 186.7050
    Epoch [6/50], Val Losses: mse: 45.1288, mae: 4.2630, huber: 3.8129, swd: 6.1175, ept: 143.4012
    Epoch [6/50], Test Losses: mse: 35.9388, mae: 3.6902, huber: 3.2455, swd: 4.8989, ept: 156.6330
      Epoch 6 composite train-obj: 2.801184
            Val objective improved 4.0309 → 3.8129, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 26.9962, mae: 3.0299, huber: 2.5998, swd: 3.0174, ept: 195.8739
    Epoch [7/50], Val Losses: mse: 46.5526, mae: 4.2531, huber: 3.8031, swd: 5.2419, ept: 151.3409
    Epoch [7/50], Test Losses: mse: 34.6112, mae: 3.5867, huber: 3.1423, swd: 3.9520, ept: 158.5988
      Epoch 7 composite train-obj: 2.599848
            Val objective improved 3.8129 → 3.8031, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 25.0802, mae: 2.8810, huber: 2.4544, swd: 2.6129, ept: 202.4607
    Epoch [8/50], Val Losses: mse: 47.6792, mae: 4.3215, huber: 3.8719, swd: 5.2835, ept: 148.4357
    Epoch [8/50], Test Losses: mse: 34.3013, mae: 3.5381, huber: 3.0962, swd: 3.7232, ept: 165.8134
      Epoch 8 composite train-obj: 2.454351
            No improvement (3.8719), counter 1/5
    Epoch [9/50], Train Losses: mse: 23.4245, mae: 2.7509, huber: 2.3277, swd: 2.3426, ept: 208.2590
    Epoch [9/50], Val Losses: mse: 44.9829, mae: 4.1480, huber: 3.7004, swd: 5.0609, ept: 157.1626
    Epoch [9/50], Test Losses: mse: 33.1406, mae: 3.4217, huber: 2.9819, swd: 3.4670, ept: 174.3583
      Epoch 9 composite train-obj: 2.327719
            Val objective improved 3.8031 → 3.7004, saving checkpoint.
    Epoch [10/50], Train Losses: mse: 22.2330, mae: 2.6634, huber: 2.2416, swd: 2.1296, ept: 212.7730
    Epoch [10/50], Val Losses: mse: 46.5023, mae: 4.1952, huber: 3.7469, swd: 4.7419, ept: 153.1900
    Epoch [10/50], Test Losses: mse: 32.1729, mae: 3.3486, huber: 2.9090, swd: 3.0477, ept: 176.8752
      Epoch 10 composite train-obj: 2.241614
            No improvement (3.7469), counter 1/5
    Epoch [11/50], Train Losses: mse: 20.6072, mae: 2.5342, huber: 2.1160, swd: 1.9096, ept: 218.2329
    Epoch [11/50], Val Losses: mse: 47.9723, mae: 4.2019, huber: 3.7579, swd: 4.4862, ept: 162.3645
    Epoch [11/50], Test Losses: mse: 30.4494, mae: 3.2120, huber: 2.7776, swd: 2.9362, ept: 184.1173
      Epoch 11 composite train-obj: 2.116041
            No improvement (3.7579), counter 2/5
    Epoch [12/50], Train Losses: mse: 19.3351, mae: 2.4268, huber: 2.0124, swd: 1.7032, ept: 223.4721
    Epoch [12/50], Val Losses: mse: 46.3100, mae: 4.0995, huber: 3.6549, swd: 3.9659, ept: 161.4356
    Epoch [12/50], Test Losses: mse: 33.2343, mae: 3.3522, huber: 2.9158, swd: 2.6841, ept: 181.6925
      Epoch 12 composite train-obj: 2.012351
            Val objective improved 3.7004 → 3.6549, saving checkpoint.
    Epoch [13/50], Train Losses: mse: 18.3068, mae: 2.3368, huber: 1.9251, swd: 1.6115, ept: 229.7725
    Epoch [13/50], Val Losses: mse: 43.9501, mae: 3.9476, huber: 3.5061, swd: 3.5627, ept: 167.5971
    Epoch [13/50], Test Losses: mse: 33.4096, mae: 3.3190, huber: 2.8846, swd: 2.5385, ept: 188.6272
      Epoch 13 composite train-obj: 1.925137
            Val objective improved 3.6549 → 3.5061, saving checkpoint.
    Epoch [14/50], Train Losses: mse: 18.1703, mae: 2.3088, huber: 1.8996, swd: 1.5724, ept: 232.0436
    Epoch [14/50], Val Losses: mse: 43.2001, mae: 3.8644, huber: 3.4258, swd: 3.5227, ept: 167.3987
    Epoch [14/50], Test Losses: mse: 32.3160, mae: 3.2396, huber: 2.8092, swd: 2.4550, ept: 187.3608
      Epoch 14 composite train-obj: 1.899570
            Val objective improved 3.5061 → 3.4258, saving checkpoint.
    Epoch [15/50], Train Losses: mse: 17.1327, mae: 2.2200, huber: 1.8138, swd: 1.4826, ept: 236.6635
    Epoch [15/50], Val Losses: mse: 45.3674, mae: 3.9448, huber: 3.5093, swd: 3.5794, ept: 175.7909
    Epoch [15/50], Test Losses: mse: 33.8404, mae: 3.2675, huber: 2.8391, swd: 2.5090, ept: 191.5795
      Epoch 15 composite train-obj: 1.813847
            No improvement (3.5093), counter 1/5
    Epoch [16/50], Train Losses: mse: 16.5684, mae: 2.1885, huber: 1.7828, swd: 1.3875, ept: 238.1482
    Epoch [16/50], Val Losses: mse: 43.0242, mae: 3.8464, huber: 3.4102, swd: 3.6697, ept: 174.8136
    Epoch [16/50], Test Losses: mse: 30.7767, mae: 3.1055, huber: 2.6774, swd: 2.4376, ept: 199.7422
      Epoch 16 composite train-obj: 1.782811
            Val objective improved 3.4258 → 3.4102, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 15.6829, mae: 2.0927, huber: 1.6917, swd: 1.3027, ept: 243.6620
    Epoch [17/50], Val Losses: mse: 42.4671, mae: 3.7741, huber: 3.3392, swd: 3.2746, ept: 178.1717
    Epoch [17/50], Test Losses: mse: 31.5949, mae: 3.1060, huber: 2.6798, swd: 2.4251, ept: 202.6484
      Epoch 17 composite train-obj: 1.691711
            Val objective improved 3.4102 → 3.3392, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 15.0602, mae: 2.0489, huber: 1.6485, swd: 1.2498, ept: 245.7041
    Epoch [18/50], Val Losses: mse: 44.5030, mae: 3.8314, huber: 3.3981, swd: 2.9492, ept: 182.7647
    Epoch [18/50], Test Losses: mse: 28.4405, mae: 2.9217, huber: 2.4988, swd: 1.8701, ept: 208.8878
      Epoch 18 composite train-obj: 1.648523
            No improvement (3.3981), counter 1/5
    Epoch [19/50], Train Losses: mse: 14.2833, mae: 1.9626, huber: 1.5679, swd: 1.1627, ept: 250.8508
    Epoch [19/50], Val Losses: mse: 41.2480, mae: 3.6512, huber: 3.2190, swd: 3.1263, ept: 185.5174
    Epoch [19/50], Test Losses: mse: 28.0151, mae: 2.9365, huber: 2.5111, swd: 1.9513, ept: 208.0519
      Epoch 19 composite train-obj: 1.567936
            Val objective improved 3.3392 → 3.2190, saving checkpoint.
    Epoch [20/50], Train Losses: mse: 13.4120, mae: 1.8873, huber: 1.4958, swd: 1.0820, ept: 255.0586
    Epoch [20/50], Val Losses: mse: 41.7579, mae: 3.7444, huber: 3.3121, swd: 3.5439, ept: 181.8960
    Epoch [20/50], Test Losses: mse: 29.3410, mae: 2.9976, huber: 2.5746, swd: 2.4437, ept: 208.7023
      Epoch 20 composite train-obj: 1.495797
            No improvement (3.3121), counter 1/5
    Epoch [21/50], Train Losses: mse: 14.1656, mae: 1.9603, huber: 1.5638, swd: 1.1556, ept: 252.4180
    Epoch [21/50], Val Losses: mse: 42.8043, mae: 3.7712, huber: 3.3388, swd: 3.3813, ept: 180.5722
    Epoch [21/50], Test Losses: mse: 28.2576, mae: 2.9132, huber: 2.4912, swd: 2.0909, ept: 209.9550
      Epoch 21 composite train-obj: 1.563793
            No improvement (3.3388), counter 2/5
    Epoch [22/50], Train Losses: mse: 13.0628, mae: 1.8578, huber: 1.4674, swd: 1.0607, ept: 256.3714
    Epoch [22/50], Val Losses: mse: 41.2377, mae: 3.6883, huber: 3.2569, swd: 3.2780, ept: 182.2425
    Epoch [22/50], Test Losses: mse: 28.0294, mae: 2.8780, huber: 2.4587, swd: 2.2027, ept: 212.7211
      Epoch 22 composite train-obj: 1.467438
            No improvement (3.2569), counter 3/5
    Epoch [23/50], Train Losses: mse: 12.9853, mae: 1.8318, huber: 1.4439, swd: 1.0275, ept: 258.4565
    Epoch [23/50], Val Losses: mse: 43.9078, mae: 3.7961, huber: 3.3613, swd: 2.8913, ept: 182.6521
    Epoch [23/50], Test Losses: mse: 27.3937, mae: 2.9121, huber: 2.4871, swd: 1.7843, ept: 206.3602
      Epoch 23 composite train-obj: 1.443850
            No improvement (3.3613), counter 4/5
    Epoch [24/50], Train Losses: mse: 12.6820, mae: 1.8147, huber: 1.4271, swd: 1.0222, ept: 259.3400
    Epoch [24/50], Val Losses: mse: 40.5759, mae: 3.6140, huber: 3.1858, swd: 3.0995, ept: 189.6215
    Epoch [24/50], Test Losses: mse: 29.0697, mae: 2.8900, huber: 2.4726, swd: 1.9846, ept: 214.3527
      Epoch 24 composite train-obj: 1.427123
            Val objective improved 3.2190 → 3.1858, saving checkpoint.
    Epoch [25/50], Train Losses: mse: 12.3958, mae: 1.7797, huber: 1.3940, swd: 1.0013, ept: 261.4020
    Epoch [25/50], Val Losses: mse: 41.0989, mae: 3.6323, huber: 3.2030, swd: 2.9196, ept: 194.4422
    Epoch [25/50], Test Losses: mse: 27.0022, mae: 2.7926, huber: 2.3762, swd: 1.8239, ept: 220.2584
      Epoch 25 composite train-obj: 1.394003
            No improvement (3.2030), counter 1/5
    Epoch [26/50], Train Losses: mse: 12.1456, mae: 1.7676, huber: 1.3822, swd: 0.9531, ept: 261.3341
    Epoch [26/50], Val Losses: mse: 38.5769, mae: 3.5004, huber: 3.0732, swd: 2.8117, ept: 194.9302
    Epoch [26/50], Test Losses: mse: 27.2886, mae: 2.8004, huber: 2.3849, swd: 2.0086, ept: 221.7224
      Epoch 26 composite train-obj: 1.382237
            Val objective improved 3.1858 → 3.0732, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 11.3982, mae: 1.6807, huber: 1.3012, swd: 0.8835, ept: 266.2733
    Epoch [27/50], Val Losses: mse: 40.5302, mae: 3.5712, huber: 3.1430, swd: 2.8230, ept: 191.7301
    Epoch [27/50], Test Losses: mse: 27.7745, mae: 2.8223, huber: 2.4048, swd: 1.9015, ept: 218.7277
      Epoch 27 composite train-obj: 1.301155
            No improvement (3.1430), counter 1/5
    Epoch [28/50], Train Losses: mse: 11.0069, mae: 1.6474, huber: 1.2688, swd: 0.8507, ept: 267.6575
    Epoch [28/50], Val Losses: mse: 40.6834, mae: 3.5982, huber: 3.1704, swd: 2.8848, ept: 193.4934
    Epoch [28/50], Test Losses: mse: 26.5045, mae: 2.7705, huber: 2.3537, swd: 1.9541, ept: 220.7189
      Epoch 28 composite train-obj: 1.268769
            No improvement (3.1704), counter 2/5
    Epoch [29/50], Train Losses: mse: 11.0243, mae: 1.6492, huber: 1.2711, swd: 0.8402, ept: 267.5001
    Epoch [29/50], Val Losses: mse: 41.0432, mae: 3.6014, huber: 3.1739, swd: 2.6662, ept: 193.8706
    Epoch [29/50], Test Losses: mse: 27.5220, mae: 2.7794, huber: 2.3659, swd: 1.7923, ept: 221.7548
      Epoch 29 composite train-obj: 1.271118
            No improvement (3.1739), counter 3/5
    Epoch [30/50], Train Losses: mse: 11.0473, mae: 1.6316, huber: 1.2560, swd: 0.8482, ept: 270.3698
    Epoch [30/50], Val Losses: mse: 41.6174, mae: 3.6596, huber: 3.2290, swd: 2.7521, ept: 193.9083
    Epoch [30/50], Test Losses: mse: 28.8428, mae: 2.8842, huber: 2.4661, swd: 1.9110, ept: 221.1275
      Epoch 30 composite train-obj: 1.255985
            No improvement (3.2290), counter 4/5
    Epoch [31/50], Train Losses: mse: 10.7799, mae: 1.6251, huber: 1.2478, swd: 0.8281, ept: 271.0114
    Epoch [31/50], Val Losses: mse: 40.0961, mae: 3.5231, huber: 3.0986, swd: 2.7000, ept: 195.5932
    Epoch [31/50], Test Losses: mse: 26.2590, mae: 2.7240, huber: 2.3096, swd: 1.7494, ept: 223.5946
      Epoch 31 composite train-obj: 1.247820
    Epoch [31/50], Test Losses: mse: 27.2853, mae: 2.8003, huber: 2.3848, swd: 2.0087, ept: 221.7133
    Best round's Test MSE: 27.2886, MAE: 2.8004, SWD: 2.0086
    Best round's Validation MSE: 38.5769, MAE: 3.5004, SWD: 2.8117
    Best round's Test verification MSE : 27.2853, MAE: 2.8003, SWD: 2.0087
    Time taken: 152.02 seconds
    
    ==================================================
     Running experiment with seed 20 (3/3)==================================================
    
    Epoch [1/50], Train Losses: mse: 74.6840, mae: 6.4880, huber: 6.0084, swd: 43.3331, ept: 34.4502
    Epoch [1/50], Val Losses: mse: 59.7391, mae: 5.8038, huber: 5.3276, swd: 25.9537, ept: 46.9833
    Epoch [1/50], Test Losses: mse: 55.0161, mae: 5.4131, huber: 4.9408, swd: 25.9127, ept: 47.0270
      Epoch 1 composite train-obj: 6.008396
            Val objective improved inf → 5.3276, saving checkpoint.
    Epoch [2/50], Train Losses: mse: 47.6908, mae: 4.8152, huber: 4.3502, swd: 16.1388, ept: 99.8448
    Epoch [2/50], Val Losses: mse: 51.0704, mae: 5.0876, huber: 4.6194, swd: 14.3898, ept: 87.9772
    Epoch [2/50], Test Losses: mse: 48.6157, mae: 4.8334, huber: 4.3666, swd: 14.3972, ept: 87.6830
      Epoch 2 composite train-obj: 4.350197
            Val objective improved 5.3276 → 4.6194, saving checkpoint.
    Epoch [3/50], Train Losses: mse: 39.7493, mae: 4.1325, huber: 3.6785, swd: 9.0254, ept: 141.5148
    Epoch [3/50], Val Losses: mse: 45.5823, mae: 4.5341, huber: 4.0755, swd: 8.5908, ept: 119.6682
    Epoch [3/50], Test Losses: mse: 40.7259, mae: 4.2056, huber: 3.7489, swd: 8.9783, ept: 119.5294
      Epoch 3 composite train-obj: 3.678523
            Val objective improved 4.6194 → 4.0755, saving checkpoint.
    Epoch [4/50], Train Losses: mse: 35.1956, mae: 3.7279, huber: 3.2822, swd: 6.2372, ept: 161.9518
    Epoch [4/50], Val Losses: mse: 43.6534, mae: 4.3111, huber: 3.8572, swd: 7.0124, ept: 128.6145
    Epoch [4/50], Test Losses: mse: 38.1745, mae: 3.9353, huber: 3.4852, swd: 6.9060, ept: 134.0101
      Epoch 4 composite train-obj: 3.282206
            Val objective improved 4.0755 → 3.8572, saving checkpoint.
    Epoch [5/50], Train Losses: mse: 32.3972, mae: 3.4675, huber: 3.0281, swd: 5.0240, ept: 175.2311
    Epoch [5/50], Val Losses: mse: 41.8422, mae: 4.1616, huber: 3.7118, swd: 6.5923, ept: 139.5820
    Epoch [5/50], Test Losses: mse: 34.5853, mae: 3.7082, huber: 3.2629, swd: 6.9690, ept: 145.7343
      Epoch 5 composite train-obj: 3.028137
            Val objective improved 3.8572 → 3.7118, saving checkpoint.
    Epoch [6/50], Train Losses: mse: 30.7509, mae: 3.3349, huber: 2.8984, swd: 4.3524, ept: 181.5840
    Epoch [6/50], Val Losses: mse: 39.0847, mae: 3.9958, huber: 3.5471, swd: 5.8599, ept: 143.1235
    Epoch [6/50], Test Losses: mse: 33.4196, mae: 3.6128, huber: 3.1682, swd: 5.3776, ept: 153.2270
      Epoch 6 composite train-obj: 2.898396
            Val objective improved 3.7118 → 3.5471, saving checkpoint.
    Epoch [7/50], Train Losses: mse: 28.4607, mae: 3.1421, huber: 2.7110, swd: 3.7542, ept: 190.2999
    Epoch [7/50], Val Losses: mse: 36.3899, mae: 3.7384, huber: 3.2941, swd: 4.0756, ept: 158.1415
    Epoch [7/50], Test Losses: mse: 31.4120, mae: 3.3987, huber: 2.9581, swd: 3.9370, ept: 162.2400
      Epoch 7 composite train-obj: 2.710955
            Val objective improved 3.5471 → 3.2941, saving checkpoint.
    Epoch [8/50], Train Losses: mse: 26.9818, mae: 3.0240, huber: 2.5952, swd: 3.3636, ept: 196.3037
    Epoch [8/50], Val Losses: mse: 37.4599, mae: 3.8053, huber: 3.3606, swd: 3.9394, ept: 159.5377
    Epoch [8/50], Test Losses: mse: 31.1087, mae: 3.3834, huber: 2.9432, swd: 3.6479, ept: 169.6585
      Epoch 8 composite train-obj: 2.595249
            No improvement (3.3606), counter 1/5
    Epoch [9/50], Train Losses: mse: 25.2309, mae: 2.8839, huber: 2.4591, swd: 2.9682, ept: 200.7845
    Epoch [9/50], Val Losses: mse: 39.8937, mae: 3.9091, huber: 3.4640, swd: 4.1581, ept: 152.3282
    Epoch [9/50], Test Losses: mse: 31.2503, mae: 3.3749, huber: 2.9365, swd: 3.6778, ept: 169.3385
      Epoch 9 composite train-obj: 2.459121
            No improvement (3.4640), counter 2/5
    Epoch [10/50], Train Losses: mse: 24.2930, mae: 2.8137, huber: 2.3906, swd: 2.7794, ept: 204.6987
    Epoch [10/50], Val Losses: mse: 36.6415, mae: 3.6682, huber: 3.2278, swd: 3.7862, ept: 169.7869
    Epoch [10/50], Test Losses: mse: 27.9071, mae: 3.1360, huber: 2.7021, swd: 3.4863, ept: 184.0912
      Epoch 10 composite train-obj: 2.390574
            Val objective improved 3.2941 → 3.2278, saving checkpoint.
    Epoch [11/50], Train Losses: mse: 22.5128, mae: 2.6619, huber: 2.2437, swd: 2.4236, ept: 210.3686
    Epoch [11/50], Val Losses: mse: 34.0890, mae: 3.5643, huber: 3.1250, swd: 3.9192, ept: 168.0319
    Epoch [11/50], Test Losses: mse: 28.0779, mae: 3.1683, huber: 2.7345, swd: 3.7520, ept: 178.4438
      Epoch 11 composite train-obj: 2.243736
            Val objective improved 3.2278 → 3.1250, saving checkpoint.
    Epoch [12/50], Train Losses: mse: 21.9830, mae: 2.6289, huber: 2.2108, swd: 2.2988, ept: 211.0147
    Epoch [12/50], Val Losses: mse: 35.7108, mae: 3.5846, huber: 3.1450, swd: 3.4872, ept: 174.6189
    Epoch [12/50], Test Losses: mse: 28.3225, mae: 3.1268, huber: 2.6924, swd: 3.2575, ept: 182.1150
      Epoch 12 composite train-obj: 2.210849
            No improvement (3.1450), counter 1/5
    Epoch [13/50], Train Losses: mse: 20.5922, mae: 2.5098, huber: 2.0957, swd: 2.1107, ept: 216.2761
    Epoch [13/50], Val Losses: mse: 36.4442, mae: 3.6031, huber: 3.1668, swd: 3.4140, ept: 174.9609
    Epoch [13/50], Test Losses: mse: 25.0748, mae: 2.9322, huber: 2.5032, swd: 2.8072, ept: 188.7661
      Epoch 13 composite train-obj: 2.095655
            No improvement (3.1668), counter 2/5
    Epoch [14/50], Train Losses: mse: 19.9274, mae: 2.4700, huber: 2.0560, swd: 1.9563, ept: 216.4522
    Epoch [14/50], Val Losses: mse: 38.5068, mae: 3.6618, huber: 3.2260, swd: 3.2726, ept: 177.6878
    Epoch [14/50], Test Losses: mse: 26.4087, mae: 2.9701, huber: 2.5407, swd: 2.6238, ept: 191.6326
      Epoch 14 composite train-obj: 2.056013
            No improvement (3.2260), counter 3/5
    Epoch [15/50], Train Losses: mse: 18.9076, mae: 2.3844, huber: 1.9736, swd: 1.8007, ept: 220.6263
    Epoch [15/50], Val Losses: mse: 38.6166, mae: 3.6421, huber: 3.2085, swd: 3.4047, ept: 178.6718
    Epoch [15/50], Test Losses: mse: 25.6922, mae: 2.8970, huber: 2.4697, swd: 2.4459, ept: 194.1435
      Epoch 15 composite train-obj: 1.973611
            No improvement (3.2085), counter 4/5
    Epoch [16/50], Train Losses: mse: 18.4870, mae: 2.3378, huber: 1.9287, swd: 1.7399, ept: 224.0895
    Epoch [16/50], Val Losses: mse: 36.7815, mae: 3.4772, huber: 3.0477, swd: 2.5978, ept: 184.2178
    Epoch [16/50], Test Losses: mse: 24.5525, mae: 2.8038, huber: 2.3806, swd: 2.1326, ept: 199.4587
      Epoch 16 composite train-obj: 1.928676
            Val objective improved 3.1250 → 3.0477, saving checkpoint.
    Epoch [17/50], Train Losses: mse: 17.1100, mae: 2.2177, huber: 1.8135, swd: 1.5541, ept: 230.9439
    Epoch [17/50], Val Losses: mse: 34.1082, mae: 3.3345, huber: 2.9054, swd: 2.6069, ept: 194.6503
    Epoch [17/50], Test Losses: mse: 25.1801, mae: 2.8194, huber: 2.3969, swd: 2.3428, ept: 206.2337
      Epoch 17 composite train-obj: 1.813548
            Val objective improved 3.0477 → 2.9054, saving checkpoint.
    Epoch [18/50], Train Losses: mse: 16.4734, mae: 2.1734, huber: 1.7701, swd: 1.4836, ept: 233.9506
    Epoch [18/50], Val Losses: mse: 35.6880, mae: 3.4450, huber: 3.0148, swd: 3.2083, ept: 187.0873
    Epoch [18/50], Test Losses: mse: 26.8347, mae: 2.9318, huber: 2.5062, swd: 2.6346, ept: 200.6930
      Epoch 18 composite train-obj: 1.770060
            No improvement (3.0148), counter 1/5
    Epoch [19/50], Train Losses: mse: 16.0466, mae: 2.1366, huber: 1.7344, swd: 1.4293, ept: 236.1891
    Epoch [19/50], Val Losses: mse: 36.0523, mae: 3.4113, huber: 2.9831, swd: 2.4222, ept: 192.5350
    Epoch [19/50], Test Losses: mse: 25.6805, mae: 2.8272, huber: 2.4051, swd: 2.2429, ept: 206.7734
      Epoch 19 composite train-obj: 1.734434
            No improvement (2.9831), counter 2/5
    Epoch [20/50], Train Losses: mse: 15.6146, mae: 2.0827, huber: 1.6839, swd: 1.3449, ept: 239.6141
    Epoch [20/50], Val Losses: mse: 38.5588, mae: 3.5104, huber: 3.0800, swd: 2.3970, ept: 195.2426
    Epoch [20/50], Test Losses: mse: 26.1060, mae: 2.8423, huber: 2.4188, swd: 2.1822, ept: 210.6239
      Epoch 20 composite train-obj: 1.683874
            No improvement (3.0800), counter 3/5
    Epoch [21/50], Train Losses: mse: 15.6177, mae: 2.0979, huber: 1.6965, swd: 1.3744, ept: 239.8272
    Epoch [21/50], Val Losses: mse: 35.8005, mae: 3.3834, huber: 2.9558, swd: 2.4806, ept: 195.2043
    Epoch [21/50], Test Losses: mse: 26.3501, mae: 2.8729, huber: 2.4493, swd: 2.1665, ept: 207.0478
      Epoch 21 composite train-obj: 1.696512
            No improvement (2.9558), counter 4/5
    Epoch [22/50], Train Losses: mse: 14.9555, mae: 2.0353, huber: 1.6372, swd: 1.3009, ept: 243.3022
    Epoch [22/50], Val Losses: mse: 33.9130, mae: 3.2350, huber: 2.8126, swd: 2.6097, ept: 202.3237
    Epoch [22/50], Test Losses: mse: 23.5998, mae: 2.6599, huber: 2.2448, swd: 1.9589, ept: 214.9655
      Epoch 22 composite train-obj: 1.637242
            Val objective improved 2.9054 → 2.8126, saving checkpoint.
    Epoch [23/50], Train Losses: mse: 13.7762, mae: 1.9194, huber: 1.5279, swd: 1.1674, ept: 248.5473
    Epoch [23/50], Val Losses: mse: 36.7103, mae: 3.3548, huber: 2.9296, swd: 2.2227, ept: 203.3667
    Epoch [23/50], Test Losses: mse: 25.0204, mae: 2.6960, huber: 2.2789, swd: 1.6889, ept: 221.1763
      Epoch 23 composite train-obj: 1.527890
            No improvement (2.9296), counter 1/5
    Epoch [24/50], Train Losses: mse: 13.5179, mae: 1.8964, huber: 1.5057, swd: 1.1427, ept: 250.1277
    Epoch [24/50], Val Losses: mse: 34.1888, mae: 3.2355, huber: 2.8142, swd: 2.2500, ept: 204.4852
    Epoch [24/50], Test Losses: mse: 25.4114, mae: 2.7008, huber: 2.2881, swd: 1.7630, ept: 221.7999
      Epoch 24 composite train-obj: 1.505716
            No improvement (2.8142), counter 2/5
    Epoch [25/50], Train Losses: mse: 13.3125, mae: 1.8819, huber: 1.4910, swd: 1.1083, ept: 251.4449
    Epoch [25/50], Val Losses: mse: 34.3320, mae: 3.2325, huber: 2.8109, swd: 2.3958, ept: 201.1525
    Epoch [25/50], Test Losses: mse: 22.3484, mae: 2.5794, huber: 2.1644, swd: 1.8728, ept: 219.8087
      Epoch 25 composite train-obj: 1.491027
            Val objective improved 2.8126 → 2.8109, saving checkpoint.
    Epoch [26/50], Train Losses: mse: 12.1814, mae: 1.7598, huber: 1.3773, swd: 0.9912, ept: 257.1493
    Epoch [26/50], Val Losses: mse: 32.7552, mae: 3.1469, huber: 2.7249, swd: 1.9558, ept: 209.0676
    Epoch [26/50], Test Losses: mse: 23.2932, mae: 2.6009, huber: 2.1876, swd: 1.5066, ept: 223.5524
      Epoch 26 composite train-obj: 1.377341
            Val objective improved 2.8109 → 2.7249, saving checkpoint.
    Epoch [27/50], Train Losses: mse: 13.1542, mae: 1.8635, huber: 1.4736, swd: 1.0810, ept: 253.7998
    Epoch [27/50], Val Losses: mse: 34.8905, mae: 3.2556, huber: 2.8314, swd: 2.2785, ept: 206.9329
    Epoch [27/50], Test Losses: mse: 21.7687, mae: 2.5321, huber: 2.1166, swd: 1.6750, ept: 223.6333
      Epoch 27 composite train-obj: 1.473566
            No improvement (2.8314), counter 1/5
    Epoch [28/50], Train Losses: mse: 12.1136, mae: 1.7587, huber: 1.3755, swd: 0.9783, ept: 257.9859
    Epoch [28/50], Val Losses: mse: 33.3134, mae: 3.1886, huber: 2.7669, swd: 2.1359, ept: 204.0075
    Epoch [28/50], Test Losses: mse: 22.7227, mae: 2.5804, huber: 2.1672, swd: 1.6178, ept: 223.2996
      Epoch 28 composite train-obj: 1.375483
            No improvement (2.7669), counter 2/5
    Epoch [29/50], Train Losses: mse: 11.5904, mae: 1.6950, huber: 1.3161, swd: 0.9367, ept: 261.4697
    Epoch [29/50], Val Losses: mse: 33.8983, mae: 3.1267, huber: 2.7122, swd: 2.0424, ept: 213.7960
    Epoch [29/50], Test Losses: mse: 22.5990, mae: 2.5235, huber: 2.1165, swd: 1.7565, ept: 229.5613
      Epoch 29 composite train-obj: 1.316073
            Val objective improved 2.7249 → 2.7122, saving checkpoint.
    Epoch [30/50], Train Losses: mse: 12.1759, mae: 1.7682, huber: 1.3840, swd: 0.9810, ept: 257.6907
    Epoch [30/50], Val Losses: mse: 34.6013, mae: 3.2546, huber: 2.8340, swd: 2.3594, ept: 200.4564
    Epoch [30/50], Test Losses: mse: 22.5538, mae: 2.5684, huber: 2.1555, swd: 1.8192, ept: 222.2149
      Epoch 30 composite train-obj: 1.383993
            No improvement (2.8340), counter 1/5
    Epoch [31/50], Train Losses: mse: 11.5323, mae: 1.7016, huber: 1.3209, swd: 0.9143, ept: 260.7917
    Epoch [31/50], Val Losses: mse: 33.5910, mae: 3.1343, huber: 2.7190, swd: 2.2545, ept: 213.9716
    Epoch [31/50], Test Losses: mse: 22.7557, mae: 2.5443, huber: 2.1345, swd: 1.6742, ept: 226.6273
      Epoch 31 composite train-obj: 1.320871
            No improvement (2.7190), counter 2/5
    Epoch [32/50], Train Losses: mse: 11.4470, mae: 1.6963, huber: 1.3161, swd: 0.9065, ept: 260.9916
    Epoch [32/50], Val Losses: mse: 33.9847, mae: 3.2309, huber: 2.8107, swd: 2.6727, ept: 203.2797
    Epoch [32/50], Test Losses: mse: 23.4331, mae: 2.5869, huber: 2.1764, swd: 1.9881, ept: 225.9702
      Epoch 32 composite train-obj: 1.316149
            No improvement (2.8107), counter 3/5
    Epoch [33/50], Train Losses: mse: 11.4511, mae: 1.6961, huber: 1.3157, swd: 0.9131, ept: 261.5330
    Epoch [33/50], Val Losses: mse: 32.7552, mae: 3.0807, huber: 2.6669, swd: 2.0059, ept: 215.2769
    Epoch [33/50], Test Losses: mse: 23.1926, mae: 2.5176, huber: 2.1122, swd: 1.7116, ept: 231.7172
      Epoch 33 composite train-obj: 1.315745
            Val objective improved 2.7122 → 2.6669, saving checkpoint.
    Epoch [34/50], Train Losses: mse: 10.9226, mae: 1.6440, huber: 1.2668, swd: 0.8626, ept: 264.3991
    Epoch [34/50], Val Losses: mse: 33.7219, mae: 3.1212, huber: 2.7070, swd: 2.2216, ept: 216.8296
    Epoch [34/50], Test Losses: mse: 23.0388, mae: 2.4777, huber: 2.0737, swd: 1.6313, ept: 235.0622
      Epoch 34 composite train-obj: 1.266759
            No improvement (2.7070), counter 1/5
    Epoch [35/50], Train Losses: mse: 11.4570, mae: 1.6919, huber: 1.3110, swd: 0.9118, ept: 263.3335
    Epoch [35/50], Val Losses: mse: 33.8078, mae: 3.1666, huber: 2.7472, swd: 2.1778, ept: 211.7570
    Epoch [35/50], Test Losses: mse: 21.6724, mae: 2.4491, huber: 2.0429, swd: 1.7474, ept: 230.9057
      Epoch 35 composite train-obj: 1.310976
            No improvement (2.7472), counter 2/5
    Epoch [36/50], Train Losses: mse: 11.3396, mae: 1.6785, huber: 1.2991, swd: 0.9225, ept: 264.1691
    Epoch [36/50], Val Losses: mse: 33.9100, mae: 3.1506, huber: 2.7363, swd: 2.1519, ept: 211.1867
    Epoch [36/50], Test Losses: mse: 21.9520, mae: 2.4442, huber: 2.0401, swd: 1.5524, ept: 232.2050
      Epoch 36 composite train-obj: 1.299081
            No improvement (2.7363), counter 3/5
    Epoch [37/50], Train Losses: mse: 10.7362, mae: 1.6073, huber: 1.2328, swd: 0.8682, ept: 267.7715
    Epoch [37/50], Val Losses: mse: 35.8196, mae: 3.1975, huber: 2.7862, swd: 2.1531, ept: 216.5322
    Epoch [37/50], Test Losses: mse: 23.2988, mae: 2.5062, huber: 2.1033, swd: 1.5729, ept: 233.2099
      Epoch 37 composite train-obj: 1.232756
            No improvement (2.7862), counter 4/5
    Epoch [38/50], Train Losses: mse: 10.5645, mae: 1.6032, huber: 1.2285, swd: 0.8302, ept: 266.9091
    Epoch [38/50], Val Losses: mse: 31.7992, mae: 2.9937, huber: 2.5841, swd: 1.9933, ept: 216.7538
    Epoch [38/50], Test Losses: mse: 22.3372, mae: 2.4186, huber: 2.0184, swd: 1.5542, ept: 238.1656
      Epoch 38 composite train-obj: 1.228529
            Val objective improved 2.6669 → 2.5841, saving checkpoint.
    Epoch [39/50], Train Losses: mse: 9.8662, mae: 1.5211, huber: 1.1527, swd: 0.7754, ept: 271.4045
    Epoch [39/50], Val Losses: mse: 33.6635, mae: 3.1139, huber: 2.6987, swd: 1.9358, ept: 215.2036
    Epoch [39/50], Test Losses: mse: 21.3358, mae: 2.4113, huber: 2.0059, swd: 1.4856, ept: 234.3138
      Epoch 39 composite train-obj: 1.152665
            No improvement (2.6987), counter 1/5
    Epoch [40/50], Train Losses: mse: 10.4159, mae: 1.5802, huber: 1.2072, swd: 0.8124, ept: 269.2082
    Epoch [40/50], Val Losses: mse: 31.8502, mae: 3.0173, huber: 2.6053, swd: 2.0093, ept: 220.2357
    Epoch [40/50], Test Losses: mse: 21.4266, mae: 2.3875, huber: 1.9855, swd: 1.5591, ept: 238.4363
      Epoch 40 composite train-obj: 1.207162
            No improvement (2.6053), counter 2/5
    Epoch [41/50], Train Losses: mse: 9.8113, mae: 1.5138, huber: 1.1459, swd: 0.7576, ept: 271.7050
    Epoch [41/50], Val Losses: mse: 31.3480, mae: 3.0370, huber: 2.6221, swd: 2.1298, ept: 213.1554
    Epoch [41/50], Test Losses: mse: 24.0684, mae: 2.5543, huber: 2.1479, swd: 1.7031, ept: 233.4647
      Epoch 41 composite train-obj: 1.145913
            No improvement (2.6221), counter 3/5
    Epoch [42/50], Train Losses: mse: 10.5075, mae: 1.5984, huber: 1.2225, swd: 0.8267, ept: 268.7765
    Epoch [42/50], Val Losses: mse: 32.1169, mae: 3.0931, huber: 2.6726, swd: 2.0752, ept: 214.1706
    Epoch [42/50], Test Losses: mse: 21.9206, mae: 2.4480, huber: 2.0393, swd: 1.4173, ept: 234.1501
      Epoch 42 composite train-obj: 1.222480
            No improvement (2.6726), counter 4/5
    Epoch [43/50], Train Losses: mse: 9.1865, mae: 1.4561, huber: 1.0919, swd: 0.7053, ept: 273.7387
    Epoch [43/50], Val Losses: mse: 36.7257, mae: 3.2601, huber: 2.8426, swd: 2.3363, ept: 214.0158
    Epoch [43/50], Test Losses: mse: 21.7123, mae: 2.3954, huber: 1.9921, swd: 1.3278, ept: 238.7236
      Epoch 43 composite train-obj: 1.091900
    Epoch [43/50], Test Losses: mse: 22.3320, mae: 2.4185, huber: 2.0182, swd: 1.5557, ept: 238.1760
    Best round's Test MSE: 22.3372, MAE: 2.4186, SWD: 1.5542
    Best round's Validation MSE: 31.7992, MAE: 2.9937, SWD: 1.9933
    Best round's Test verification MSE : 22.3320, MAE: 2.4185, SWD: 1.5557
    Time taken: 216.05 seconds
    
    ==================================================
    Experiment Summary (ACL_lorenz_seq336_pred336_20250514_1855)
    ==================================================
    Number of runs: 3
    Seeds: [1955, 7, 20]
    
    Test Performance at Best Validation (mean ± std):
      mse: 25.2282 ± 2.1050
      mae: 2.6036 ± 0.1561
      huber: 2.1972 ± 0.1497
      swd: 1.6267 ± 0.2868
      ept: 233.8361 ± 8.6810
      count: 15.0000 ± 0.0000
    
    Corresponding Validation Performance (mean ± std):
      mse: 35.6723 ± 2.8505
      mae: 3.2177 ± 0.2110
      huber: 2.8008 ± 0.2035
      swd: 2.1224 ± 0.5182
      ept: 212.0179 ± 12.4765
      count: 15.0000 ± 0.0000
    ==================================================
    Three seeds Time taken: 547.47 seconds
    
    Experiment complete: ACL_lorenz_seq336_pred336_20250514_1855
    Model: ACL
    Dataset: lorenz
    Sequence Length: 336
    Prediction Length: 336
    Seeds: [1955, 7, 20]
    


