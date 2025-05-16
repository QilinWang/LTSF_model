
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd
import numpy as np 
from data_generator import generate_trajectory, TrajectoryFactory
from dataclasses import dataclass
from typing import Optional, Dict, Any

class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for time series data.

    Args:
        data_x (array-like): Input time series data.
        data_y (array-like): Target time series data.
        seq_len (int): Length of the input sequences.
        label_len (int): Length of the known part of the target sequences.
        pred_len (int): Length of the prediction horizon.
        device (str): Device to allocate tensors ('cuda' or 'cpu').

    """
    def __init__(self, data_x, data_y, seq_len, label_len, pred_len, device='cuda'):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        if seq_len <= 0 or label_len < 0 or pred_len <= 0:
            raise ValueError("Sequence lengths must be positive integers.")
        # self.device = device

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        input_start = index
        input_end = input_start + self.seq_len
        target_start = input_end - self.label_len
        target_end = target_start + self.label_len + self.pred_len

        seq_x = self.data_x[input_start:input_end]
        if not isinstance(seq_x, torch.Tensor):
            seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = self.data_y[target_start:target_end]
        if not isinstance(seq_y, torch.Tensor):
            seq_y = torch.tensor(seq_y, dtype=torch.float32)

        return seq_x, seq_y


def generate_data_split(dataset, train_split, val_split, test_dataset=None, device='cuda'):
    # Validate split ratios
    if not (0 < train_split < val_split < 1):
        raise ValueError("train_split must be between 0 and val_split; val_split must be between train_split and 1.")

    total_num = len(dataset)

    # Compute split indices
    split_idx_train = int(total_num * train_split)
    split_idx_val = int(total_num * val_split)

    # Split the data
    train_data = dataset[:split_idx_train].to(device)
    val_data = dataset[split_idx_train:split_idx_val].to(device)
    if test_dataset is None:
        test_data = dataset[split_idx_val:].to(device)
    else:
        test_data = test_dataset[split_idx_val:]

    # Print shapes for verification
    print(f"Shape of training data: {train_data.shape}") 
    print(f"Shape of validation data: {val_data.shape}")
    print(f"Shape of testing data: {test_data.shape}")
    return train_data, val_data, test_data

def prepare_data_loaders(
    train_data, val_data, test_data,
    seq_len: int,
    label_len: int,
    pred_len: int,
    batch_size: int, 
    device: str = 'cuda'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the total_data into training, validation, and test sets,
    creates TimeSeriesDataset instances, and wraps them in DataLoaders.

    Args:
        total_data (torch.Tensor): The complete dataset to be split.
        seq_len (int): Length of the input sequences.
        label_len (int): Length of the label sequences.
        pred_len (int): Length of the prediction sequences.
        batch_size (int): Batch size for DataLoaders.
        train_split (float): Fraction of data to use for training (between 0 and 1).
        val_split (float): Fraction of data to use for validation (between train_split and 1).
        device (str): Device to allocate tensors ('cuda' or 'cpu').

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and testing.
    """ 
    # Create TimeSeriesDataset instances
    train_dataset = TimeSeriesDataset(
        data_x=train_data,
        data_y=train_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        device=device
    )
    val_dataset = TimeSeriesDataset(
        data_x=val_data,
        data_y=val_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        device=device
    )
    test_dataset = TimeSeriesDataset(
        data_x=test_data,
        data_y=test_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        device=device
    )

    # Print dataset sample shapes
    print(f"Train set sample shapes: {train_dataset[0][0].shape}, {train_dataset[0][1].shape}")
    print(f"Validation set sample shapes: {val_dataset[0][0].shape}, {val_dataset[0][1].shape}")
    print(f"Test set data shapes: {test_dataset.data_x.shape}, {test_dataset.data_y.shape}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Optionally, inspect a batch from the training loader
    print(f"Number of batches in train_loader: {len(train_loader)}")
    for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data shape {data_batch.shape}, Target shape {target_batch.shape}")
        # print(f"Data: {data_batch[0, 1:3, 0:]}, Target: {target_batch[0, 1:3, 0:]}")
        if batch_idx == 0:
            break  # Only inspect the first batch

    return train_loader, val_loader, test_loader

@dataclass
class DatasetInfo:
    data: torch.Tensor
    type: str
    name: str
    shape: Tuple[int, int]
    channels: int
    length: int
    extra: Optional[Dict[str, Any]] = None

class DatasetManager:
    """Manages dataset loading, preprocessing, and information display."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.datasets = {}
        self.current_dataset = None
        self.raw_train_data = None
        self.raw_val_data = None
        self.raw_test_data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.channels = None
    
    def load_trajectory(self, name="lorenz", steps=50399, dt=1e-2, **kwargs):
        """Load a synthetic trajectory dataset."""
        trajectory = TrajectoryFactory.create_trajectory(
            system=name,
            steps=steps,
            dt=dt,
            **kwargs,
        )
        self.datasets[name] = {
            'data': trajectory,
            'type': 'synthetic',
            'name': name,
            'shape': trajectory.shape,
            'channels': trajectory.shape[-1],
            'length': len(trajectory),
            'params': {'steps': steps, 'dt': dt, **kwargs}
        }
        self.print_dataset_info(name)
        self.current_dataset = name
        return self
    
    def load_csv(self, name, path, date_column='date', replace_value=-9999.0, 
                 drop_na=True, random_feature_num_seed_pair=None):
        """Load a dataset from CSV file following the original steps."""
        # Read CSV and preprocess data
        df = pd.read_csv(path)
        df = df.replace(replace_value, np.nan) # for weather dataset, replace -9999.0 with NaN
        if drop_na:
            df = df.dropna()

        # Drop the date column if it exists (as in the original code)
        if date_column in df.columns:
            relevant_columns = df.columns.drop(date_column)
        else:
            relevant_columns = df.columns

        if random_feature_num_seed_pair is None:
            data = df[relevant_columns].values
        else:
            random_feature_num, random_seed = random_feature_num_seed_pair
            np.random.seed(random_seed)
            selected_columns = np.random.choice(relevant_columns, size=random_feature_num, replace=False)
            print(f"Original Num Columns: {len(relevant_columns)}, Selected Num Columns: {len(selected_columns)}")
            print("Selected Columns:", selected_columns)
            data = df[selected_columns].values


        # Convert data to tensor (assumes remaining columns are numeric)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        
        # Store dataset info
        self.datasets[name] = {
            'data': data_tensor,
            'type': 'csv',
            'name': name,
            'path': path,
            'shape': data_tensor.shape,
            'channels': data_tensor.shape[-1],
            'length': len(data_tensor),
            'df': df  # Store the DataFrame for reference
        }

        self.print_dataset_info(name)
        self.current_dataset = name
        return self
    
    def calculate_global_std(self, dataset: np.ndarray, standardize=False) -> np.ndarray:
        """
        Calculate the global standard deviation of each feature (channel) in the dataset.

        Args:
            dataset (np.ndarray): The input dataset of shape [samples, features].

        Returns:
            np.ndarray: An array of standard deviation values, one per feature.
        """
        if isinstance(dataset, torch.Tensor):
            dataset = dataset.cpu().numpy()
        # print(f"dataset.shape: {dataset.shape}")
        
        # Calculate standard deviation along the sample axis (0)
        global_std = np.std(dataset, axis=0)
        
        # print(f"Global Standard Deviation per Feature: {global_std}")
        return global_std

    def prepare_data(self, dataset_name, seq_len, pred_len, batch_size,
                     train_split=0.7, val_split=0.8, 
                     scale=False):
        """Prepare data splits and loaders for a dataset."""
        # Use specified dataset or current one
        dataset_name = dataset_name or self.current_dataset
        if dataset_name is None or dataset_name not in self.datasets:
            raise ValueError("No dataset specified or loaded")
        
        dataset = self.datasets[dataset_name]['data']
        
        # Split data
        raw_train_data, raw_val_data, raw_test_data = generate_data_split(
            dataset=dataset,
            train_split=train_split,
            val_split=val_split,
            test_dataset=None
        )
        
        # Scale data if requested
        if scale:
            scaler = StandardScaler()
            raw_train_data = torch.tensor(
                scaler.fit_transform(raw_train_data.cpu().numpy()), 
                dtype=torch.float32, device=self.device
            )
            raw_val_data = torch.tensor(
                scaler.transform(raw_val_data.cpu().numpy()), 
                dtype=torch.float32, device=self.device
            )
            raw_test_data = torch.tensor(
                scaler.transform(raw_test_data.cpu().numpy()), 
                dtype=torch.float32, device=self.device
            )
            dataset = scaler.transform(dataset)

        # Calculate global std and store it
        global_std = torch.tensor(self.calculate_global_std(dataset), dtype=torch.float32, device=self.device)
        print(f"global_std.shape: {global_std.shape}")
        print(f"Global Std for {dataset_name}: {global_std}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            raw_train_data, raw_val_data, raw_test_data,
            seq_len=seq_len,
            label_len=0,
            pred_len=pred_len,
            batch_size=batch_size,
            device=self.device
        )
        
        # Store everything
        self.raw_train_data = raw_train_data.permute(1,0) # [total_step, channels] -> [channels, total_step]
        self.raw_val_data = raw_val_data.permute(1,0)
        self.raw_test_data = raw_test_data.permute(1,0)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.channels = raw_train_data.shape[-1]
        
        # Store preparation parameters in dataset info
        self.datasets[dataset_name].update({
            'prepared': True,
            'seq_len': seq_len,
            'pred_len': pred_len,
            'batch_size': batch_size,
            'scale': scale,
            'train_split': train_split,
            'val_split': val_split,
            'global_std': global_std
        })
        
        # Print preparation summary
        self.print_preparation_info(dataset_name)
        return self
    
    def print_dataset_info(self, name=None):
        """Print information about a dataset."""
        name = name or self.current_dataset
        if name is None or name not in self.datasets:
            print("No dataset specified or loaded")
            return
        
        dataset = self.datasets[name]
        print("\n" + "="*50)
        print(f"Dataset: {name} ({dataset['type']})")
        print("="*50)
        print(f"Shape: {dataset['shape']}")
        print(f"Channels: {dataset['channels']}")
        print(f"Length: {dataset['length']}")
        
        if dataset['type'] == 'synthetic':
            print(f"Parameters: {dataset['params']}")
        elif dataset['type'] == 'csv':
            print(f"Source: {dataset['path']}")
            
        # Print sample data
        data = dataset['data']
        if len(data) > 0:
            print("\nSample data (first 2 rows):")
            print(data[:2])
        
        print("="*50)
    
    def print_preparation_info(self, name=None):
        """Print information about data preparation."""
        name = name or self.current_dataset
        if name is None or name not in self.datasets:
            print("No dataset specified or loaded")
            return
        
        if 'prepared' not in self.datasets[name] or not self.datasets[name]['prepared']:
            print(f"Dataset {name} has not been prepared yet")
            return
        
        dataset = self.datasets[name]
        print("\n" + "="*50)
        print(f"Data Preparation: {name}")
        print("="*50)
        print(f"Sequence Length: {dataset['seq_len']}")
        print(f"Prediction Length: {dataset['pred_len']}")
        print(f"Batch Size: {dataset['batch_size']}")
        print(f"Scaling: {'Yes' if dataset['scale'] else 'No'}")
        print(f"Train Split: {dataset['train_split']}")
        print(f"Val Split: {dataset['val_split']}")
        print(f"Training Batches: {len(self.train_loader)}")
        print(f"Validation Batches: {len(self.val_loader)}")
        print(f"Test Batches: {len(self.test_loader)}")
        print("="*50)
    
    def get_loaders(self):
        """Get the current data loaders."""
        return self.train_loader, self.val_loader, self.test_loader
    



# importlib.reload(train_config)
# acl_config = ModelConfigFactory.get_config(
#     'ACL', seq_len=336, channels=CHANNELS
# )
# trainer = Train.TSTrainer(config=acl_config)
# train_losses, val_losses, test_losses = trainer.train(train_loader, val_loader, test_loader)
