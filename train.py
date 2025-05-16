import torch
import torch.nn as nn
# import FLOW  
from collections import defaultdict 
from abc import ABC 
from typing import List, Dict, Tuple 
from metrics import SWDMetric, TargetStdMetric, EPTMetric, EPTHingeLoss, EPTHingeLossSmooth

from monotonic import ACL 
import numpy as np
from PatchTST import Model as PatchTST
from DLinear import Model as DLinear
from dataclasses import dataclass, field
from typing import List
from TimeMixer import Model as TimeMixer
import os
from datetime import datetime
from copy import deepcopy
import time
import json
from typing import Optional
from train_config import BaseTrainingConfig
from data_manager import DatasetManager
import dataclasses
from typing import Callable

# === Metrics ===
# region Metrics
@dataclass 
class MetricAccumulator:
    """Tracks a single metric's statistics at batch level"""
    total: float = 0.0
    total_sq: float = 0.0
    count: int = 0 

    def accumulate_loss(self, x: float):
        self.total += x
        self.total_sq += x ** 2
        self.count += 1

    def update_mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    # def update_std(self) -> float:
    #     if self.count < 2: return 0.0
    #     mean = self.update_mean()
    #     return ((self.total_sq / self.count) - mean ** 2) ** 0.5
    
    def reset(self):
        self.total = 0.0
        self.total_sq = 0.0
        self.count = 0

    def get_count(self):
        return self.count


# === Loss Factory === 
@dataclass
class MetricEntry:
    """ Handles Epoch level metrics for different phases"""
    name: str
    cfg: BaseTrainingConfig = field(repr=False)
    weight_backward: float = 0.0
    weight_validate: float = 0.0

    fn: Callable[..., torch.Tensor] = field(init=False, repr=False)

    # per phase accumulator
    train_accumulator: MetricAccumulator = field(default_factory=MetricAccumulator)
    val_accumulator: MetricAccumulator = field(default_factory=MetricAccumulator)
    test_accumulator: MetricAccumulator = field(default_factory=MetricAccumulator)

    def __post_init__(self):
        factory = LOSS_FN_FACTORY[self.name]
        self.fn = factory(self.cfg)
        self._phase_map = {
            'Train': self.train_accumulator,
            'Val':   self.val_accumulator,
            'Test':  self.test_accumulator,
        }
    
    def accumulate_batch_loss_for_phase(self, phase: str, value: float):
        self._phase_map[phase].accumulate_loss(value)

    def reset(self, phase: Optional[str] = None):
        if phase is None or phase.lower() == 'all':
            self.train_accumulator.reset()
            self.val_accumulator.reset()
            self.test_accumulator.reset()
        else:
            self._phase_map[phase].reset()

    def get_epoch_mean_for_phase(self, phase: str):
        return self._phase_map[phase].update_mean()
    
    def get_count(self, phase: str):
        return self._phase_map[phase].get_count()
    
    # def get_epoch_std_for_phase(self, phase: str):
    #     return self._phase_map[phase].update_std()
    
    def compute_batch_loss(self, pred: torch.Tensor, target: torch.Tensor):  
        return self.fn(pred, target)
    
    def compute_and_accumulate_batch_loss(self, phase: str, pred: torch.Tensor, target: torch.Tensor):  
        raw_result = self.fn(pred, target)
        self._phase_map[phase].accumulate_loss(raw_result.item())
        return raw_result
    

LOSS_FN_FACTORY = {
    'mse': lambda cfg: nn.MSELoss(),
    'mae': lambda cfg: nn.L1Loss(),
    'huber': lambda cfg: nn.HuberLoss(delta=1),
    'swd': lambda cfg: SWDMetric(dim=cfg.pred_len, num_proj=cfg.num_proj_swd, seed=cfg.seed),
    # 'target_std': lambda cfg: TargetStdMetric(),
    'ept': lambda cfg: EPTMetric(cfg.ept_global_std),
    'ept_hinge': lambda cfg: EPTHingeLoss(cfg.ept_global_std),
    'ept_hinge_smooth': lambda cfg: EPTHingeLossSmooth(cfg.ept_global_std),
}


class LossManager:
    def __init__(self, cfg: BaseTrainingConfig):
        self.cfg = cfg
        self.metrics = []
        for i, loss_name in enumerate(cfg.losses):
            self.metrics.append(MetricEntry(
                name=loss_name,
                cfg=cfg,
                weight_backward=cfg.loss_backward_weights[i],
                weight_validate=cfg.loss_validate_weights[i],
            ))

        # Early stopping
        self.best_val_loss = float('inf') # Best validation loss

        # Best metrics
        self.best_val_metrics = None
        self.best_test_metrics = None
        self.train_obj_acc = MetricAccumulator()
         
    def compute_all_losses(self, phase: str, pred: torch.Tensor, target: torch.Tensor):
        return {metric.name: metric.get_epoch_mean_for_phase(phase) for metric in self.metrics}

    def reset_loss_manager(self, which: str = 'all'):
        for metric in self.metrics:
            metric.reset(which)

    def accumulate(self, losses: Dict[str, torch.Tensor], phase: str):
        stats = self.epoch_metric_dict.select_phase(phase.capitalize()) # Ensure phase name matches keys in PhaseStatsManager
        scalar_losses = {name: float(val.item()) for name, val in losses.items() if name in stats.metrics_dict} # Ensure metric exists
        stats.accumulate_all_losses(scalar_losses)

    def report_phase_metric_means(self, phase: str, epoch: int = None, total_epochs: int = None):
        header = f"Epoch [{epoch+1}/{total_epochs}], " if epoch is not None else ""
        mean_dict, std_dict = {}, {}
        for metric in self.metrics:
            mean_dict[metric.name] = metric.get_epoch_mean_for_phase(phase)
        
        loss_str = ", ".join( f"{k}: {mean_dict[k]:.4f}" for k in mean_dict )
        print(f"{header}{phase} Losses: {loss_str}")
        return mean_dict
    
    def accumulate_and_return_weighted_phase_loss(self, phase, pred, targets):
        # total = torch.tensor(0.0, device=self.cfg.device)
        terms = []
        for e in self.metrics:
            raw = e.compute_and_accumulate_batch_loss(phase, pred, targets)
            w   = e.weight_backward if phase=='Train' else e.weight_validate
            terms.append(w * raw)
        total = terms[0] if len(terms)==1 else sum(terms)
        if total is None:
            raise ValueError(f"Loss value is None for metric {e.name} in phase {phase}")
        if phase=='Train':
            self.train_obj_acc.accumulate_loss(total.item())
        return total
    
    def get_count(self, phase: str):
        return self.metrics[0].get_count(phase=phase.capitalize())
    
    def get_train_obj_mean(self):
        return self.train_obj_acc.update_mean()


# === Callbacks ===
# Quick note: This is an arrangement to say "what happens when".
# region Callbacks
class Callback(ABC):
    def on_epoch_begin(self, epoch: int, **kwargs): pass
    def on_forward_end(self, **kwargs): 
        """Subclasses may:
        - Accumulate losses for batch level logging
        - Compute loss to backward or validate
        """
        return None 
    def on_batch_end(self, phase: str, **kwargs): 
        """
        - Prediction collector collect data if it is a test phase
        """
        return None 
    def on_phase_begin(self, phase: str, **kwargs):
        """
        - reset phase metrics
        """
        return None
    def on_phase_end(self, phase: str, **kwargs): 
        """ 
        - report phase stats
        - clear out recorded val losses
        """
        return None 
    def on_epoch_end(self, epoch: int, total_epochs: int, **kwargs): 
        """ 
        - reset test tracker for best model calculation
        """
        return None 
    
    def on_experiment_end(self, epoch: int, total_epochs: int, **kwargs):
        """
        - save best model
        """
        return None

class LossCallback(Callback):
    def __init__(self, loss_manager: LossManager):
        self.loss_manager = loss_manager
        

    def on_epoch_begin(self, epoch):
        self.loss_manager.reset_loss_manager(which='all')

    def on_forward_end(self, phase, pred, target):
        loss_to_use = self.loss_manager.accumulate_and_return_weighted_phase_loss(phase, pred, target)
        
        # all_losses  = self.loss_manager.compute_all_losses(pred, batch_y)
        # loss_to_use  = self.loss_manager.compute_loss_to_use(is_training, all_losses)
        return loss_to_use
 

    def on_phase_begin(self, phase, **kwargs):
        self.loss_manager.reset_loss_manager(which=phase)
        if phase == 'Train':
            self.loss_manager.train_obj_acc.reset()

    def on_phase_end(self, phase, epoch, total_epochs, **kwargs):
        dict_phase_metric_means = self.loss_manager.report_phase_metric_means(phase, epoch, total_epochs)
        dict_phase_metric_means['count'] = self.loss_manager.get_count(phase='Val')
        # TODO: Add early stopping
        # self.loss_manager.reset(type='phase')
        return dict_phase_metric_means
    
    def on_epoch_end(self, epoch, total_epochs):
        # Reset test tracker for best model Calculation
        
        pass

    def on_experiment_end(self, epoch, total_epochs):
        # save best model
        # self.checkpoint_mgr.save_model(self.model, self.config.seed)
        pass

class PredictionCollector(Callback):
    def __init__(self):
        self.preds = []
        self.targets = []
        self.inputs = []

    def on_batch_end(self, phase: str, batch_x, pred, batch_y):
        if phase == 'Test': # We only collect test data
            self.inputs.append(batch_x.detach().cpu())
            self.preds.append(pred.detach().cpu())
            self.targets.append(batch_y.detach().cpu())

    def on_phase_begin(self, phase: str, **_):
        if phase == 'Test':
            self.reset()

    def get_all(self):
        import torch
        return { # [B, S, D] 
            'inputs':  self.inputs,
            'preds':   self.preds,
            'targets': self.targets
        }
    def reset(self):
        self.preds = []
        self.targets = []
        self.inputs = []
 
# endregion
# === Unify Model Signatures ===
class ACLAdapter(nn.Module):
    """ACL already speaks (x, update)—just passthrough."""
    def __init__(self, acl_model):
        super().__init__()
        self.model = acl_model

    def forward(self, x, update: bool = True):
        return self.model(x, update=update)

class PatchTSTAdapter(nn.Module):
    """PatchTST wants (x, None, None, None)."""
    def __init__(self, pst_model):
        super().__init__()
        self.model = pst_model

    def forward(self, x, update: bool = True):
        return self.model(x, None, None, None)

class TimeMixerAdapter(nn.Module):
    """TimeMixer wants (x, None, None, None)."""
    def __init__(self, tm_model):
        super().__init__()
        self.model = tm_model

    def forward(self, x, update: bool = True):
        return self.model(x, None, None, None)

class DLinearAdapter(nn.Module):
    """DLinear wants just x."""
    def __init__(self, dlinear_model):
        super().__init__()
        self.model = dlinear_model

    def forward(self, x, update: bool = True):
        return self.model(x)

class NaiveAdapter(nn.Module):
    """Naive baseline: repeats last value."""
    def __init__(self, seq_len, pred_len, device):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device

    def forward(self, x, update: bool = True):
        # x: [B, S, D] → repeat last along time
        last = x[:, -1:, :]              # [B, 1, D]
        return last.repeat(1, self.pred_len, 1)  # [B, pred_len, D]



def LossFactory(cfg):
    losses = []
    for name, disp, bw, vw, phase in zip(
        cfg.loss_names, cfg.loss_display, cfg.loss_weight_backward, 
        cfg.loss_weight_validate, ['Train', 'Val', 'Test']):
        if disp or bw > 0 or vw > 0:
            losses.append(MetricEntry(
                name = name,
                fn = LOSS_FN_FACTORY[name](cfg),
                display = disp,
                weight_backward = bw,
                weight_validate = vw,
                phase = phase
            ))
    return losses

# === Model Factory ===
MODEL_REGISTRY: Dict[str, Callable[[BaseTrainingConfig], nn.Module]] = {
    'ACL':      lambda cfg: ACLAdapter(ACL(cfg)),
    'PatchTST': lambda cfg: PatchTSTAdapter(PatchTST(cfg)),
    'DLinear':  lambda cfg: DLinearAdapter(DLinear(cfg)),
    'TimeMixer':lambda cfg: TimeMixerAdapter(TimeMixer(cfg)),
    'naive':    lambda cfg: NaiveAdapter(cfg.seq_len, cfg.pred_len, cfg.device),
}

def ModelFactory(cfg):
    try:
        build_fn = MODEL_REGISTRY[cfg.model_type]
    except KeyError:
        raise ValueError(f"Unknown model: {cfg.model_type}")
    model = build_fn(cfg)
    return model.to(cfg.device)


    
def execute_model_evaluation(dataname:str, train_config: BaseTrainingConfig, data_manager: DatasetManager, scale: bool = False):
    """
    Run an experiment with multiple seeds for a specific model and data configuration. 
    Returns: MultiSeedExperiment object with results
    """
    cfg = train_config
    data_manager.prepare_data(dataname, seq_len=cfg.seq_len, pred_len=cfg.pred_len, batch_size=cfg.batch_size, scale=scale)
    train_loader, val_loader, test_loader = data_manager.get_loaders()
    cfg.ept_global_std = data_manager.datasets[dataname]['global_std']

    dataset_name = data_manager.current_dataset
    experiment_name = f"{cfg.model_type}_{dataset_name}_seq{cfg.seq_len}_pred{cfg.pred_len}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Run the multi-seed experiment
    experiment = MultiSeedRunner(
        base_config=cfg,
        seeds=cfg.seeds,
        experiment_name=experiment_name,
        raw_test_data=data_manager.raw_test_data
    )

    # Execute the experiment
    experiment.run_multiple_seeds(train_loader, val_loader, test_loader)

    # Print summary
    print(f"\nExperiment complete: {experiment_name}")
    print(f"Model: {cfg.model_type}")
    print(f"Dataset: {dataset_name}")
    print(f"Sequence Length: {cfg.seq_len}")
    print(f"Prediction Length: {cfg.pred_len}")
    print(f"Seeds: {cfg.seeds}")

    return experiment

class CheckpointManager:
    """
    Handles model saving/loading and creation of necessary directories.
    """
    def __init__(self, experiment_name: str, model_type: str):
        self.base_path = os.path.join("results", experiment_name)
        self.models_path = os.path.join(self.base_path, "models")
        self.predictions_path = os.path.join(self.base_path, "predictions")
        self.model_type = model_type

     # Create directories
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.predictions_path, exist_ok=True)
    
    def save_model(self, model: nn.Module, seed: int):
        save_path = os.path.join(self.models_path, f"{self.model_type}_seed_{seed}.pt")
        torch.save(model.state_dict(), save_path)
        return save_path

    def load_model(self, model: nn.Module, seed: int):
        load_path = os.path.join(self.models_path, f"{self.model_type}_seed_{seed}.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No checkpoint found at {load_path}.")
        model.load_state_dict(torch.load(load_path))
        return load_path
    
    def save_predictions(self, predictions: Dict[str, torch.Tensor], seed: int):
        save_path = os.path.join(self.predictions_path, f"predictions_seed_{seed}.pt")
        torch.save(predictions, save_path)
        return save_path

@dataclass
class ExperimentResult:
    """Stores the results of a single training run."""
    cfg: Dict
    val_losses: Dict[str, Dict[str, float]]
    test_losses: Dict[str, Dict[str, float]]
    preds: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    inputs: Optional[torch.Tensor] = None
    model_state_dict: Optional[Dict] = None
    training_time: float = 0.0
    seed: int = 0
    raw_test_data: Optional[torch.Tensor] = None



    def to_dict(self):
        """Convert to a dictionary for serialization. 
        We don't include tensors or model state in the dict version """
        def sanitize(o):
            if isinstance(o, (str, int, float, bool)) or o is None:
                return o
            if isinstance(o, dict):
                return {k: sanitize(v) for k, v in o.items()}
            if isinstance(o, list):
                return [sanitize(v) for v in o]
            # fallback for classes, tensors, etc:
            return str(o)
        result = { 
            'config': sanitize(self.cfg),
            'val_losses': self.val_losses,
            'test_losses': self.test_losses,
            'training_time': self.training_time,
            'seed': self.seed
        }
        return result

def reconstruct_series_from_windows_torch(
        windows: torch.Tensor,   # (B, D, S)
        stride: int = 1,
        offset: int = 0,         # shift into the full timeline
        mode: str = 'average'    # 'average' or 'sum'
    ) -> torch.Tensor:
        """
        Reconstruct a series of length T = offset + (B-1)*stride + S 
        from overlapping windows.

        For inputs (seq windows) use offset=0.

        For preds (forecast windows) use offset=seq_len (or label_len if you have one).

        windows[b,:, :] covers positions
        [ offset + b*stride : offset + b*stride + S ]  

        Returns:
            full: Tensor of shape (D, T)

        Example: 
        x = [0,1 ,2 ,3 ,4 ,5]
        seq_len = 3, pred_len = 2, stride = 1. Window = x[b: b+3]
        b | window = x[b: b+3]
        b = 0 -> [0,1,2]
        b = 1 -> [1,2,3]
        b = 2 -> [2,3,4]
        b = 3 -> [3,4,5]
        B_inp = 4, S = 3.
        Tnp = offset + (Bnp - 1) * stride + S = 0 + (4 - 1) * 1 + 3 = 6
        Predictions:
        b | pred window = x[b+3 : b+3+2]
        b = 0 -> [3,4]
        b = 1 -> [4,5]
        B_pred = seq_len - pred_len + 1 = 2
        With offset = seq_len = 3:
        Tpred = offset + (Bpred - 1) * stride + pred_len 
              = 3 + (2 - 1) * 1 + 2 = 6
        
        After overlap-add (e.g. average), you fill in the true forecast slots at t=3-5, 
        leaving t=0-2 untouched (still zero in the acc array).

        Inputs reconstruct the entire original series when you use offset=0. The tail stays at 0
        Preds reconstruct only the tail (from t=seq_len onward) when you use offset=seq_len.
        Times 0…seq_len-1 are untouched (zero)

        # reconstruct inputs and preds separately
        inp_full  = reconstruct_series_from_windows_torch(inp_windows,  stride=1, offset=0,          mode='average')
        pred_full = reconstruct_series_from_windows_torch(pred_windows, stride=1, offset=seq_len,    mode='average')

        # now stitch them: use the true inputs up to seq_len, then the preds
        full = inp_full.clone()
        full[:, seq_len:] = pred_full[:, seq_len:]
        """
        B, D, S = windows.shape
        T = offset + (B - 1) * stride + S
        acc   = torch.zeros((D, T), dtype=windows.dtype, device=windows.device)
        count = torch.zeros((D, T), dtype=torch.int32, device=windows.device)
        for b in range(B):
            start = offset + b * stride
            acc[:, start:start+S]    += windows[b]
            count[:, start:start+S]  += 1

        if mode == 'average':
            count = count.clamp(min=1)
            return acc / count.to(windows.dtype)
        elif mode == 'sum':
            return acc
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        
class MultiSeedRunner:
    """
    Orchestrates running the same config across multiple random seeds.
    Aggregates and saves results.
    """ 
    def __init__(self, base_config: BaseTrainingConfig, 
                 seeds: List[int], experiment_name: str = None,
                 raw_test_data: torch.Tensor = None): 
        self.cfg = deepcopy(base_config)
        self.seeds = seeds
        self.raw_test_data = raw_test_data # [channels, total_step]
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{base_config.model_type}_{timestamp}"
        else:
            self.experiment_name = experiment_name

        self.results = []
            
        # Create directories for results
        os.makedirs("results", exist_ok=True)
        self.checkpoint_mgr = CheckpointManager(
            self.experiment_name, base_config.model_type)

    


    def _run_single_seed(self, seed, seed_idx, train_loader, val_loader, test_loader):
        print(f"\n{'='*50}\n Running experiment with seed {seed} ({seed_idx+1}/{len(self.seeds)}){'='*50}\n")
            
        current_config = deepcopy(self.cfg)
        current_config.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
            
        trainer = ModelTrainer(config=current_config, checkpoint_mgr=self.checkpoint_mgr)
        start_time = time.time()
        best_val_losses, best_test_losses, test_data_dict  = trainer.train_model(train_loader, val_loader, test_loader)
        end_time = time.time()

        print(f"Time taken: {end_time - start_time:.2f} seconds")
            
        result = ExperimentResult(
            cfg=dataclasses.asdict(current_config),
            val_losses=best_val_losses,
            test_losses=best_test_losses,
            preds=test_data_dict['preds'],
            targets=test_data_dict['targets'],
            inputs=test_data_dict['inputs'],
            raw_test_data=self.raw_test_data,
            model_state_dict=trainer.model.state_dict(),
            training_time=end_time - start_time,
            seed=seed
        )

        return result
    

    def run_multiple_seeds(self, train_loader, val_loader, test_loader):
        """
        Iterate over seeds, run training, and store results. 
        """
        total_time = 0
        for i, seed in enumerate(self.seeds):
            start_time = time.time()
            result = self._run_single_seed(seed, i, train_loader, val_loader, test_loader)
            self.results.append(result)
            self._save_experiment_result(result)
            end_time = time.time()
            total_time += end_time - start_time
        

        agg_result = self._compute_aggregates()
        self._save_aggregates(agg_result)
        self._print_summary(agg_result)

        print(f"Three seeds Time taken: {total_time:.2f} seconds")

        return self
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save a single experiment result."""
        txt_path = os.path.join("results", self.experiment_name, f"result_seed_{result.seed}.txt")
        with open(txt_path, 'w') as f:
            f.write(str(result.to_dict()))
        
        if result.model_state_dict is not None: # Save model state
            model_path = os.path.join("results", self.experiment_name, "models", f"model_seed_{result.seed}.pt")
            torch.save(result.model_state_dict, model_path)
        
        # if result.preds is not None: # Save predictions
        #     pred_path = os.path.join("results", self.experiment_name, "predictions", f"predictions_seed_{result.seed}.pt")
        #     torch.save({
        #         'predictions': result.preds,
        #         'targets': result.targets,
        #         'inputs': result.inputs
        #     }, pred_path)

    def _compute_aggregates(self):
        """Compute and save aggregate statistics across all seeds."""
        if not self.results:
            return
        
        val_losses, test_losses = defaultdict(list), defaultdict(list)
        
        for result in self.results:
            for loss_type, value in result.val_losses.items():
                val_losses[loss_type].append(value)
            for loss_type, value in result.test_losses.items():
                test_losses[loss_type].append(value)
        
        return { # Create aggregate result
            'val': {
                'mean': {k: np.mean(v) for k, v in val_losses.items()},
                'std': {k: np.std(v) for k, v in val_losses.items()},
                'values': dict(val_losses)
            },
            'test': {
                'mean': {k: np.mean(v) for k, v in test_losses.items()},
                'std': {k: np.std(v) for k, v in test_losses.items()},
                'values': dict(test_losses)
            },
            'config': self.cfg,
            'seeds': self.seeds,
            'experiment_name': self.experiment_name
        }  
        
    def _save_aggregates(self, agg_result):
        full_result = {
            **agg_result,
            'config': self.cfg,
            'seeds': self.seeds,
            'experiment_name': self.experiment_name
        }
        agg_path = os.path.join("results", self.experiment_name, "aggregate_results.txt") # Save aggregate result
        with open(agg_path, 'w') as f:
            f.write(str(full_result))
            
        
    def _print_summary(self, agg_result):
        print("\n" + "="*50) # Print summary
        print(f"Experiment Summary ({self.experiment_name})")
        print("="*50)
        print(f"Number of runs: {len(self.seeds)}")
        print(f"Seeds: {self.seeds}")
        print("\nTest Performance at Best Validation (mean ± std):")
        for loss_type in agg_result['test']['mean'].keys():
            print(f"  {loss_type}: {agg_result['test']['mean'][loss_type]:.4f} ± {agg_result['test']['std'][loss_type]:.4f}")
        print("\nCorresponding Validation Performance (mean ± std):")
        for loss_type in agg_result['val']['mean'].keys():
            print(f"  {loss_type}: {agg_result['val']['mean'][loss_type]:.4f} ± {agg_result['val']['std'][loss_type]:.4f}")
        print("="*50)

class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int, checkpoint_mgr: CheckpointManager, 
            model: nn.Module,
            seed: int,
            monitor: str = 'composite'):  # or 'mse', etc.)
        self.patience = patience
        self.ckpt_mgr = checkpoint_mgr
        self.model = model
        self.monitor = monitor
        self.best_val = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
        # self.best_val_metrics = None
        # self.best_test_metrics = None
        self.seed = seed
    
    def reset(self):
        self.counter = 0
        self.best_val = float('inf')

    def on_phase_end(self, loss_to_use: torch.Tensor, phase: str, epoch: int, total_epochs: int, **_):
        pass
        # if phase != 'Val':
        #     return
        # else:
        #     mean_val = loss_to_use.item()
        #     if mean_val < self.best_val:
        #         print(f"        Val objective improved {self.best_val:.4f} → {mean_val:.4f}, saving checkpoint.")
        #         self.best_val    = mean_val
        #         self.counter     = 0
        #         self.ckpt_mgr.save_model(self.model, self.seed) 
        #         # self.best_val_metrics = self.loss_manager.best_val_metrics
        #         # self.best_test_metrics = self.loss_manager.best_test_metrics

        #     else:
        #         self.counter += 1
        #         print(f"        No improvement ({mean_val:.4f}), counter {self.counter}/{self.patience}")
        #         if self.counter >= self.patience:
        #             print("  Early stopping triggered.")
        #             self.should_stop = True

    def on_epoch_end(self, epoch: int, total_epochs: int, **_):
        # self.reset()
        pass

class ModelTrainer:
    def __init__(self, config: BaseTrainingConfig, checkpoint_mgr: CheckpointManager):
        self.cfg = config
        self.generator = torch.Generator().manual_seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.epochs = config.epochs  
        self.model = ModelFactory(self.cfg)
        # print("Model Parameter Gradients Required:")
        # for name, param in self.model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"  - {name}: REQUIRES_GRAD=FALSE")
        # print("---")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate#, fused=True, weight_decay=0,
        )
        self.loss_manager = LossManager(cfg=self.cfg)

        self.patience_counter = 0
        self.best_val_loss = float('inf')

        self.checkpoint_mgr = checkpoint_mgr

        # Callbacks Management
        self.callbacks: List[Callback] = [
            LossCallback(self.loss_manager),
            # EarlyStoppingCallback(self.cfg.patience, self.checkpoint_mgr),
            # you can add more, e.g. SWD logging, TensorBoard, etc.
        ]

        self.collector = PredictionCollector()
        self.callbacks.append(self.collector)
        self.earlystop_cb = EarlyStoppingCallback(self.cfg.patience, 
            self.checkpoint_mgr, self.model, seed=self.cfg.seed)
        self.callbacks.append(self.earlystop_cb)

        # Best metrics
        self.best_val_metrics = None
        self.best_test_metrics = None

    def _get_checkpoint_name(self) -> str:
        """Override in subclass to specify model-specific checkpoint name."""
        return 'checkpoint.pth'
 

    def run_batch(self, batch_x, batch_y, phase: str):
        """Runs a single batch through the model, computes loss, and performs backward pass if training."""
        # on_batch_begin
        is_training = phase.capitalize() == 'Train'
        phase = phase.capitalize()
        loss_to_use = None
        
        with torch.set_grad_enabled(is_training):
            out = self.model(batch_x, update=is_training)
            if isinstance(out, dict):
                pred = out.get('pred', None) # Use .get for safety
                if pred is None:
                    raise ValueError("Model output is dict but has no 'pred' key")

            elif hasattr(out, 'pred'):
                pred = out.pred
            elif isinstance(out, torch.Tensor):
                pred = out
            else:
                raise TypeError(f"Unexpected model output type: {type(out)}")
            # print(pred)
            if not isinstance(pred, torch.Tensor):
                raise TypeError(f"Prediction is not a Tensor after extraction: {type(pred)}")
            if pred.shape != batch_y.shape:
                raise ValueError(f"Shape mismatch: pred {pred.shape}, target {batch_y.shape}")
            
            # if is_training: 
            #     print(f"  [Train Batch] Pred requires_grad: {pred.requires_grad}, Pred Max: {pred.abs().max().item():.4f}")

            loss_cb_found = False
            for cb in self.callbacks:
                # We assume LossCallback is the one returning the loss for backward
                if isinstance(cb, LossCallback):
                    loss_cb_found = True
                    result = cb.on_forward_end(
                        phase=phase,
                        pred=pred,
                        target=batch_y,
                    )
                if result is not None and isinstance(result, torch.Tensor):
                    loss_to_use = result
                elif is_training: # Add a check if training loss is None unexpectedly
                    print(f"  WARNING: LossCallback on_forward_end didn't return a Tensor for phase {phase}")
                else:
                    cb.on_forward_end(
                        phase=phase,
                        pred=pred,
                        target=batch_y,
                    )
                    # This shouldn't happen if LossCallback is implemented correctly
                    # print(f"  WARNING: LossCallback on_forward_end didn't return a Tensor for phase {phase}")
            if loss_to_use is None and is_training:
                 if not loss_cb_found:
                     raise RuntimeError("LossCallback instance not found in self.callbacks list!")
                 else:
                     raise ValueError(f"Loss value (loss_to_use) is still None after checking LossCallback in phase {phase}")
            # if is_training:
            #      print(f"  [Train Batch] loss_to_use: {loss_to_use.item():.6f}, Requires Grad: {loss_to_use.requires_grad}")
            # losses = self.loss_tracker.compute_batch_losses(pred, batch_y)
            # loss_to_backward = self.choose_loss_to_backward(losses)
            if loss_to_use is None:
                raise ValueError(f"No loss value returned from callbacks in phase {phase}")
            
            if phase.capitalize() == 'Train' and self.cfg.model_type != 'naive': 
                self.optimizer.zero_grad()
                # loss_to_use.backward()
                try:
                    loss_to_use.backward()
                    # --- GRAD CHECK 3: Parameter Gradients ---
                    grads = [p.grad.abs().max().item() for p in self.model.parameters() if p.grad is not None]
                    # if not grads:
                    #      print("  [Train Batch] WARNING: No gradients found on any parameters after backward!")
                    # else:
                    #      pass
                        #  print(f"  [Train Batch] Max abs grad among params: {max(grads):.6f}")

                except Exception as e:
                    # print(f"  ERROR during backward pass: {e}")
                    # Handle or re-raise
                    raise e
                self.optimizer.step()

        return loss_to_use, pred
    

    def evaluate_phase(self, data_loader, phase: str, epoch: int, collect_data: bool = False) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        assert phase in ['Train', 'Val', 'Test']
        is_training = phase == 'Train'
        self.model.train() if is_training else self.model.eval()

        # on_phase_begin
        for cb in self.callbacks:
            cb.on_phase_begin(phase=phase, epoch=epoch, total_epochs=self.cfg.epochs)

        with torch.set_grad_enabled(is_training):
            for batch_x, batch_y in data_loader:  
                batch_x = batch_x.to(self.cfg.device)
                batch_y = batch_y.to(self.cfg.device)
                # on_batch_begin

                loss_to_use, pred = self.run_batch(batch_x, batch_y, phase=phase)
                
                # on_batch_end
                for cb in self.callbacks: 
                    cb.on_batch_end(
                        phase=phase,
                        batch_x=batch_x,
                        pred=pred,
                        batch_y=batch_y,
                    )
        # on_phase_end
        for cb in self.callbacks: 
            result = cb.on_phase_end(
                phase=phase,
                epoch=epoch,
                total_epochs=self.cfg.epochs,
                loss_to_use=loss_to_use
            )
            if result is not None:
                dict_phase_metric_means = result 

        return dict_phase_metric_means
    
    def train_model(self, train_loader, val_loader, test_loader=None):
        """Main training loop with optional test evaluation at each epoch.""" 
        self.earlystop_cb.reset()

        # Create directory for checkpoints if it doesn't exist
        # os.makedirs("checkpoint", exist_ok=True)
        # checkpoint_path = os.path.join("checkpoint", self._get_checkpoint_name())

        for epoch in range(self.cfg.epochs): 
            # on_epoch_begin
            for cb in self.callbacks: cb.on_epoch_begin(epoch)
            
            dict_train = self.evaluate_phase(train_loader, "Train", epoch) 
            dict_val = self.evaluate_phase(val_loader, "Val", epoch)
            dict_test = self.evaluate_phase(test_loader, "Test", epoch)

            train_obj = sum(
                metric.weight_backward * dict_train[metric.name]
                for metric in self.loss_manager.metrics
            )
            print(f"  Epoch {epoch+1} composite train-obj: {train_obj:.6f}")

            val_obj = sum(
                metric.weight_validate * dict_val[metric.name]
                for metric in self.loss_manager.metrics
            )
            
            # val_obj = dict_val['loss']
            if val_obj < self.earlystop_cb.best_val:
                # update best in the same place you save the model
                print(f"        Val objective improved {self.earlystop_cb.best_val:.4f} → {val_obj:.4f}, saving checkpoint.")
                self.earlystop_cb.best_val = val_obj
                self.loss_manager.best_val_metrics  = dict_val.copy()
                self.loss_manager.best_test_metrics = dict_test.copy()
                self.checkpoint_mgr.save_model(self.model, self.cfg.seed)
                self.earlystop_cb.counter = 0

            else:
                self.earlystop_cb.counter += 1
                if self.earlystop_cb.counter >= self.earlystop_cb.patience:
                    break
                print(f"        No improvement ({val_obj:.4f}), counter {self.earlystop_cb.counter}/{self.earlystop_cb.patience}")
            # if self.earlystop_cb.should_stop:
            #     print("Early stopping")
            #     break

            # on_epoch_end
            for cb in self.callbacks: 
                cb.on_epoch_end(epoch=epoch, total_epochs=self.cfg.epochs)   
            # print(f"  Avg train objective: {self.loss_manager.get_train_obj_mean():.6f}")

        self.checkpoint_mgr.load_model(self.model, self.cfg.seed)
        self.model.eval()
        
          
        
        # self.loss_tracker.reset(type='test')
        best_test_losses = self.evaluate_phase(test_loader, "Test", epoch, collect_data=True)

        test_data_dict = self.collector.get_all() 
        
        
        # assert np.allclose(recorded_best_test_losses['mse'], best_test_losses['mse'])
        for cb in self.callbacks:
            cb.on_experiment_end(epoch=epoch, total_epochs=self.cfg.epochs) 
            
        print(f"Best round's Test MSE: {self.loss_manager.best_test_metrics['mse']:.4f}, MAE: {self.loss_manager.best_test_metrics['mae']:.4f}, SWD: {self.loss_manager.best_test_metrics['swd']:.4f}")  
        print(f"Best round's Validation MSE: {self.loss_manager.best_val_metrics['mse']:.4f}, MAE: {self.loss_manager.best_val_metrics['mae']:.4f}, SWD: {self.loss_manager.best_val_metrics['swd']:.4f}")      
        print(f"Best round's Test verification MSE : {best_test_losses['mse']:.4f}, MAE: {best_test_losses['mae']:.4f}, SWD: {best_test_losses['swd']:.4f}")
        torch.cuda.empty_cache()
        return self.loss_manager.best_val_metrics, self.loss_manager.best_test_metrics, test_data_dict # Return best test losses

    
