from typing import Optional, Union, Callable, Tuple, List
from copy import deepcopy
from pathlib import Path

from addict import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from deeptime.util.types import to_dataset
from deeptime.decomposition import VAMP
from deeptime.decomposition.deep import VAMPNet, vampnet_loss, VAMPNetModel


def create_lobes(input_dim, device, output_dim): 
    lobe = nn.Sequential(
        nn.BatchNorm1d(input_dim), 
        nn.Linear(input_dim, 30), nn.ELU(),
        nn.Linear(30, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 100), nn.ELU(),
        nn.Linear(100, 30), nn.ELU(),
        nn.Linear(30, output_dim),
        nn.Softmax(dim=1)  # obtain fuzzy probability distribution over output states
    )
    
    lobe_timelagged = deepcopy(lobe).to(device=device)
    lobe = lobe.to(device=device)
    return (lobe, lobe_timelagged)


def split_data(data: List[np.ndarray], config: Dict) -> Tuple[DataLoader, DataLoader]:

    dataset = to_dataset(data=data, lagtime=config.lag_time)
    n_val = int(len(dataset)*config.validation_split)
    train_data, val_data = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])

    loader_train = DataLoader(train_data, batch_size=10000, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    return (loader_train, loader_val)


def load_data(data_path: Path) -> List[np.ndarray]:
    with np.load(data_path.open('rb')) as fh:
        data = [fh[f"arr_{i}"].astype(np.float32) for i in range(3)] 


def fit(config: Dict, device: torch.device):
    pass
     


def run(config: Dict):
    pass
    # config = Dict(lag_time = 1, 
    #             validation_split = 0.3, 
    #             input_dim = int(data[0].shape[1]), 
    #             output_dim = 6, 
    #             lr = 5e-4, 
    #             n_epochs = 30, 
    #             optimizer=torch.optim.Adam, 
    #             score_method='VAMP2', 
    #             score_mode='regularize', 
    #             score_eps=1e-6)