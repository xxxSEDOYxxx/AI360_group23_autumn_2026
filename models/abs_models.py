import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path


class AbstractModel(nn.Module):
    @abstractmethod
    def __init__(self, nb_classes):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class BaseLogger(ABC):
    @abstractmethod
    def __call__(self, model: AbstractModel, global_step: int) -> None:
        pass

    @abstractmethod
    def get_history(self) -> Any:
        pass


@dataclass
class TrainConfig:
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    batch_size: int
    epoch_count: int
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    logger: Optional[BaseLogger] = None
    is_warm_start: bool = False


def fit_model(model: AbstractModel, config: TrainConfig):
        pass

def load_model(model: AbstractModel, load_path: Path):
    pass

def save_model(model: AbstractModel, save_path: Path):
    pass
