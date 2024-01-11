from dataclasses import dataclass
from omegaconf import MISSING
from typing import List, Dict, Optional, Any


@dataclass
class DatasetConfig:
    name: str = "MISSING"
    root_dir: str = "/MISSING"
    # tasks
    tasks: List[str] = MISSING

    # number of workers for data loaders
    num_workers: int = 8

@dataclass
class MIMICDatasetConfig(DatasetConfig):
    name: str = "MIMIC"
    root_dir: str = "PUT DATA DIR MIMIC HERE"

@dataclass
class ModelConfig:
    # model name (resnet18mimic)
    name: str = MISSING
    # train config
    seed: int = 42
    device: str = "cuda"
    # general
    batch_size: int = 128
    epochs: int = 10
    lr: float = 0.001

@dataclass
class Resnet18MIMICConfig(ModelConfig):
    name: str = "resnet18mimic"


@dataclass
class MultiModalModelConfig(ModelConfig):
    #just an exqmple
    name: str = "MultiModalModel"


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "mml"
    wandb_group: str = "mml"
    wandb_run_name: str = "mml"
    wandb_project_name: str = "mml"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "PUT LOG DIR HERE"

@dataclass
class MyConfig:
    # Dataset Config
    dataset: DatasetConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # logger
    log: LogConfig = MISSING