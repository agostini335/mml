from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import List, Dict, Optional


@dataclass
class DatasetConfig:
    name: str = "MISSING"
    root_dir: str = "MISSING"
    # number of workers for data loaders
    num_workers: int = 8


@dataclass
class MIMICDatasetConfig(DatasetConfig):
    name: str = "MIMIC-CXR"
    root_dir_PA: str = "/Users/ago/PycharmProjects/mml/data/PA/mimic-cxr-preprocessed-16012024"
    root_dir_AP: str = "/Users/ago/PycharmProjects/mml/data/AP/mimic-cxr-preprocessed-17012024"
    trans_resize: int = 224
    augmentation: str = "imagenet_style"


@dataclass
class ExperimentConfig:
    # task not used at the moment
    task: str = "binary_classification"
    # experiment name
    name: str = "experiment"
    # target columns
    target_list: List[str] = field(default_factory=lambda: ["Pneumonia"])
    # labeling policy
    label_policy: str = "remove_uncertain"
    # viewPosition
    view_position: str = "PA"
    # splitting method
    splitting_method: str = "random"
    # splitting seed
    seed: int = 42
    # train val split ratio
    train_val_split: float = 0.6
    # test val split ratio
    test_val_split: float = 0.5
    # checkpoint metric
    checkpoint_metric: str = "val_auc"
    # checkpoint mode
    checkpoint_mode: str = "max"

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
    initial_lr: float = 0.0001


@dataclass
class Resnet18MIMICConfig(ModelConfig):
    name: str = "resnet18mimic"


@dataclass
class MultiModalModelConfig(ModelConfig):
    # just an example
    name: str = "MultiModalModel"


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "aa335"
    wandb_group: str = "mml"
    wandb_run_name: str = "mml"
    wandb_project_name: str = "mml"
    wandb_log_freq: int = 50
    wandb_offline: bool = True

    # logs
    dir_logs: str = "/Users/ago/PycharmProjects/mml/logs"


@dataclass
class MyConfig:
    # Dataset Config
    dataset: DatasetConfig = MISSING
    # model
    model: ModelConfig = MISSING
    # logger
    log: LogConfig = MISSING
    # experiment
    experiment: ExperimentConfig = MISSING
