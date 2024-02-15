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
    root_dir_split: str = "/Users/ago/PycharmProjects/mml/data"
    trans_resize: int = 224
    augmentation: str = "imagenet_style"
    reduced_size: bool = True


@dataclass
class ExperimentConfig:
    # binary_classification
    task: str = "binary_classification"
    # experiment name
    name: str = "experiment"
    # target columns
    target_list: List[str] = field(default_factory=lambda: ['Pneumonia'])
    # labeling policy
    label_policy: str = "uncertain_to_negative"
    # viewPosition PA, AP, FRONTAL
    view_position: str = "FRONTAL"
    # splitting method random, original
    splitting_method: str = "original"
    # splitting seed
    seed: int = 42
    # train val split ratio, ignored if splitting method is original
    train_val_split: float = 0.8
    # test val split ratio, ignored if splitting method is original
    test_val_split: float = 0.5
    # checkpoint metric
    checkpoint_metric: str = "val_auc"
    # checkpoint mode
    checkpoint_mode: str = "max"

@dataclass
class MultiLabelExperimentConfig(ExperimentConfig):
    task: str = "multi_label_classification"
    # experiment name
    name: str = "multi label experiment"
    # target columns
    target_list: List[str] = field(default_factory=lambda: ['Atelectasis', 'Cardiomegaly',
                                                            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                                                            'Fracture',
                                                            'Lung Lesion', 'Lung Opacity', 'No Finding',
                                                            'Pleural Effusion',
                                                            'Pleural Other', 'Pneumonia', 'Pneumothorax',
                                                            'Support Devices'])
    # labeling policy
    label_policy: str = "uncertain_to_negative"
    # splitting method
    splitting_method: str = "original"
    # viewPosition
    view_position: str = "FRONTAL"



@dataclass
class ModelConfig:
    # model name
    name: str = MISSING
    # train config
    seed: int = 42
    device: str = "cuda"
    # general
    batch_size: int = 128
    epochs: int = 150
    initial_lr: float = 0.0001
    pretrained: bool = True




@dataclass
class MultiModalModelConfig(ModelConfig):
    # just an example
    name: str = "MultiModalModel"


@dataclass
class LogConfig:
    # wandb ?anandrea
    wandb_entity: str = "aa335"
    wandb_group: str = "mml"
    # wandb_run_name: str = "mml"
    wandb_project_name: str = "mml"
    wandb_log_freq: int = 50
    wandb_offline: bool = False

    # logs
    dir_logs: str = "../logs"


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
