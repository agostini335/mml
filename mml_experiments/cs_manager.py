from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import DictConfig, OmegaConf

from MyConfig import MyConfig
from MyConfig import MIMICDatasetConfig, LogConfig, Resnet18MIMICConfig, DatasetConfig, ModelConfig, \
    MultiModalModelConfig, ExperimentConfig


def set_cs():
    # configuring config store
    cs = ConfigStore.instance()

    # DatasetConfig
    cs.store(group="dataset", name="mimic", node=MIMICDatasetConfig)

    # ModelConfig
    cs.store(group="model", name="resnet18mimic", node=Resnet18MIMICConfig)
    cs.store(group="model", name="MultiModalModel", node=MultiModalModelConfig)

    # LogConfig
    cs.store(group="log", name="log", node=LogConfig)

    # ExperimentConfig
    cs.store(group="experiment", name="experiment", node=ExperimentConfig)

    # base config
    cs.store(name="base_config", node=MyConfig)
    return cs
