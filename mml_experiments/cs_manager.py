from hydra.core.config_store import ConfigStore
import hydra
from omegaconf import DictConfig, OmegaConf

from MyConfig import MyConfig
from MyConfig import MIMICDatasetConfig, LogConfig, DatasetConfig, ModelConfig, \
    MultiModalModelConfig, ExperimentConfig, MultiLabelExperimentConfig


def set_cs():
    # configuring config store
    cs = ConfigStore.instance()

    # DatasetConfig
    cs.store(group="dataset", name="mimic", node=MIMICDatasetConfig)

    # ModelConfig
    cs.store(group="model", name="model", node=ModelConfig)
    cs.store(group="model", name="MultiModalModel", node=MultiModalModelConfig)

    # LogConfig
    cs.store(group="log", name="log", node=LogConfig)

    # ExperimentConfig
    cs.store(group="experiment", name="experiment", node=ExperimentConfig)
    cs.store(group="experiment", name="multilabel_experiment", node=MultiLabelExperimentConfig)

    # base config
    cs.store(name="base_config", node=MyConfig)
    return cs
