import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../configs', config_name='preprocessing_config')
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
