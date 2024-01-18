import importlib
import sys

sys.path.append("..")
from utils.utils import get_transformations

importlib.import_module('MyConfig')
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from MyConfig import MyConfig
from hydra.core.config_store import ConfigStore
from cs_manager import set_cs
from dataio.MimicCXRDataset import get_splits_MIMIC_CXR, CXRDataset
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import test
from models.ResnetMimic import ResnetMimic

# set config store
set_cs()


def create_dataloaders(cfg):
    if cfg.dataset.name == 'MIMIC-CXR' and cfg.experiment.task == 'binary_classification':

        train_dict, test_dict, val_dict = get_splits_MIMIC_CXR(cfg)
        train_transform, test_transform, val_transform = get_transformations(cfg)

        # get datasets
        train_dataset = CXRDataset(label_name=cfg.experiment.target_list[0], labels_df=train_dict['labels'],
                                   images=train_dict['images'], transform=train_transform)

        test_dataset = CXRDataset(label_name=cfg.experiment.target_list[0], labels_df=test_dict['labels'],
                                  images=test_dict['images'], transform=test_transform)

        val_dataset = CXRDataset(label_name=cfg.experiment.target_list[0], labels_df=val_dict['labels'],
                                 images=val_dict['images'], transform=val_transform)
        # get dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)

        test_dataloader = DataLoader(test_dataset, batch_size=cfg.model.batch_size, shuffle=False)

        val_dataloader = DataLoader(test_dataset, batch_size=cfg.model.batch_size, shuffle=False)

        return train_dataloader, test_dataloader, val_dataloader
    else:
        raise NotImplementedError


def get_model(cfg):
    if cfg.model.name == 'resnet18mimic':
        model = ResnetMimic()
        return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyConfig):
    print(OmegaConf.to_yaml(cfg))
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(cfg)
    model = get_model(cfg)

    # Setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project="mimic-mml",
        group="experiment_1",
        job_type='new_config',
        mode="online",
    )
    project_name = 'Resnet18_mimic'
    wandb_logger = WandbLogger(project=project_name, log_model="all")
    wandb_logger.watch(model, log="all")
    trainer = pl.Trainer(max_epochs=cfg.model.epochs, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    wandb.finish()


if __name__ == "__main__":
    run_experiment()


