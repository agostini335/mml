import importlib
import sys

sys.path.append("..")
from utils.utils import get_transformations

importlib.import_module('MyConfig')
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from MyConfig import MyConfig
from cs_manager import set_cs
from dataio.MimicCXRDataset import get_splits_MIMIC_CXR, CXRDataset
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import test
from models.ResnetMimic import ResnetMimicBinary

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

        val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size, shuffle=False)

        return train_dataloader, test_dataloader, val_dataloader
    elif cfg.dataset.name == 'MIMIC-CXR' and cfg.experiment.task == 'multi_label_classification':
        train_dict, test_dict, val_dict = get_splits_MIMIC_CXR(cfg)
        #ok fino a qua
        train_transform, test_transform, val_transform = get_transformations(cfg)
    else:
        raise NotImplementedError


def get_model(cfg):
    if cfg.model.name == 'resnet18' or cfg.model.name == 'resnet50':
        model = ResnetMimicBinary(cfg)
        return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: MyConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(cfg)
    model = get_model(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.log.dir_logs,
        monitor=cfg.experiment.checkpoint_metric,
        mode=cfg.experiment.checkpoint_mode,
        save_last=True,
    )
    wandb_logger = WandbLogger(
        #name=cfg.log.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        save_dir=cfg.log.dir_logs,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        devices=1,
        accelerator="gpu" if cfg.model.device == "cuda" else cfg.model.device,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        deterministic=True,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    run_experiment()
