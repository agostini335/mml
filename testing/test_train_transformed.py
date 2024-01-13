import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import sys

sys.path.append("..")
from dataio.MimicCXRDataset import CXRDataset
from torchvision import transforms


from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from dataio.MimicCXRDataset import train_test_split_CXR
from models.ResnetMimic import ResnetMimic

with initialize(version_base=None, config_path="../configs"):
    cfg_training = compose(config_name="basic_resnet_experiment")
    cfg_dataset = compose(config_name="dataset_config")
print(cfg_training)
print(cfg_dataset)

# create datasets
train_dict, test_dict, val_dict = train_test_split_CXR(cfg_dataset)

transformList = [transforms.ToTensor()]

transform = transforms.Compose(transformList)


# transform train
transformList = [transforms.RandomAffine(degrees=(0, 5), translate=(0.05, 0.05), shear=(5)),
                 transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
train_transform = transforms.Compose(transformList)



# transform test
transformList = [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
test_transform = transforms.Compose(transformList)




train_dataset = CXRDataset(label_name=cfg_dataset.class_names[0], labels_df=train_dict['labels'],
                           images=train_dict['images'], transform=train_transform)

test_dataset = CXRDataset(label_name=cfg_dataset.class_names[0], labels_df=test_dict['labels'],
                          images=test_dict['images'], transform=test_transform)

# create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ResnetMimic()
wandb.finish()

if __name__ == '__main__':

    # Setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project="mimic-mml",
        group="experiment_1",
        job_type='pneumonia_transform',
        mode="online",
    )
    project_name = 'Resnet18_mimic'
    wandb_logger = WandbLogger(project=project_name, log_model="all")
    wandb_logger.watch(model, log="all")
    trainer = pl.Trainer(max_epochs=cfg_training.epochs, logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    wandb.finish()
