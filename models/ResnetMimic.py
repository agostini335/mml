from torchvision.models import resnet18
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import roc_auc_score


# TODO implement cfg
class ResnetMimic(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.model.initial_lr
        self.pretrained = False
        self.model = resnet18(pretrained=self.pretrained)
        self.model.fc = nn.Linear(512, out_features=1, bias=True)
        self.loss = nn.BCEWithLogitsLoss()
        print(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        # compute logits
        logits = self(x)
        # compute loss
        loss = self.loss(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_no):
        x, y = batch
        # compute logits
        logits = self(x)
        # compute loss
        loss = self.loss(logits, y)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        try:
            auc = roc_auc_score(y.cpu(), logits.cpu())
        except:
            auc = 0
            print("ERROR AUC")

        self.log('val_auc', auc, prog_bar=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        print('epoch end')

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), self.lr)
