from torchvision.models import resnet18, resnet50, ResNet50_Weights, ResNet18_Weights
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score


class ResnetMimicBinary(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.model.initial_lr
        self.pretrained = cfg.model.pretrained

        if cfg.model.name == 'resnet18':
            if self.pretrained:
                self.model = resnet18(weights=ResNet18_Weights)
            else:
                self.model = resnet18()
            self.model.fc = nn.Linear(512, out_features=1, bias=True)
        elif cfg.model.name == 'resnet50':
            if self.pretrained:
                self.model = resnet50(weights=ResNet50_Weights)
            else:
                self.model = resnet50()
            self.model.fc = nn.Linear(2048, out_features=1, bias=True)
        else:
            raise NotImplementedError

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
            self.log("ERROR in VAL AUC", 0)
        except:
            auc = 0
            print("ERROR AUC")
            self.log("ERROR in VAL AUC", 1)

        self.log('val_auc', auc, prog_bar=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        print('epoch end')

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), self.lr)
