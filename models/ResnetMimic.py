import os
from torchvision.models import resnet18, resnet50
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelAUROC


class ResnetMimicBinary(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.model.initial_lr
        self.pretrained = cfg.model.pretrained

        if cfg.model.name == 'resnet18':
            if self.pretrained:
                raise NotImplementedError
            else:
                self.model = resnet18()
            self.model.fc = nn.Linear(512, out_features=1, bias=True)
        elif cfg.model.name == 'resnet50':
            if self.pretrained:
                raise NotImplementedError
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


class ResnetMimicMulti(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.lr = cfg.model.initial_lr
        self.pretrained = cfg.model.pretrained
        self.root_pretrained_models = cfg.model.root_dir_pretrained
        self.target_list = cfg.experiment.target_list

        self.train_auc = MultilabelAUROC(num_labels=len(self.target_list))
        self.val_auc = MultilabelAUROC(num_labels=len(self.target_list))
        self.train_auc_vec = MultilabelAUROC(num_labels=len(self.target_list), average=None)
        self.val_auc_vec = MultilabelAUROC(num_labels=len(self.target_list), average=None)

        if cfg.model.name == 'resnet18':
            self.model = resnet18()
            if self.pretrained:
                # url="https://download.pytorch.org/models/resnet18-f37072fd.pth", pretrained weights
                self.model.load_state_dict(torch.load(os.path.join(self.root_pretrained_models, "resnet18-f37072fd.pth")))
            # change the last layer for multi-label classification
            self.model.fc = nn.Linear(512, out_features=len(self.target_list), bias=True)
            # define loss for multi-label classification
            self.loss = nn.BCEWithLogitsLoss()
            print(self.model)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_auc(logits, y.to(torch.int))
        self.log('train_auc', self.train_auc, prog_bar=True, on_step=False, on_epoch=True)
        self.train_auc_vec(logits, y.to(torch.int))
        # log auc for each label
        for i, label in enumerate(self.target_list):
            self.log(f'train_auc_{label}', self.train_auc_vec[i], prog_bar=True, on_step=False, on_epoch=True,
                     metric_attribute='train_auc_vec')
        return loss

    def validation_step(self, batch, batch_no):
        x, y = batch
        # compute logits
        logits = self(x)
        # compute loss
        loss = self.loss(logits, y)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.val_auc(logits, y.to(torch.int))
        self.log('val_auc', self.val_auc, prog_bar=True, on_step=False, on_epoch=True)
        self.val_auc_vec(logits, y.to(torch.int))
        # log auc for each label
        for i, label in enumerate(self.target_list):
            self.log(f'val_auc_{label}', self.val_auc_vec[i], prog_bar=False, on_step=False, on_epoch=True,
                     metric_attribute='val_auc_vec')  # TODO check order

    def on_train_epoch_end(self):
        print('\n epoch end \n')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)


# auc roc
# https://discuss.pytorch.org/t/multi-class-roc/39849/3
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# example
# https://discuss.pytorch.org/t/multi-label-classification-in-pytorch/905/28
# https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
