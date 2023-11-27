import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config


class ModelErm(pl.LightningModule):
    def __init__(self, num_classes=2):
        super(ModelErm, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss',loss,on_step=False,on_epoch=True,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('validation_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def common_step(self, batch, batch_idx):
        x, y, c = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=config.first_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        return [optimizer], [scheduler]


class ModelAfr(pl.LightningModule):
    def __init__(self, first_stage_model, num_classes=2, freeze_base=True):
        super(ModelAfr, self).__init__()
        self.model = copy.deepcopy(first_stage_model)
        self.gamma = config.gamma
        self.regularization = config.regularization
        self.initial_param_last_layer = []
        if freeze_base:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
                else:
                    self.initial_param_last_layer.append(param.data.clone().to("cuda"))
        self.corrects = torch.zeros(4)
        self.totals = torch.zeros(4)
        self.worst_total = torch.tensor(0)
        self.worst_correct = torch.tensor(0)
        self.val_accuracy = 0
        self.test_accuracy = 0
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, labels, weights = batch
        preds = self.model(img)
        loss = self.loss_afr(preds, labels, weights, self.gamma, self.regularization, self.initial_param_last_layer)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, labels, backgrounds = batch

        preds = self.model(img)
        loss = self.loss_afr(preds, labels, torch.ones_like(labels), self.gamma, self.regularization, self.initial_param_last_layer)
        preds = torch.argmax(preds, dim=1)
        accuracy_2 = (preds == labels).float().mean()
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            if i == 0 and j == 0:
                self.totals[0] += 1
                if preds[idx] == i:
                    self.corrects[0] += 1
            elif i == 0 and j == 1:
                self.totals[1] += 1
                if preds[idx] == i:
                    self.corrects[1] += 1
            elif i == 1 and j == 0:
                self.totals[2] += 1
                if preds[idx] == i:
                    self.corrects[2] += 1
            else:
                self.totals[3] += 1
                if preds[idx] == i:
                    self.corrects[3] += 1

        result = [self.corrects[i] / self.totals[i] if self.totals[i] != 0 else 0 for i in range(len(self.totals))]
        minn = min(result)
        index = result.index(minn)
        self.worst_total = self.totals[index]
        self.worst_correct = self.corrects[index]
        self.val_accuracy = self.worst_correct.float() / self.worst_total.float() if self.worst_total.float() != 0 else torch.tensor(
            0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.corrects = torch.zeros(4)
        self.totals = torch.zeros(4)
        self.worst_total = torch.tensor(0)
        self.worst_correct = torch.tensor(0)
        self.log('val_accuracy_WGA', torch.tensor(self.val_accuracy), on_epoch=True, prog_bar=True)
        self.val_accuracy = 0

    def test_step(self, batch, batch_idx):

        img, labels, backgrounds = batch
        preds = self.model(img)
        loss = self.loss_afr(preds, labels, torch.ones_like(labels), self.gamma, self.regularization, self.initial_param_last_layer)
        preds = torch.argmax(preds, dim=1)
        accuracy_2 = (preds == labels).float().mean()
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            if i == 0 and j == 0:
                self.totals[0] += 1
                if preds[idx] == i:
                    self.corrects[0] += 1
            elif i == 0 and j == 1:
                self.totals[1] += 1
                if preds[idx] == i:
                    self.corrects[1] += 1
            elif i == 1 and j == 0:
                self.totals[2] += 1
                if preds[idx] == i:
                    self.corrects[2] += 1
            else:
                self.totals[3] += 1
                if preds[idx] == i:
                    self.corrects[3] += 1

        result = [self.corrects[i] / self.totals[i] if self.totals[i] != 0 else 0 for i in range(len(self.totals))]
        minn = min(result)
        index = result.index(minn)
        self.worst_total = self.totals[index]
        self.worst_correct = self.corrects[index]
        self.test_accuracy = self.worst_correct.float() / self.worst_total.float() if self.worst_total.float() != 0 else torch.tensor(
            0)


        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def on_test_epoch_end(self):
        self.corrects = torch.zeros(4)
        self.totals = torch.zeros(4)
        self.worst_total = torch.tensor(0)
        self.worst_correct = torch.tensor(0)
        self.log('test_WGA_accuracy', torch.tensor(self.test_accuracy), on_epoch=True, prog_bar=True)
        self.val_accuracy = 0
    def configure_optimizers(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.second_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=10)

        return [self.optimizer], [scheduler]

    def loss_afr(self, preds, labels, weights, gamma, regularization, initial_value):

        pre_loss = nn.CrossEntropyLoss(reduction="none")
        weighted_afr_loss = torch.sum(weights * pre_loss(preds, labels))
        fc_params = []
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                fc_params.append(param.data.clone().to("cuda"))

        difference_params = [params1 - params2 for params1, params2 in
                             zip(initial_value, fc_params)]

        squared_l2_norm = sum(torch.norm(param.view(-1), p=2) ** 2 for param in difference_params)
        return torch.tensor(regularization * squared_l2_norm, requires_grad=True) + weighted_afr_loss


