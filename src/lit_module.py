from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LandmarksBaseModule(pl.LightningModule):
    def __init__(self, hparams, model, loss='ce'):
        super().__init__()
        self.hparams = hparams
        # model
        self.model = model
        if loss in ("ce", 'cross-entropy', 'softmax'):
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss in ('arcface', 'cosface', 'adacos'):
            self.loss_fn = None
        else:
            raise NotImplementedError("Unknown loss")

    def forward(self, x, label):
        return self.model.forward(x, label)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        labels = batch["targets"]
        features = batch["features"]
        y_hat = self.model(features, labels)
        if self.loss_fn is not None:
            loss = self.loss_fn(y_hat, labels)
        else:
            loss = y_hat

        # (log keyword is optional)
        return {'loss': loss, 'log': {'train_loss': loss}}
