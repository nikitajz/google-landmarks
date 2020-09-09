from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LandmarksBaseModule(pl.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.hparams = hparams
        # model
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        y = batch["targets"]
        y_hat = self(batch["features"])
        loss = self.loss_fn(y_hat, y)

        # (log keyword is optional)
        return {'loss': loss, 'log': {'train_loss': loss}}
