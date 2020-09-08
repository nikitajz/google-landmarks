import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class LandmarksBaseModule(pl.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.hparams = hparams
        # model
        self.model = model

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # (log keyword is optional)
        return {'loss': loss, 'log': {'train_loss': loss}}