import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from src.metrics import GAPMetric
from src.modeling.focal_loss import FocalLoss


class LandmarksPLBaseModule(pl.LightningModule):
    def __init__(self, hparams, model, loss='ce'):
        super().__init__()
        self.hparams = hparams
        # model
        self.model = model
        if loss in ("ce", 'cross-entropy', 'logloss', 'arcface', 'cosface', 'adacos'):
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == 'focal_loss':
            self.loss_fn = FocalLoss(gamma=2)
        else:
            raise NotImplementedError(f"Unknown loss: {loss}")
        self.train_mode = 'train'
        self.val_mode = 'val'
        self.gap = {self.train_mode: GAPMetric(),
                    self.val_mode: GAPMetric()
                    }

    def forward(self, x, label):
        return self.model.forward(x, label)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                   threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0)
        return [optimizer], [scheduler]

    def _compute_step(self, batch, batch_idx, mode):
        assert mode in ('train', 'val')
        labels = batch["targets"]
        features = batch["features"]
        y_hat = self.model(features, labels)
        if self.loss_fn is not None:
            loss_value = self.loss_fn(y_hat, labels)
        else:
            loss_value = y_hat
        gap_value = self.gap[mode].forward(y_hat, labels)
        return loss_value, gap_value

    def training_step(self, batch, batch_idx):
        loss, gap_batch = self._compute_step(batch, batch_idx, mode=self.train_mode)
        # (log keyword is optional)
        logs_train = {'train_loss': loss, 'gap': gap_batch}
        return {'loss': loss, 'log': logs_train, 'progress_bar': logs_train}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss, gap_batch = self._compute_step(batch, batch_idx, mode=self.val_mode)
        logs_val = {'val_loss': loss, 'gap': gap_batch}
        return {'val_loss': loss,
                'log': logs_val,
                'progress_bar': logs_val}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        gap_epoch = self.gap[self.val_mode].compute_final()
        val_logs = {'val_loss': avg_loss, 'gap': gap_epoch}
        progress_bar_metrics = val_logs
        return {
            'val_loss': avg_loss,
            'log': val_logs,
            'progress_bar': progress_bar_metrics
        }
