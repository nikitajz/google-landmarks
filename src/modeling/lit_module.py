import logging
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from src.metrics import GAPMetric
from src.modeling.focal_loss import FocalLoss
from src.modeling.metric_learning import ArcFaceLoss

logger = logging.getLogger(__name__)


class LandmarksPLBaseModule(pl.LightningModule):
    def __init__(self, hparams, model, loss='ce'):
        super().__init__()
        self.hparams = hparams
        # model
        self.model = model
        if loss in ("ce", 'cross-entropy'):
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss in ('arcface', 'cosface', 'adacos'):
            self.loss_fn = ArcFaceLoss(s=hparams.get('s', 30.0),
                                       m=hparams.get('m', 0.50))
        elif loss == 'focal_loss':
            self.loss_fn = FocalLoss(gamma=2)
        else:
            raise NotImplementedError(f"Unknown loss: {loss}")
        self.train_mode = 'train'
        self.val_mode = 'val'
        self.gap = {self.train_mode: GAPMetric(),
                    self.val_mode: GAPMetric()
                    }

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        if self.hparams.freeze_backbone:
            logger.info('Freezing backbone')
            params = [param for name, param in self.named_parameters() if 'backbone' not in name]
        else:
            params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=self.hparams.lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def _compute_step(self, batch, batch_idx, mode):
        assert mode in ('train', 'val')
        labels = batch["targets"]
        features = batch["features"]
        logits = self.model(features)
        if self.loss_fn is not None:
            loss_value = self.loss_fn(logits, labels)
        else:
            loss_value = logits
        gap_value = None
        if mode == self.val_mode:
            gap_value = self.gap[mode].forward(logits, labels)
        y_hat = torch.argmax(logits, dim=1)
        return loss_value, gap_value, y_hat

    def training_step(self, batch, batch_idx):
        loss, gap_batch, preds = self._compute_step(batch, batch_idx, mode=self.train_mode)
        logs_train = {'train_loss': loss}
        return {'loss': loss, 'log': logs_train}

    def on_train_epoch_end(self) -> None:
        self.gap[self.train_mode].reset_stats()

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        loss, gap_batch, preds = self._compute_step(batch, batch_idx, mode=self.val_mode)
        outputs = {'val_loss': loss,
                   'targets': batch['targets'],
                   'preds': preds}
        return outputs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        gap_epoch = self.gap[self.val_mode].compute_final()
        acc_metric = Accuracy()
        val_acc = acc_metric.forward(pred=torch.stack([batch['preds'] for batch in outputs]),
                                     target=torch.stack([batch['targets'] for batch in outputs]))
        val_logs = {'val_loss': avg_loss, 'val_gap': gap_epoch, 'val_acc': val_acc}
        # reset metrics every epoch
        self.gap[self.val_mode].reset_stats()

        return {
            'val_loss': avg_loss,
            'val_acc': val_acc,
            'log': val_logs,
            'progress_bar': {'val_acc': val_acc, 'gap': gap_epoch}
        }
