import logging
from typing import Optional

from pandas import DataFrame

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.config.config_template import TrainingArgs
from src.data.dataset import PathType, LandmarksImageDataset, CollateBatchFn
from src.data.samplers import get_imbalanced_sampler, DistributedSamplerWrapper


# TODO: make `valid_df` optional
class LandmarksDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: DataFrame, valid_df: DataFrame,
                 hparams: TrainingArgs,
                 image_dir: PathType,
                 batch_size: int,
                 num_workers: int = 4,
                 use_weighted_sampler: bool = True,
                 **kwargs
                 ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.image_dir = image_dir
        self.image_size = hparams.image_size
        self.crop_size = hparams.crop_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.hparams = hparams
        self.tpu_cores = hparams.tpu_cores
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler
        self.limit_samples_to_draw = hparams.limit_samples_to_draw
        self.replacement = hparams.replacement

        if self.limit_samples_to_draw and self.use_weighted_sampler:
            n_uniq_classes = self.train_df.landmark_id.nunique()
            # n_samples = int(n_uniq_classes * self.train_df.landmark_id.value_counts().mean())
            n_samples = min(self.limit_samples_to_draw, len(self.train_df))
        else:
            n_samples = len(self.train_df)

        self.sampler = None
        if self.use_weighted_sampler:
            self.logger.info(f'Using weighted sampler with total {n_samples} to draw. Replacement: {self.replacement}')
            imbalanced_sampler = get_imbalanced_sampler(self.train_df.landmark_id, num_samples=n_samples,
                                                        replacement=self.replacement)
            if self.tpu_cores is not None and self.tpu_cores > 1:
                import torch_xla.core.xla_model as xm
                distributed_sampler = DistributedSamplerWrapper(
                    imbalanced_sampler,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=self.hparams.shuffle
                )
                self.sampler = distributed_sampler
            else:
                self.sampler = imbalanced_sampler

        self.collate_fn = None
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.collate_fn = CollateBatchFn()
        self.train_dataset = LandmarksImageDataset(self.train_df, image_dir=self.image_dir,
                                                   image_size=self.image_size, crop_size=self.crop_size,
                                                   mode="train")
        self.valid_dataset = LandmarksImageDataset(self.valid_df, image_dir=self.image_dir,
                                                   image_size=self.image_size, crop_size=self.crop_size,
                                                   mode="valid")

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,  # due to using sampler
                                  sampler=self.sampler,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn,
                                  drop_last=True
                                  )
        return train_loader

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  sampler=None,
                                  num_workers=self.num_workers,
                                  collate_fn=self.collate_fn,
                                  drop_last=True
                                  )
        return valid_loader

    # def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
