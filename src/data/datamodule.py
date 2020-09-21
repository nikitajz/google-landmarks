import logging
from typing import Optional

from pandas import DataFrame

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.data.dataset import PathType, LandmarksImageDataset, CollateBatchFn


def class_imbalance_sampler(labels, num_samples=None, replacement=False):
    labels = torch.LongTensor(labels.to_numpy())
    class_count = torch.bincount(labels).to(dtype=torch.float)
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=replacement)
    return sampler


# TODO: make `valid_df` optional
class LandmarksDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_df: DataFrame, valid_df: DataFrame,
                 image_dir: PathType,
                 image_size: int,
                 batch_size: int,
                 num_workers: int = 4,
                 use_weighted_sampler: bool = True,
                 limit_samples_to_draw: bool = True,
                 replacement: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.image_dir = image_dir
        self.image_size = image_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.hparams = dict()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampler = use_weighted_sampler
        self.limit_samples_to_draw = limit_samples_to_draw
        self.replacement = replacement

        if self.limit_samples_to_draw and self.use_weighted_sampler:
            n_uniq_classes = self.train_df.landmark_id.nunique()
            n_samples = int(n_uniq_classes * self.train_df.landmark_id.value_counts().mean())
        else:
            n_samples = len(self.train_df)

        self.sampler = None
        if self.use_weighted_sampler:
            self.logger.info(f'Using weighted sampler with total {n_samples} to draw. Replacement: {self.replacement}')
            self.sampler = class_imbalance_sampler(self.train_df.landmark_id, num_samples=n_samples,
                                                   replacement=self.replacement)

        self.collate_fn = None
        self.train_dataset = None
        self.valid_dataset = None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.collate_fn = CollateBatchFn()
        self.train_dataset = LandmarksImageDataset(self.train_df, image_dir=self.image_dir, image_size=self.image_size,
                                                   mode="train")
        self.valid_dataset = LandmarksImageDataset(self.valid_df, image_dir=self.image_dir, image_size=self.image_size,
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
