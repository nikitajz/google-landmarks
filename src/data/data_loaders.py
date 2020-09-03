import logging
from pathlib import Path
from typing import Union, Callable, Tuple, Optional

import pandas as pd
from pandas import DataFrame
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger()
PathType = Union[str, Path]


class LandmarksImageDataset(Dataset):
    def __init__(self, dataframe: DataFrame, image_dir: PathType, mode: str, transform: Callable = None):
        assert mode in ("train", "valid", "test")
        self.df = dataframe
        self.mode = mode
        image_subdir = "train" if self.mode == "valid" else self.mode
        self.image_dir = Path(image_dir) / image_subdir
        self.image_path = ImagePath(self.image_dir)
        self.transform = transform if transform is not None else self._get_default_transform(self.mode)

    def __getitem__(self, idx: int):
        image_id = self.df.iat[idx, self.df.columns.get_loc("id")]
        img = Image.open(self.image_path.get_image_path(image_id))
        outputs = (self.transform(img),)
        if self.mode == "train" or self.mode == "valid":
            label = self.df.iat[idx, self.df.columns.get_loc("landmark_id")]
            outputs = outputs + (label,)
        return outputs

    def __len__(self) -> int:
        return self.df.shape[0]

    @staticmethod
    def _get_default_transform(mode: str):
        base_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]

        convert_n_normalize = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        augmentations = [
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(224),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                        scale=(0.8, 1.2), shear=15,
                                        resample=Image.BILINEAR)
            ]),
        ]

        if mode == 'train':
            all_transforms = [base_transforms, augmentations, convert_n_normalize]
        else:
            all_transforms = [base_transforms, convert_n_normalize]

        return transforms.Compose([item for sublist in all_transforms for item in sublist])


class ImagePath:
    def __init__(self, image_dir: PathType):
        self.dir = Path(image_dir)

    def get_image_path(self, image_id: str) -> PathType:
        image_path = self.dir / image_id[0] / image_id[1] / image_id[2] / image_id
        return image_path.with_suffix(".jpg")

    def exists(self, img_id):
        img_path = self.get_image_path(img_id)
        return img_path.exists()


def load_train_dataframe(csv_path: PathType, min_class_samples: Optional[int] = 10,
                         validate_path: bool = False) -> Tuple[DataFrame, LabelEncoder]:
    """
    Load dataframe from the file `train.csv` containing `image_id` and `landmark_id` (class_id) mapping.

    Parameters
    ----------
    csv_path:
        Path to the `train.csv` file.
    min_class_samples
        Minimum number of samples per class. Remove classes with number of samples below threshold.
    validate_path
        Whether to check if file corresponding to image_id actually exists (time-consuming).

    Returns
    -------
    (df: DataFrame, label_encoder: LabelEncoder)
    """
    df = pd.read_csv(csv_path)
    logger.debug(f'Initial number of samples: {df.shape[0]}')
    if validate_path:
        logger.debug('Validating images exist')
        image_dir = csv_path.parent/'train'
        image_path = ImagePath(image_dir)
        df = df.loc[df.id.apply(image_path.exists)]
        logger.debug(f'Sample after filtering non-existing images: {df.shape[0]}')
    if min_class_samples is not None and min_class_samples > 0:
        grouped_df = df.groupby("landmark_id").size()
        logger.debug(f'Initial number of classes: {grouped_df.shape[0]}')
        selected_classes = grouped_df[grouped_df >= min_class_samples].index.to_numpy()
        num_classes = len(selected_classes)
        logger.debug(f'Selected number of classes: {num_classes}')
        df = df.loc[df.landmark_id.isin(selected_classes)]
    else:
        num_classes = df.landmark_id.nunique()
        logger.info(f'Number of classes: {num_classes}')

    label_encoder = LabelEncoder()
    label_encoder.fit(df.landmark_id.values)
    assert len(label_encoder.classes_) == num_classes
    df.landmark_id = label_encoder.transform(df.landmark_id)

    return df, label_encoder


class CollateBatch:
    def __init__(self, features_name: str = "features", target_name: str = "targets"):
        self.features_name = features_name
        self.target_name = target_name

    def __call__(self, batch):
        features = torch.stack([row[0] for row in batch], dim=0)

        batch_dict = {
            self.features_name: features
        }

        if len(batch[0]) == 2:
            targets = torch.tensor([row[1] for row in batch])
            batch_dict[self.target_name] = targets
        return batch_dict


def get_data_loaders(train_df: DataFrame, valid_df: DataFrame,
                     image_dir: PathType,
                     batch_size: int,
                     num_workers: int = 4,
                     shuffle: bool = False,
                     sampler=None):

    train_dataset = LandmarksImageDataset(train_df,
                                          image_dir=image_dir,
                                          mode="train"
                                          )

    valid_dataset = LandmarksImageDataset(valid_df,
                                          image_dir=image_dir,
                                          mode="valid"
                                          )

    collate_fn = CollateBatch()

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              sampler=sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn
                              )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              sampler=sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn
                              )

    loaders = {"train": train_loader,
               "valid": valid_loader}
    return loaders
