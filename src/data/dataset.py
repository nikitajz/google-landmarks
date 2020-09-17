import logging
from pathlib import Path
from typing import Union, Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = logging.getLogger()
PathType = Union[str, Path]


class LandmarksImageDataset(Dataset):
    def __init__(self, dataframe: DataFrame, image_dir: PathType, mode: str, transform: Callable = None,
                 get_img_id=False, features_name='features', target_name='targets', img_id_name='image_ids'):
        assert mode in ("train", "valid", "test")
        self.df = dataframe
        self.mode = mode
        image_subdir = "train" if self.mode == "valid" else self.mode
        self.image_dir = Path(image_dir) / image_subdir
        self.image_path = ImagePath(self.image_dir)
        self.transform = transform if transform is not None else self._get_default_transform(self.mode)
        self.get_img_id = get_img_id
        self.features_name = features_name
        self.target_name = target_name
        self.img_id_name = img_id_name

    def __getitem__(self, idx: int):
        image_id = self.df.iat[idx, self.df.columns.get_loc("id")]
        img = Image.open(self.image_path.get_image_path(image_id)).convert('RGB')
        outputs = dict()
        outputs[self.features_name] = self.transform(img)
        if self.get_img_id:
            outputs[self.img_id_name] = image_id
        if self.mode in ("train", "valid"):
            label = self.df.iat[idx, self.df.columns.get_loc("landmark_id")]
            outputs[self.target_name] = label
        return outputs

    def __len__(self) -> int:
        return self.df.shape[0]

    @staticmethod
    def _get_default_transform(mode: str):
        base_transforms = [
            transforms.Resize(256)
        ]

        center_crop = [
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
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                        scale=(0.8, 1.2), shear=15,
                                        resample=Image.BILINEAR)
            ]),
            transforms.RandomCrop(224)
        ]

        if mode == 'train':
            all_transforms = [base_transforms, augmentations, convert_n_normalize]
        else:
            all_transforms = [base_transforms, center_crop, convert_n_normalize]

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
                         validate_image_path: bool = False) -> Tuple[DataFrame, LabelEncoder]:
    """
    Load dataframe from the file `train.csv` containing `image_id` and `landmark_id` (class_id) mapping.
    Parameters
    ----------
    csv_path:
        Path to the `train.csv` file.
    min_class_samples
        Minimum number of samples per class. Remove classes with number of samples below threshold.
    validate_image_path
        Whether to check if file corresponding to image_id actually exists (time-consuming).
    Returns
    -------
    (df: DataFrame, label_encoder: LabelEncoder)
    """
    df = pd.read_csv(csv_path)
    logger.debug(f'Initial number of samples: {df.shape[0]}')
    if validate_image_path:
        logger.debug('Validating images exist')
        image_dir = csv_path.parent / 'train'
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


class CollateBatchFn:
    def __init__(self, features_name: str = "features", target_name: str = "targets", image_id_name='image_ids'):
        self.features_name = features_name
        self.target_name = target_name
        self.image_id_name = image_id_name

    def __call__(self, batch):
        features = torch.stack([row[self.features_name] for row in batch], dim=0)

        batch_dict = {self.features_name: features}
        keys = batch[0].keys()

        if self.target_name in keys:
            targets = torch.tensor([row[self.target_name] for row in batch])
            batch_dict[self.target_name] = targets
        if self.image_id_name in keys:
            image_ids = [row[self.image_id_name] for row in batch]
            batch_dict[self.image_id_name] = image_ids
        return batch_dict


def get_test_data_loader(sub_df: DataFrame,
                         image_dir: PathType,
                         batch_size: int,
                         num_workers: int = 4,
                         get_img_id=True,
                         ):
    test_dataset = LandmarksImageDataset(sub_df,
                                         image_dir=image_dir,
                                         mode="test",
                                         get_img_id=get_img_id
                                         )

    collate_fn = CollateBatchFn()
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return test_loader
