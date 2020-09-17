import datetime
import logging
from pprint import pformat
import os

import joblib
import torch
from torch.utils.data import DataLoader

from src.config.config_template import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.modeling.index import extract_features, build_index
from src.datamodule import load_train_dataframe, CollateBatchFn, LandmarksImageDataset
from src.modeling.checkpoints import load_model_state_from_checkpoint
from src.modeling.model import LandmarkModel
from src.utils import fix_seed


CHECKPOINT_DIR = os.path.expanduser('~/kaggle/landmark_recognition_2020/logs/Landmarks/4a293h13/checkpoints')
CHECKPOINT_NAME = 'epoch=1.ckpt'
IVF_INDEX = False
DEVICE = torch.device('cuda:0')
BATCH_SIZE = 512
NUM_WORKERS = 20
LOAD_VECTORS_FROM_CHECKPOINT = True

if __name__ == "__main__":
    SEED = 17
    fix_seed(SEED)
    logger = logging.getLogger(__name__)
    start_time = datetime.datetime.now()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    if not LOAD_VECTORS_FROM_CHECKPOINT:
        model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True,
                                                       json_path=os.path.join(CHECKPOINT_DIR, 'config.json'))

        train_orig_df, _ = load_train_dataframe(training_args.data_train,
                                                min_class_samples=training_args.min_class_samples)

        logger.info(f'Loading persisted LabelEncoder and num_classes from checkpoints {CHECKPOINT_DIR}')
        label_enc = joblib.load(filename=training_args.checkpoints_path / training_args.label_encoder_filename)
        num_classes = len(label_enc.classes_)
        assert train_orig_df.landmark_id.nunique() == num_classes, "Num classes should be the same in DF and loaded obj"

        model = LandmarkModel(model_name=model_args.model_name,
                              n_classes=num_classes,
                              loss_module=model_args.loss_module,
                              pooling_name=model_args.pooling_name,
                              args_pooling=model_args.args_pooling,
                              use_fc=model_args.use_fc,
                              fc_dim=model_args.fc_dim,
                              dropout=model_args.dropout,
                              s=model_args.s,
                              margin=model_args.margin,
                              ls_eps=model_args.ls_eps,
                              theta_zero=model_args.theta_zero
                              )
        logger.info("Model params:")
        logger.info(pformat(model_args))
        logger.info('Loading model weights from checkpoint')
        model = load_model_state_from_checkpoint(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME), model)

        logger.info('Creating train dataloader')
        collate_fn = CollateBatchFn()
        train_dataset = LandmarksImageDataset(train_orig_df, image_dir=training_args.data_path, mode="train")

        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  sampler=None,
                                  num_workers=NUM_WORKERS,
                                  collate_fn=collate_fn
                                  )
        logger.info('Extracting features from train images using pretrained model')
        img_mapping, img_vectors = extract_features(model, train_loader, device=DEVICE, save_to_disk=True)
    else:
        logger.info('Loading vectors from checkpoint')
        img_mapping = joblib.load(os.path.join(CHECKPOINT_DIR, 'meta_vectors_train.pkl'))
        img_vectors = joblib.load(os.path.join(CHECKPOINT_DIR, 'vectors_train.pkl'))

    logger.info('Building faiss index on extracted features (vectors)')
    index = build_index(img_vectors, k=5000, path_to_save=os.path.join(CHECKPOINT_DIR, 'faiss_flat.index'))

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))
