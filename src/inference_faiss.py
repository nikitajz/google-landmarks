import datetime
import logging
import os
from pprint import pformat

import joblib
import faiss
import numpy as np
from scipy.stats import mode
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src.config.config_template import TrainingArgs, ModelArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.dataset import get_test_data_loader, TARGET_NAME
from src.modeling.checkpoints import load_model_state_from_checkpoint
from src.modeling.features_index import extract_features
from src.modeling.model import LandmarkModel
from src.utils import fix_seed

KAGGLE_KERNEL_RUN_TYPE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost')
if KAGGLE_KERNEL_RUN_TYPE in ('Batch', 'Interactive'):
    CODE_DIR = '/kaggle/input/landmarks-2020-lightning/'
    CHECKPOINT_DIR = os.path.join(CODE_DIR, 'checkpoints')
    SUBMISSION_PATH = os.path.join(CHECKPOINT_DIR, 'submission.csv')

    import sys

    sys.path.append(CODE_DIR)
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    NUM_WORKERS = 4
elif KAGGLE_KERNEL_RUN_TYPE == 'Localhost':
    CHECKPOINT_DIR = os.path.expanduser('~/kaggle/landmark_recognition_2020/logs/Landmarks/4a293h13/checkpoints')
    SUBMISSION_PATH = 'submission.csv'
    DEVICE = 'cuda:0'
    BATCH_SIZE = 512
    NUM_WORKERS = 20
else:
    raise ValueError("Unknown environment exception")

CHECKPOINT_NAME = 'epoch_1.ckpt'
NORMALIZE_VECTORS = True
LOAD_VECTORS_FROM_CHECKPOINT = False
TOPK = 5
SEED = 17
DEVICE = torch.device(DEVICE)
fix_seed(SEED)


def main():
    start_time = datetime.datetime.now()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)

    # load config file
    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True,
                                                   json_path=os.path.join(CHECKPOINT_DIR, 'config.json'))

    # load label_encoder
    logger.info(f'Loading persisted LabelEncoder and num_classes from checkpoints {CHECKPOINT_DIR}')
    label_enc: LabelEncoder = joblib.load(filename=os.path.join(CHECKPOINT_DIR, training_args.label_encoder_filename))
    num_classes = len(label_enc.classes_)

    # create model and load weights from checkpoint
    model = LandmarkModel(model_name=model_args.model_name,
                          n_classes=num_classes,
                          loss_module=model_args.loss_module,
                          pooling_name=model_args.pooling_name,
                          args_pooling=model_args.args_pooling,
                          use_fc=model_args.use_fc,
                          fc_dim=model_args.fc_dim,
                          dropout=model_args.dropout,
                          )
    logger.info("Model params:")
    logger.info(pformat(model_args))
    logger.info('Loading model weights from checkpoint')
    model = load_model_state_from_checkpoint(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME), model)

    # create test dataloader
    submission_df = pd.read_csv(training_args.data_path / 'sample_submission.csv')
    test_loader = get_test_data_loader(submission_df, image_dir=training_args.data_path,
                                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # load index
    index = faiss.read_index(os.path.join(CHECKPOINT_DIR, 'flat.index'))

    # extract feature vectors for test images
    if not LOAD_VECTORS_FROM_CHECKPOINT:
        img_mapping_test, img_vectors_test = extract_features(model, test_loader, mode='test', device=DEVICE,
                                                              normalize=NORMALIZE_VECTORS,
                                                              dir_to_save=CHECKPOINT_DIR)
    else:
        logger.info('Loading vectors from checkpoint')
        img_mapping_test = joblib.load(os.path.join(CHECKPOINT_DIR, 'meta_vectors_test.pkl'))
        img_vectors_test = joblib.load(os.path.join(CHECKPOINT_DIR, 'vectors_test.pkl'))

    logger.info('Loading train vectors mapping')
    train_vec_mapping = joblib.load(os.path.join(CHECKPOINT_DIR, 'meta_vectors_train.pkl'))

    # train_vec_image_ids = train_vec_mapping['image_ids']
    train_vec_targets = train_vec_mapping[TARGET_NAME]

    # predict kNN for each test image (topk = 3)
    logger.info('Searching for nearest neighbours')
    # this predicts nearest train vector id which needs to be converted in to the label
    pred_dist, pred_vec_id = index.search(img_vectors_test, TOPK)

    logger.info('Extracting label encoded target class_id')
    preds_vec = np.vectorize(lambda x: train_vec_targets[x])(pred_vec_id)

    logger.info('Picking up most common labels for each vector')
    pred_mode, pred_cnt = mode(preds_vec, axis=1)
    # threshold_dist = 0.007
    # pred_mode_final = np.where(pred_dist >= threshold_dist, preds_vec, np.nan)

    # inverse_transform predicted labels by label encoder
    logger.info('Label encoder inverse transform labels')
    pred_labels = label_enc.inverse_transform(pred_mode[:, 0])

    # rerank (optional)

    # save submit file
    logger.info('Saving the predictions to submission.csv')
    submission_df['labels'] = pred_labels
    submission_df['cnt'] = pred_cnt / pred_cnt.max()
    submission_df['landmarks'] = submission_df.apply(lambda x: str(x['labels']) + ' ' + str(x['cnt']), axis=1)
    del submission_df['labels']
    del submission_df['cnt']
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
