import datetime
import gc
import logging
import operator
import os
from pprint import pformat

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import spatial
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.config.config_template import TrainingArgs, ModelArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.dataset import get_test_data_loader, load_train_dataframe, CollateBatchFn, LandmarksImageDataset
from src.modeling.checkpoints import load_model_state_from_checkpoint
from src.modeling.model import LandmarkModel
from src.utils import fix_seed

KAGGLE_KERNEL_RUN_TYPE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost')
if KAGGLE_KERNEL_RUN_TYPE in ('Batch', 'Interactive'):
    CODE_DIR = '/kaggle/input/landmarks-2020-lightning/'
    CHECKPOINT_DIR = os.path.join(CODE_DIR, 'checkpoints')
    SUBMISSION_PATH = 'submission.csv'

    import sys

    sys.path.append(CODE_DIR)
    DEVICE = 'cuda:0'
    BATCH_SIZE = 128
    NUM_WORKERS = 4
elif KAGGLE_KERNEL_RUN_TYPE == 'Localhost':
    CHECKPOINT_DIR = os.path.expanduser('~/kaggle/landmark_recognition_2020/logs/Landmarks/8ad4twsl/checkpoints')
    SUBMISSION_PATH = os.path.join(CHECKPOINT_DIR, 'submission.csv')
    DEVICE = 'cpu'  # 'cuda:1'
    BATCH_SIZE = 256
    NUM_WORKERS = 1
else:
    raise ValueError("Unknown environment exception")

CONFIG_FILE = "config.json"
CHECKPOINT_NAME = 'epoch=14.ckpt'
NORMALIZE_VECTORS = True
EMBEDDING_SIZE = 512  # model_args.fc_dim
IMAGE_SIZE = None  # override value, TBD later from the config
CROP_SIZE = None  # override value, TBD later from the config
# LOAD_VECTORS_FROM_CHECKPOINT = False
NUM_TO_RERANK = 3  # initially 5
THRESHOLD = 0.01  # empty string for images below the score
SEED = 17
DEVICE = torch.device(DEVICE)
fix_seed(SEED)


# postprocessing
def postprocessing_omit_low_scores(row):
    if row['scores'] > THRESHOLD:
        landmark_str = str(row['labels']) + ' ' + str(row['scores'])
    else:
        landmark_str = ''
    return landmark_str


def generate_embeddings(model, loader):
    model.eval()
    model.to(DEVICE)
    num_images = len(loader.dataset)
    batch_size = loader.batch_size
    ids = num_images * [None]
    embeddings = np.empty((num_images, EMBEDDING_SIZE))

    with torch.no_grad():
        for i, batch in enumerate(loader):
            sample_size = len(batch['image_ids'])
            ids[i * batch_size:i * batch_size + sample_size] = batch['image_ids']
            features = batch['features'].to(DEVICE)
            embeddings[i * batch_size:i * batch_size + sample_size, :] = model.extract_feat(features).cpu().numpy()

    return ids, embeddings


def get_similarities(config_path, image_size=None, crop_size=None):
    # load config file
    m_args, tr_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True, json_path=config_path)
    image_size = image_size if image_size is not None else tr_args.image_size
    crop_size = crop_size if crop_size is not None else tr_args.crop_size

    logger.debug(f'Loading persisted LabelEncoder and num_classes from checkpoints {CHECKPOINT_DIR}')
    label_enc: LabelEncoder = joblib.load(filename=os.path.join(CHECKPOINT_DIR, tr_args.label_encoder_filename))
    num_classes = len(label_enc.classes_)

    train_df, _ = load_train_dataframe(tr_args.data_train, min_class_samples=None)
    sub_df = pd.read_csv(tr_args.data_path / 'sample_submission.csv')

    # create model and load weights from checkpoint
    model = LandmarkModel(model_name=m_args.model_name,
                          pretrained=False,
                          n_classes=num_classes,
                          loss_module=m_args.loss_module,
                          pooling_name=m_args.pooling_name,
                          args_pooling=m_args.args_pooling,
                          use_fc=m_args.use_fc,
                          fc_dim=m_args.fc_dim,
                          dropout=m_args.dropout,
                          )
    logger.info("Model params:")
    logger.info(pformat(m_args))
    logger.info('Loading model weights from checkpoint')
    model = load_model_state_from_checkpoint(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME), model)
    # create train dataloader
    train_dataset = LandmarksImageDataset(train_df, image_dir=tr_args.data_path, mode="train",
                                          get_img_id=True,
                                          # transform=transforms.ToTensor(),  # in case on rescaling required
                                          image_size=image_size, crop_size=crop_size
                                          )
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,  # due to using sampler
                              sampler=None,
                              num_workers=NUM_WORKERS,
                              collate_fn=CollateBatchFn(),
                              drop_last=False
                              )
    # create test dataloader
    test_loader = get_test_data_loader(sub_df, image_dir=tr_args.data_path,
                                       image_size=image_size,
                                       crop_size=crop_size,
                                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # generate embeddings
    train_ids, train_embeddings = generate_embeddings(model, train_loader)
    test_ids, test_embeddings = generate_embeddings(model, test_loader)

    train_ids_labels_and_scores = [None] * test_embeddings.shape[0]
    # Using (slow) for-loop, as distance matrix doesn't fit in memory
    for test_idx in range(test_embeddings.shape[0]):
        distances = spatial.distance.cdist(
            test_embeddings[np.newaxis, test_idx, :], train_embeddings, 'cosine')[0]
        # Get the indices of the closest images
        top_k = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]
        # Get the nearest ids and distances using the previous indices
        nearest = sorted([(train_ids[p], distances[p]) for p in top_k], key=lambda x: x[1])
        # Get the labels and score results
        train_ids_labels_and_scores[test_idx] = [(train_df[train_id], 1.0 - cosine_distance) for
                                                 train_id, cosine_distance in nearest]

    del test_embeddings
    del train_embeddings
    gc.collect()

    return test_ids, train_ids_labels_and_scores


def generate_predictions(test_ids, train_ids_labels_and_scores):
    targets = []
    scores = []

    # Iterate through each test id
    for test_index, test_id in enumerate(test_ids):
        aggregate_scores = {}
        # Iterate through the similar images with their corresponding score for the given test image
        for target, score in train_ids_labels_and_scores[test_index]:
            if target not in aggregate_scores:
                aggregate_scores[target] = 0
            aggregate_scores[target] += score
        # Get the best score
        target, score = max(aggregate_scores.items(), key=operator.itemgetter(1))
        targets.append(target)
        scores.append(score)

    sub_df = pd.DataFrame({'id': test_ids, 'target': targets, 'scores': scores})
    sub_df['landmarks'] = sub_df['target'].astype(str) + ' ' + sub_df['scores'].astype(str)
    sub_df[['id', 'landmarks']].to_csv('submission.csv', index=False)
    return sub_df


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    logger = logging.getLogger(__name__)

    test_indices, train_ids_labels_n_scores = get_similarities(config_path=os.path.join(CHECKPOINT_DIR, CONFIG_FILE),
                                                               image_size=IMAGE_SIZE, crop_size=CROP_SIZE)

    # generate and save submission file
    logger.info('Saving the predictions to submission.csv')
    final_submission_df = generate_predictions(test_indices, train_ids_labels_n_scores)

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))
