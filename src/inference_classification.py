import datetime
import logging
import os
from pprint import pformat

import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from src.config.config_template import TrainingArgs, ModelArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.dataset import get_test_data_loader
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
    CHECKPOINT_DIR = os.path.expanduser('~/kaggle/landmark_recognition_2020/logs/Landmarks/2uglfbx6/checkpoints')
    SUBMISSION_PATH = os.path.join(CHECKPOINT_DIR, 'submission.csv')
    DEVICE = 'cuda:1'
    BATCH_SIZE = 8
    NUM_WORKERS = 1
else:
    raise ValueError("Unknown environment exception")

CHECKPOINT_NAME = 'epoch-4.ckpt'
NORMALIZE_VECTORS = True
LOAD_VECTORS_FROM_CHECKPOINT = False
TOPK = 10
THRESHOLD = 0.45  # empty string for images below the score
DEVICE = torch.device(DEVICE)
SEED = 17


# postprocessing
def postprocessing_omit_low_scores(row):
    if row['scores'] > THRESHOLD:
        landmark_str = str(row['labels']) + ' ' + str(row['scores'])
    else:
        landmark_str = ''
    return landmark_str


def main():
    fix_seed(SEED)
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
                          pretrained=False,
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
                                       image_size=training_args.image_size,
                                       crop_size=training_args.crop_size,
                                       batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # predict on test dataset
    model.eval()
    model.to(DEVICE)
    activation = nn.Softmax(dim=1)
    confs_list = []
    preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            y_hat = model(batch['features'].to(DEVICE))
            # y_hat = activation(y_hat)

            confs_batch, preds_batch = torch.topk(y_hat, TOPK)
            confs_batch = activation(confs_batch)
            confs_list.append(confs_batch)
            preds_list.append(preds_batch)
        confs = torch.cat(confs_list).cpu().numpy()
        preds = torch.cat(preds_list).cpu().numpy()

    pred_labels = label_enc.inverse_transform(preds[:, 0])  # decode only first element
    confidence_score = confs[:, 0]
    # pred_labels = [label_enc.inverse_transform(pred) for pred in preds]
    #
    # pred_labels = [label[0] for label in pred_labels]
    # confidence_score = [score[0] for score in confs]

    # save submit file
    logger.info('Saving the predictions to submission.csv')
    submission_df['labels'] = pred_labels
    submission_df['scores'] = confidence_score
    submission_df['landmarks'] = submission_df.apply(postprocessing_omit_low_scores, axis=1)
    del submission_df['labels']
    del submission_df['scores']
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))


if __name__ == "__main__":
    main()
