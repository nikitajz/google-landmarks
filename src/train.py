import datetime
import logging
from pathlib import Path

import joblib
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pprint import pformat
from catalyst.contrib.utils import split_dataframe_train_test  # TODO: replace with non-catalyst method

from src.config.config_template import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.datamodule import load_train_dataframe, LandmarksDataModule
from src.lit_module import LandmarksPLBaseModule
from src.modeling.model import LandmarkModel
from src.utils import fix_seed


def get_logger_path(pl_logger):
    """Using Pytorch-lightning logger attribute, find the directory where checkpoints to be saved for this run"""
    checkpoints_path = Path(pl_logger.save_dir) / pl_logger.name / pl_logger.version / 'checkpoints'
    return checkpoints_path


if __name__ == '__main__':
    SEED = 17
    fix_seed(SEED)
    logger = logging.getLogger(__name__)
    start_time = datetime.datetime.now()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)

    train_orig_df, label_enc = load_train_dataframe(training_args.data_train,
                                                    min_class_samples=training_args.min_class_samples)
    train_df, valid_df = split_dataframe_train_test(train_orig_df, test_size=training_args.test_size, random_state=SEED,
                                                    stratify=train_orig_df.landmark_id)
    num_classes = train_df.landmark_id.nunique()

    model = LandmarkModel(model_name='resnet50',  # 'efficientnet-b0',
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

    logger.info('Initializing the model')
    lit_module = LandmarksPLBaseModule(hparams=training_args.__dict__, model=model, loss=model_args.loss_module)

    # init data
    dm = LandmarksDataModule(train_df, valid_df,
                             image_dir=training_args.data_path,
                             batch_size=training_args.batch_size,
                             num_workers=training_args.num_workers,
                             use_weighted_sampler=training_args.use_weighted_sampler,
                             replacement=training_args.replacement
                             )

    # train
    dt_str = datetime.datetime.now().strftime("%y%m%d_%H-%M")
    wandb_logger = WandbLogger(name=f'Baseline_GeM_ArcFace_{dt_str}',
                               save_dir='logs/',
                               project='landmarks')

    trainer = pl.Trainer(gpus=training_args.gpus,
                         logger=wandb_logger,
                         max_epochs=training_args.n_epochs,
                         val_check_interval=training_args.val_check_interval,
                         progress_bar_refresh_rate=100,
                         resume_from_checkpoint=training_args.resume_checkpoint
                         )
    trainer.fit(lit_module, datamodule=dm)

    try:
        training_args.checkpoints_path = get_logger_path(trainer.logger)
        logger.info(f'Saving checkpoints to the current directory: {training_args.checkpoints_path}')
    except:
        logger.warning(f'Unable to get current checkpoints directory, using default one: '
                       f'{training_args.checkpoints_path}')

    training_args.checkpoints_path.mkdir(exist_ok=True, parents=True)
    joblib.dump(label_enc, filename=training_args.checkpoints_path / training_args.label_encoder_filename)
    logger.info(f'Persisted LabelEncoder to {training_args.label_encoder_filename}')
    joblib.dump(num_classes, filename=training_args.checkpoints_path / training_args.num_classes_filename)

    # # test
    # trainer.test(datamodule=dm)

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))
