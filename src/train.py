import datetime
import logging
from pprint import pformat

import joblib
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from catalyst.contrib.utils import split_dataframe_train_test  # TODO: replace with non-catalyst method

from src.config.config_template import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.datamodule import LandmarksDataModule
from src.data.dataset import load_train_dataframe
from src.modeling.lit_module import LandmarksPLBaseModule
from src.modeling.model import LandmarkModel
from src.utils import fix_seed, get_checkpoints_path, save_config_checkpoint


def main():
    logger = logging.getLogger(__name__)
    start_time = datetime.datetime.now()
    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)
    train_orig_df, label_enc = load_train_dataframe(training_args.data_train,
                                                    min_class_samples=training_args.min_class_samples)
    # assert training_args.test_size % training_args.batch_size == 0, "Test size should be multiple of batch size"
    train_df, valid_df = split_dataframe_train_test(train_orig_df, test_size=training_args.test_size,
                                                    stratify=train_orig_df.landmark_id, random_state=SEED)
    num_classes = train_df.landmark_id.nunique() if training_args.min_class_samples is None else len(label_enc.classes_)
    model = LandmarkModel(model_name=model_args.model_name,
                          n_classes=num_classes,
                          loss_module=model_args.loss_module,
                          pooling_name=model_args.pooling_name,
                          args_pooling=model_args.args_pooling,
                          use_fc=model_args.use_fc,
                          fc_dim=model_args.fc_dim,
                          dropout=model_args.dropout
                          )
    logger.info("Model params:")
    logger.info(pformat(model_args))
    logger.info('Initializing the model')
    lit_module = LandmarksPLBaseModule(hparams=training_args.__dict__,
                                       model=model,
                                       loss=model_args.loss_module)
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

    checkpoint_callback = ModelCheckpoint(filepath=training_args.ckpt_path,
                                          monitor='val_acc',
                                          save_top_k=3)
    trainer = pl.Trainer(gpus=training_args.gpus,
                         logger=wandb_logger,
                         max_epochs=training_args.n_epochs,
                         val_check_interval=training_args.val_check_interval,
                         checkpoint_callback=checkpoint_callback,
                         progress_bar_refresh_rate=100,
                         resume_from_checkpoint=training_args.resume_checkpoint,
                         # fast_dev_run=True,
                         # limit_train_batches=5,
                         # limit_val_batches=5
                         )
    trainer.fit(lit_module, datamodule=dm)
    try:
        training_args.checkpoints_path = get_checkpoints_path(trainer.logger)
        logger.info(f'Saving checkpoints to the current directory: {training_args.checkpoints_path}')
    except:
        logger.warning(f'Unable to get current checkpoints directory, using default one: '
                       f'{training_args.checkpoints_path}')
    training_args.checkpoints_path.mkdir(exist_ok=True, parents=True)
    joblib.dump(label_enc, filename=training_args.checkpoints_path / training_args.label_encoder_filename)
    logger.info(f'Persisted LabelEncoder to {training_args.label_encoder_filename}')
    save_config_checkpoint(training_args.checkpoints_path)
    # # test
    # trainer.test(datamodule=dm)
    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    SEED = 17
    fix_seed(SEED)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main()
