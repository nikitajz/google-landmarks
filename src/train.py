import datetime
import logging

import joblib
import pytorch_lightning as pl
from catalyst.contrib.utils import split_dataframe_train_test  # TODO: replace with non-catalyst method

from src.config.config_template import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.datamodule import load_train_dataframe, LandmarksDataModule
from src.lit_module import LandmarksBaseModule
from src.model import get_model
from src.utils import fix_seed


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
    train_df, valid_df = split_dataframe_train_test(train_orig_df, test_size=0.2, random_state=SEED,
                                                    stratify=train_orig_df.landmark_id)

    training_args.checkpoints_path.mkdir(exist_ok=True, parents=True)
    joblib.dump(label_enc, filename=training_args.checkpoints_path / training_args.label_encoder_filename)
    logger.info(f'Persisted LabelEncoder to {training_args.label_encoder_filename}')

    num_classes = train_df.landmark_id.nunique()
    joblib.dump(num_classes, filename=training_args.checkpoints_path / training_args.num_classes_filename)

    model = get_model(model_name='efficientnet-b0', n_classes=num_classes)
    lit_module = LandmarksBaseModule(hparams=training_args.__dict__, model=model)

    # init data
    dm = LandmarksDataModule(train_df, valid_df,
                             image_dir=training_args.data_path,
                             batch_size=training_args.batch_size,
                             num_workers=training_args.num_workers,
                             use_weighted_sampler=training_args.use_weighted_sampler,
                             replacement=training_args.replacement
                             )

    # train
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(lit_module, datamodule=dm)

    print(trainer.ckpt_path)

    # # test
    # trainer.test(datamodule=dm)

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))
