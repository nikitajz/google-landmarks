import datetime
import logging

import joblib
import torch
import torch.nn as nn
from catalyst.contrib.dl.callbacks import WandbLogger
from catalyst.contrib.utils import split_dataframe_train_test
from catalyst.dl import SupervisedRunner

from src.config.config_base import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.data_loaders import load_train_dataframe, get_data_loaders
from src.model import get_model
from src.utils.metrics import GAPMetricCallback
from src.utils.utils import fix_seed


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

    training_args.checkpoints_path.mkdir(exist_ok=True)
    joblib.dump(label_enc, filename=training_args.checkpoints_path / training_args.label_encoder_filename)
    logger.info(f'Persisted LabelEncoder to {training_args.label_encoder_filename}')

    loaders = get_data_loaders(train_df, valid_df,
                               image_dir=training_args.data_path,
                               batch_size=training_args.batch_size,
                               num_workers=training_args.num_workers,
                               use_weighted_sampler=training_args.use_weighted_sampler,
                               replacement=training_args.replacement)

    num_classes = train_df.landmark_id.nunique()
    joblib.dump(num_classes, filename=training_args.checkpoints_path / training_args.num_classes_filename)

    model = get_model(model_name=model_args.model_name, n_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6], gamma=0.1)

    dt_str = datetime.datetime.now().strftime("%y%m%d_%H-%M")

    callbacks = [
        GAPMetricCallback(prefix="gap"),
        WandbLogger(project="Landmarks", name=f'baseline_{dt_str}', log_on_batch_end=True,
                    config={**model_args.__dict__, **training_args.__dict__})
    ]

    runner = SupervisedRunner(device=training_args.device)

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=training_args.log_dir,
        num_epochs=training_args.n_epochs,
        main_metric="gap",
        minimize_metric=False,
        fp16=False,
        verbose=True
    )

    end_time = datetime.datetime.now()
    logger.info('Duration: {}'.format(end_time - start_time))
