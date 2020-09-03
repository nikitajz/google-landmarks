import datetime
import logging

import torch
import torch.nn as nn
from catalyst.contrib.dl.callbacks import WandbLogger
from catalyst.contrib.utils import split_dataframe_train_test
from catalyst.dl import SupervisedRunner
from efficientnet_pytorch import EfficientNet

from src.config.config_base import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.data_loaders import load_train_dataframe, get_data_loaders
from src.utils.metrics import GAPMetricCallback
from src.utils.utils import fix_seed


def get_model(model_name: str, n_classes: int):
    if model_name.startswith("efficientnet"):
        pretrained_model = EfficientNet.from_pretrained(
            model_name=model_name,
            num_classes=n_classes)
        return pretrained_model
    else:
        raise NotImplementedError("No other models available so far")


if __name__ == '__main__':
    SEED = 17
    fix_seed(SEED)
    logger = logging.getLogger(__name__)

    dt_str = datetime.datetime.now().strftime("%y%m%d_%H-%M")

    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    train_orig_df, label_enc = load_train_dataframe(training_args.data_train)
    train_df, valid_df = split_dataframe_train_test(train_orig_df, test_size=0.2, random_state=SEED,
                                                    stratify=train_orig_df.landmark_id)

    loaders = get_data_loaders(train_df, valid_df,
                               image_dir=training_args.data_path,
                               batch_size=training_args.batch_size,
                               num_workers=training_args.num_workers,
                               shuffle=training_args.shuffle)

    num_classes = train_df.landmark_id.nunique()
    model = get_model(model_name=model_args.model_name, n_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

    callbacks = [
        # AccuracyCallback(num_classes=num_classes),
        # AUCCallback(num_classes=num_classes, input_key="targets"),
        GAPMetricCallback(),
        WandbLogger(project="Landmarks", name=f'tryout_{dt_str}', log_on_batch_end=True)
    ]

    runner = SupervisedRunner(device=training_args.gpus)

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=training_args.log_dir,
        num_epochs=training_args.n_epochs,
        main_metric="loss",
        minimize_metric=True,
        fp16=False,
        verbose=True
    )
