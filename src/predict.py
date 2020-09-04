import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import torch
from catalyst.dl import SupervisedRunner
from torch.nn.functional import softmax

from src.config.config_base import ModelArgs, TrainingArgs
from src.config.hf_argparser import load_or_parse_args
from src.data.data_loaders import get_test_data_loader
from src.model import get_model


def run_prediction():
    logger.info("Running prediction")
    model_args, training_args = load_or_parse_args((ModelArgs, TrainingArgs), verbose=True)

    submission_df = pd.read_csv(training_args.data_path / "sample_submission.csv")

    logger.info('Loading test dataloader')
    test_loader = get_test_data_loader(submission_df,
                                       image_dir=training_args.data_path,
                                       batch_size=training_args.batch_size * 2,
                                       num_workers=training_args.num_workers)

    is_kernel = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    artifacts_dir = Path("/kaggle/input/landmarks-model-plus") if is_kernel else Path(training_args.log_dir)
    device = torch.device(training_args.device)

    logger.info('Loading model...')
    checkpoint_path = artifacts_dir / "best.pth"
    num_classes = joblib.load(filename=artifacts_dir/training_args.num_classes_filename)
    model = get_model(model_name=model_args.model_name, n_classes=num_classes, pretrained=False)

    runner = SupervisedRunner(device=device)
    output = torch.cat([softmax(pred[runner.output_key], dim=1) for pred in
                        runner.predict_loader(loader=test_loader, model=model, resume=str(checkpoint_path))])
    probs, preds = torch.max(output, dim=1)

    predictions, scores = preds.cpu().numpy(), probs.cpu().numpy()
    del preds, probs

    label_enc = joblib.load(artifacts_dir/training_args.label_encoder_filename)
    labels = label_enc.inverse_transform(predictions)

    submission_df['landmarks'] = [f'{label} {score}' for label, score in zip(labels, scores)]
    submission_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    run_prediction()
