import logging
from pathlib import Path

import joblib
import pandas as pd
import torch
from catalyst.dl import SupervisedRunner
from torch.nn.functional import softmax

from src.data.data_loaders import get_test_data_loader
from src.model import get_model


def run_prediction():
    artifacts_dir = Path("/kaggle/input/landmarks-model-plus")
    checkpoint_path = artifacts_dir / "best.pth"
    label_encoder_path = artifacts_dir / "label_encoder.jl"
    num_classes_path = artifacts_dir / "num_classes.jl"
    data_path = Path("/kaggle/input/landmark-recognition-2020")
    batch_size = 64
    num_workers = 6
    model_name = "efficientnet-b0"
    device = torch.device('cuda:0')

    logger.info("Running prediction")

    submission_df = pd.read_csv(data_path / "sample_submission.csv")

    logger.info('Loading test dataloader')
    test_loader = get_test_data_loader(submission_df,
                                       image_dir=data_path,
                                       batch_size=batch_size,
                                       num_workers=num_workers)

    logger.info('Loading model...')
    num_classes = joblib.load(filename=num_classes_path)
    model = get_model(model_name=model_name, n_classes=num_classes, pretrained=False)

    runner = SupervisedRunner(device=device)
    output = torch.cat([softmax(pred[runner.output_key], dim=1) for pred in
                        runner.predict_loader(loader=test_loader, model=model, resume=str(checkpoint_path))])
    probs, preds = torch.max(output, dim=1)

    label_enc = joblib.load(label_encoder_path)
    labels = label_enc.inverse_transform(preds.cpu().numpy())

    submission_df['landmarks'] = [f'{label} {score}' for label, score in zip(labels, probs.cpu().numpy())]
    submission_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    run_prediction()
