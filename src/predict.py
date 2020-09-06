import datetime
from pathlib import Path

import joblib
import pandas as pd
import torch
from catalyst.dl import SupervisedRunner
from torch.nn.functional import softmax

from src.data.data_loaders import get_test_data_loader
from src.model import get_model
from src.utils.utils import fix_seed


ARTIFACTS_DIR = Path("/kaggle/input/landmarks-model-plus")
CHECKPOINT_PATH = ARTIFACTS_DIR / "best.pth"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.jl"
NUM_CLASSES_PATH = ARTIFACTS_DIR / "num_classes.jl"
DATA_PATH = Path("/kaggle/input/landmark-recognition-2020")
BATCH_SIZE = 64
NUM_WORKERS = 6
MODEL_NAME = "efficientnet-b0"
DEVICE = torch.device('cuda:0')

SEED = 17
fix_seed(SEED)
start_time = datetime.datetime.now()

print('Loading model...')
num_classes = joblib.load(filename=NUM_CLASSES_PATH)
model = get_model(model_name=MODEL_NAME, n_classes=num_classes, pretrained=False)

print('Loading test dataloader')
submission_df = pd.read_csv(DATA_PATH / "sample_submission.csv")
test_loader = get_test_data_loader(submission_df,
                                   image_dir=DATA_PATH,
                                   batch_size=BATCH_SIZE,
                                   num_workers=NUM_WORKERS)

print("Running prediction")
runner = SupervisedRunner(device=DEVICE)
output = torch.cat([softmax(pred[runner.output_key], dim=1) for pred in
                    runner.predict_loader(loader=test_loader, model=model, resume=str(CHECKPOINT_PATH))])
probs, preds = torch.max(output, dim=1)

label_enc = joblib.load(LABEL_ENCODER_PATH)
labels = label_enc.inverse_transform(preds.cpu().numpy())

print("Saving predictions to the file `submission.csv`")
submission_df['landmarks'] = [f'{label} {score}' for label, score in zip(labels, probs.cpu().numpy())]
submission_df.to_csv('submission.csv', index=False)

end_time = datetime.datetime.now()
print('Duration: {}'.format(end_time - start_time))
