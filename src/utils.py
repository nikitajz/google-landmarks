import json
import sys
from pathlib import Path
from typing import Union

import torch
import numpy as np
import random
import os


def fix_seed(seed=42, cuda=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_wandb_logger_checkpoints_path(pl_logger):
    """Using Pytorch-lightning logger attribute, find the directory where checkpoints to be saved for this run"""
    checkpoints_path = Path(pl_logger.save_dir) / pl_logger.name / pl_logger.version / 'checkpoints'
    return checkpoints_path


def save_config_checkpoint(checkpoint_path: Union[str, Path], json_path=None):
    if json_path is not None:
        pass
    elif sys.argv[1].endswith(".json"):
        json_path = sys.argv[1]
    else:
        raise FileNotFoundError("No config file provided for backup.")
    conf_js = json.loads(Path(json_path).read_text())
    json.dump(conf_js, open(checkpoint_path / 'config.json', 'w'))
