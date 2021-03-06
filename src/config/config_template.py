import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict


@dataclass
class ModelArgs:
    model_name: str = field(
        metadata={"help": "Pretrained model name, e.g. `efficientnet-b4`"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})
    pooling_name: str = field(
        default='GeM',
        metadata={"help": "Pooling layer. Available options are: 'MAC', 'SPoC', 'GeM', 'GeMmp', 'RMAC', 'Rpool'"})
    args_pooling: Dict = field(
        default_factory=dict, metadata={"help": "Pooling arguments"})
    normalize: bool = field(
        default=True, metadata={"help": "Whether to use L2N to normalize vectors (see model details)"})
    use_fc: bool = field(
        default=False, metadata={"help": "Whether to use FC block layer (see model details)"})
    fc_dim: int = field(
        default=512, metadata={"help": "FC layer dimension"})
    dropout: float = field(
        default=0.0, metadata={"help": "Dropout probability in FC layer"})
    loss_module: str = field(
        default='softmax',
        metadata={"help": "Loss module. Available options are: ('softmax', 'arcface', 'cosface', 'adacos')"})
    s: float = field(
        default=30.0, metadata={"help": "ArcFaceLoss option 's'"})
    margin: float = field(
        default=0.50, metadata={"help": "ArcFaceLoss option 'margin'"})

    def __post_init__(self):
        self.loss_module = self.loss_module.lower()


@dataclass
class TrainingArgs:
    freeze_backbone: bool = field(
        default=False, metadata={"help": "Freeze Roberta model and train only classifier"})
    log_dir: Optional[str] = field(
        default="logs",
        metadata={"help": "Path to save training artifacts (logs, checkpoints, etc)"})
    resume_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint to resume training from"}
    )
    checkpoints_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save model checkpoint (can include placeholders)"}
    )
    image_size: int = field(
        default=256, metadata={"help": "Image size, integer, converted to square image"}
    )
    crop_size: int = field(
        default=224, metadata={"help": "Crop size, integer, converted to square image"}
    )
    data_path: str = field(
        default="data/x256/", metadata={"help": "Folder where data located"})
    data_train: str = field(
        default="train.csv",
        metadata={"help": "Train filename to load csv file from into dataframe"})
    data_valid: Optional[str] = field(
        default=None,
        metadata={"help": "Validation filename"})
    data_test: str = field(
        default="test.csv",
        metadata={"help": "Test filename to load csv file from into dataframe"})
    test_size: int = field(
        default=0.1,
        metadata={"help": "Filter out classes with fewer samples"})
    min_class_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Filter out classes with fewer samples"})
    use_weighted_sampler: bool = field(
        default=False, metadata={"help": "Use weighted sampler"})
    limit_samples_to_draw: Optional[int] = field(
        default=None, metadata={"help": "Use weighted sampler"})
    replacement: bool = field(
        default=False, metadata={"help": "Whether to use replacement in weighted sampler for train data dataloader"})
    shuffle: bool = field(
        default=False, metadata={"help": "Shuffle train data"})
    num_workers: int = field(
        default=1, metadata={"help": "How many workers to use for dataloader"})
    seed: int = field(
        default=42, metadata={"help": "Random number"})
    gpus: Optional[str] = field(
        default=None, metadata={"help": "Device to train model on. Int for number of gpus, " +
                                        "str to select specific one or List[str] to select few specific gpus"})
    tpu_cores: Optional[int] = field(
        default=None, metadata={"help": "Number of TPU cores"})
    n_epochs: int = field(
        default=2, metadata={"help": "Number of epochs to train"})
    accumulate_grad_batches: int = field(
        default=1, metadata={"help": "Steps interval to accumulate gradient."})
    batch_size: int = field(
        default=16, metadata={"help": "Batch size"})
    optimizer: str = field(
        default='Adam', metadata={"help": "Optimizer, available options are: 'Adam', 'SGD'."})
    lr: float = field(
        default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    momentum: float = field(
        default=0.0, metadata={"help": "Momentum for SGD if we apply some."})
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    warmup_steps: int = field(
        default=7, metadata={"help": "Warm up steps for optimizer."})
    scheduler: Optional[str] = field(
        default="cosine_annealing", metadata={"help": "Scheduler"})
    factor: float = field(
        default=0.1, metadata={"help": "Scheduler ReduceLROnPlateau factor."})
    patience: int = field(
        default=3, metadata={"help": "Scheduler ReduceLROnPlateau patience."})
    step_size: int = field(
        default=5, metadata={"help": "Scheduler StepLR size."})
    gamma: float = field(
        default=0.1, metadata={"help": "Scheduler gamma value."})
    gradient_clip_val: float = field(
        default=0, metadata={"help": "Clip gradient value. Set 0 to disable."})
    val_check_interval: Optional[int] = field(
        default=0.5, metadata={"help": "How often within one training epoch to check validation set." +
                                       "Set float for fraction or int for steps."})
    early_stop_callback: bool = field(
        default=True, metadata={"help": "Whether to use early stopping"})
    tensorboard_enable: bool = field(
        default=False, metadata={"help": "Whether to use tensorboard"})
    tb_log_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to save Tensorboard logs"})
    early_stopping_checkpoint_path: str = field(
        default=".early_stopping", metadata={"help": "Checkpoint path."})
    patience: int = field(
        default=2, metadata={"help": "Early stopping patience"})

    def __post_init__(self):
        not_kernel = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None
        self.data_path = Path(self.data_path) if not_kernel else Path("/kaggle/input/landmark-recognition-2020")
        self.data_train = self.data_path / self.data_train
        self.log_dir = Path(self.log_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)
        self.label_encoder_filename = "label_encoder.jl"
        self.num_classes_filename = "num_classes.jl"
