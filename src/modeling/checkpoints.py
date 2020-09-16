from collections import OrderedDict

import torch


def load_model_state_from_checkpoint(checkpoint_path, net=None, prefix='model.'):
    """
    Load model weights from Pytorch Lightning trainer checkpoint.

    Parameters
    ----------
    net: nn.Module
        Instance of the PyTorch model to load weights to
    checkpoint_path: str
        Path to PL Trainer checkpoint 
    prefix: str
        Prefix used in LightningModule for model attribute.
    Returns
    -------
        nn.Module
    """

    if net is None:
        # create new instance model
        raise NotImplementedError

    trainer_checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    # chop off the prefix from LitModule, e.g. `self.model = model`
    model_checkpoint = OrderedDict(((key[len(prefix):] if key.startswith(prefix) else key, value)
                                    for key, value in trainer_checkpoint['state_dict'].items()))

    net.load_state_dict(model_checkpoint)
    return net
