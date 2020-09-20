from collections import OrderedDict
from typing import Tuple

import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
import cirtorch
from cirtorch.layers.normalization import L2N
from .metric_learning import ArcMarginProduct


class LandmarkModel(nn.Module):
    def __init__(self,
                 n_classes: int,
                 model_name: str = 'resnet50',
                 pretrained: bool = True,
                 pooling_name: str = 'adaptive',  # 'GeM',
                 args_pooling: dict = {},
                 normalize: bool = True,
                 use_fc: bool = False,
                 fc_dim: int = 512,
                 dropout: float = 0.0,
                 loss_module: str = 'softmax'
                 ):
        super().__init__()
        self.backbone, final_in_features = self.get_backbone(model_name, pretrained, num_classes=fc_dim)
        if pooling_name in ('AdaptiveAvgPool2d', 'adaptive'):
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pooling_name in ('MAC', 'SPoC', 'GeM', 'GeMmp', 'RMAC', 'Rpool'):
            self.pooling = getattr(cirtorch.pooling, pooling_name)(**args_pooling)
        elif pooling_name is None:
            self.pooling = None
        else:
            raise ValueError("Incorrect pooling name")
        if normalize:
            self.norm = L2N()
        else:
            self.norm = None

        self.use_fc = use_fc
        if use_fc:
            self.final_block = nn.Sequential(OrderedDict([
                ('bn1', nn.BatchNorm1d(final_in_features)),
                ('dropout', nn.Dropout(p=dropout)),
                ('fc2', nn.Linear(final_in_features, fc_dim)),
                ('bn2', nn.BatchNorm1d(fc_dim))
            ]))
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.final_block.fc2.weight)
        nn.init.constant_(self.final_block.fc2.bias, 0)
        nn.init.constant_(self.final_block.bn1.weight, 1)
        nn.init.constant_(self.final_block.bn1.bias, 0)
        nn.init.constant_(self.final_block.bn2.weight, 1)
        nn.init.constant_(self.final_block.bn2.bias, 0)

    def forward(self, x):
        feature = self.extract_feat(x)
        logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        # feature extraction -> pooling -> norm
        x = self.backbone(x)
        if self.pooling is not None:
            x = self.pooling(x)
        if self.norm is not None:
            x = self.norm(x)
        x = x.view(batch_size, -1)

        if self.use_fc:
            x = self.final_block(x)

        return x

    @staticmethod
    def get_backbone(model_name, pretrained, num_classes=1000) -> Tuple[nn.Module, int]:
        if model_name.startswith("efficientnet"):
            if pretrained:
                model = EfficientNet.from_pretrained(
                    model_name=model_name,
                    num_classes=num_classes)
            else:
                model = EfficientNet.from_name(
                    model_name=model_name,
                    num_classes=num_classes)
            return model, model._fc.out_features
            # fc_in_features = model._fc.in_features
            # if not remove_head:
            #     return model, model._fc.out_features
            # else:
            #     # can't chop off head because forward method has reference to those layers.
            #     # Requires more sophisticated approach
            #     exclude_layers = ('_bn1', '_avg_pooling', '_dropout', '_fc', '_swish')
            #     model = nn.Sequential(
            #         OrderedDict((name, m) for name, m in model.named_children() if name not in exclude_layers)
            #         )
            #     return model, fc_in_features

        elif model_name in pretrainedmodels.model_names:
            num_classes = 1000  # due to models limitations
            model = getattr(pretrainedmodels, model_name)(num_classes=num_classes,
                                                          pretrained='imagenet' if pretrained else None)
            fc_in_features = model.last_linear.in_features
            # remove avgpool, fc, last_linear
            exclude_layers = ('avgpool', 'fc', 'last_linear')
            model = nn.Sequential(
                OrderedDict((name, m) for name, m in model.named_children() if name not in exclude_layers))
            return model, fc_in_features
        else:
            raise NotImplementedError("No other models available so far")
