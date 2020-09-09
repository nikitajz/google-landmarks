from efficientnet_pytorch import EfficientNet


def get_model(model_name: str, n_classes: int, pretrained=True):
    if model_name.startswith("efficientnet"):
        if pretrained:
            pretrained_model = EfficientNet.from_pretrained(
                model_name=model_name,
                num_classes=n_classes)
            return pretrained_model
        else:
            empty_model = EfficientNet.from_name(
                model_name=model_name,
                num_classes=n_classes)
            return empty_model
    else:
        raise NotImplementedError("No other models available so far")
