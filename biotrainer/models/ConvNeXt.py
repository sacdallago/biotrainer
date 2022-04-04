from convnext import ConvNeXt as ConvNeXt_base


class ConvNeXt(ConvNeXt_base):
    def __init__(self, n_classes: int, n_features: int):
        super(ConvNeXt, self).__init__(
            depths=[3, 3, 9, 3], dims=[256, 128, 64, 32], # dims=[96, 192, 384, 768],
            in_chans=n_features, num_classes=n_classes)
