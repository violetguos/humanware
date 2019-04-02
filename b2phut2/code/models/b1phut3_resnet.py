"""Define models and generator functions which receives params as parameter, then add model to available models"""
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class B1phut3Resnet(ResNet):
    """ResNet implementation based on the best team of block 1 implementation"""

    def __init__(self, params):
        """
        :param params: parameters from the config file
        """
        assert params.FEATURES_EXTRACTION.BLOCK in ['basic', 'bottleneck'], "The desired block does not exist"
        if params.FEATURES_EXTRACTION.BLOCK == 'basic':
            block = BasicBlock
        else:
            block = Bottleneck
        super().__init__(block, params.FEATURES_EXTRACTION.LAYERS, params.FEATURES_EXTRACTION.OUTPUT_SIZE)
