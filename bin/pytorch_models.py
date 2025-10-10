# https://docs.pytorch.org/vision/main/models.html
import torchvision.models as models

model_dict = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'vgg11_bn': models.vgg11_bn,
    'vgg13_bn': models.vgg13_bn,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
}

weight_class_dict = {
    'vgg13_bn': models.VGG13_BN_Weights,
    'vgg16_bn': models.VGG16_BN_Weights,
}