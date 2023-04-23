from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.nn as nn


def get_model(model_type, pretrained=False):
    if pretrained:
        print('using pretrained model')
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if model_type == 'resnet18' else resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(in_features=512, out_features=5) if model_type =='resnet18'else nn.Linear(in_features=2048, out_features=5)
        return model
    else:
        print('not using pretrained model')
        model = resnet18() if model_type == 'resnet18' else resnet50()
        model.fc = nn.Linear(in_features=512, out_features=5) if model_type =='resnet18'else nn.Linear(in_features=2048, out_features=5)
        return model
