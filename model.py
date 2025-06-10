import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
