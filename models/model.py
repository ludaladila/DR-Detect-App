# models/model.py
import torch
import torch.nn as nn
import torchvision.models as models

# Deep Learning Model
class DeepLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        # Modify the classifier for 3 classes
        self.classifier[6] = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
