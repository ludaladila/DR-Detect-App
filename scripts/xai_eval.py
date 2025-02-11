from torchvision import models, transforms
import torch.nn as nn
from pytorch_grad_cam import GradCAM
import torch

def load_model(model_path: str):
    """Load a fine-tuned VGG model from model path"""
    vgg_model = models.vgg16(pretrained=False)
    vgg_model.classifier[6] = nn.Sequential(
        nn.Linear(vgg_model.classifier[6].in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,3)
    ) 
    vgg_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    vgg_model.eval()
    return vgg_model

def convert_to_gradcam(model):
    """Initialize a Grad-CAM explainer for the provided model"""
    target_layers = [model.features[-1]]
    return GradCAM(model=model, target_layers=target_layers)

def preprocess_image(image):
    """Apply image pre-processing for VGG-16 model"""
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3205, 0.2244, 0.1613], 
                           std=[0.2996, 0.2158, 0.1711])
    ])
    return transform(image)