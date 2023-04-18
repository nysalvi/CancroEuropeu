import io
import json

import torchvision.transforms as transforms
from PIL import Image
import torch

from .model_initialize import ModelInitialize

imagenet_class_index = json.load(open('models/classes.json'))
ModelInitialize.initialize()
model = torch.load('models/model.pth')
model.eval()
threshold = 0.53

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
                        transforms.RandomResizedCrop((224, 224)),
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output_with_threshold = (tensor > threshold).float()
    outputs = model.forward(output_with_threshold)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]