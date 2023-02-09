import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load a pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one
num_classes = 2  # Number of classes in your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Put the model in evaluation mode
model.eval()

# Perform a forward pass on an example image
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
outputs = model(x)

# Outputs is a dictionary containing the model's predictions
print(outputs)