
from torchvision import models
import torch
# Replace 'your_model_path.pt' with the actual path to your .pt file
model_path = 'best.pt'

# Load the model
model = torch.load(model_path, map_location=torch.device('cpu'))

# Print the model summary
print(model)
