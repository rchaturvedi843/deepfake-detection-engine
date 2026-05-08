import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path

# =============================
# SETTINGS
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
model_path = PROJECT_ROOT / "deepfake_strong_model.pth"
device = torch.device("cpu")

# =============================
# LOAD MODEL
# =============================
model = models.resnet18(weights=None)
state = torch.load(model_path, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

if any(k.startswith("fc.1.") for k in state.keys()):
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )
else:
    model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(state, strict=True)
model = model.to(device)
model.eval()

# =============================
# TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# PREDICT FUNCTION
# =============================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ["Fake", "Real"]
    return classes[predicted.item()], confidence.item() * 100


# =============================
# RUN FROM TERMINAL
# =============================
if __name__ == "__main__":
    image_path = input("Enter image path: ")
    label, confidence = predict_image(image_path)
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    
