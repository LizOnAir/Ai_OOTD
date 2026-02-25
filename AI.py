from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load CLIP model (CPU-friendly)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("Jeans.webp").convert("RGB")

# Clothing labels you want to detect
types = [
    "shirt",
    "long sleeve sweater",
    "hoodie",
    "jacket",
    "jeans",
    "trousers",
    "skirt",
    "dress"
]

colors = [
    "Black",
    "White",
    "Red",
    "Green",
    "Blue",
    "Purple",
    "Orange",
    "Yellow"
]

# Process inputs
inputs = processor(
    text=types + colors,
    images=image,
    return_tensors="pt",
    padding=True
)

# Run model
with torch.no_grad():
    outputs = model(**inputs)

# Get similarity scores
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Print types results
for type, prob in zip(types, probs[0]):
    print(f"{type:25s}: {prob.item():.2%}")

# Print colours results
for color, prob in zip(colors, probs[0]):
    print(f"{color:25s}: {prob.item():.2%}")

# Best match
best_idx = probs[0].argmax().item()
print("\nDetected clothing:", colors[best_idx], types[best_idx])