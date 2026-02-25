from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load CLIP model (CPU-friendly)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("Jeans.webp").convert("RGB")

# Clothing labels you want to detect
labels = [
    "a white shirt",
    "a black shirt",
    "a long sleeve sweater",
    "a hoodie",
    "a jacket",
    "jeans",
    "black trousers",
    "skirt",
    "dress"
]

# Process inputs
inputs = processor(
    text=labels,
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

# Print results
for label, prob in zip(labels, probs[0]):
    print(f"{label:25s}: {prob.item():.2%}")

# Best match
best_idx = probs[0].argmax().item()
print("\nDetected clothing:", labels[best_idx])
print("done")