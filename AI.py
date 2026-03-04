from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Labels
types = [
    "a shirt",
    "a long sleeve sweater",
    "a hoodie",
    "a jacket",
    "jeans",
    "trousers",
    "a skirt",
    "a dress",
    "cargo pants",
]

colors = [
    "black clothing",
    "white clothing",
    "red clothing",
    "green clothing",
    "blue clothing",
    "purple clothing",
    "orange clothing",
    "yellow clothing",
]

# Precompute TEXT embeddings (FAST)
with torch.no_grad():
    type_inputs = processor(text=types, return_tensors="pt", padding=True).to(device)
    type_features = model.get_text_features(**type_inputs)
    type_features = type_features / type_features.norm(dim=-1, keepdim=True)

    color_inputs = processor(text=colors, return_tensors="pt", padding=True).to(device)
    color_features = model.get_text_features(**color_inputs)
    color_features = color_features / color_features.norm(dim=-1, keepdim=True)

# Load images
image_paths = [
    "clothes/blackpants.webp",
    "clothes/bluedress.jpeg",
    "clothes/cargoPants.webp",
    "clothes/jeans.webp",
    "clothes/sweater.jpeg"
]

images = [Image.open(p).convert("RGB") for p in image_paths]

# Process each image
for img, path in zip(images, image_paths):

    with torch.no_grad():
        image_inputs = processor(images=img, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Similarity (cosine similarity via dot product)
        type_similarity = image_features @ type_features.T
        color_similarity = image_features @ color_features.T

        type_probs = type_similarity.softmax(dim=-1)
        color_probs = color_similarity.softmax(dim=-1)

        best_type = types[type_probs.argmax().item()]
        best_color = colors[color_probs.argmax().item()]

    print(f"\nImage: {path}")
    print("Detected Type:", best_type)
    print("Detected Color:", best_color)