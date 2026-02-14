from diffusers import DiffusionPipeline
import torch

# ==============================
# Generative AI Fashion Recommender
# ==============================

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# High-quality enhancement prompts
positive_magic = {
    "en": ", Ultra HD, 4K, professional fashion photography, detailed fabric texture, cinematic lighting.",
}

negative_prompt = "blurry, low quality, distorted face, extra fingers, bad anatomy"

# ==============================
# User Inputs (Fashion Profile)
# ==============================

gender = input("Enter gender (male/female): ")
occasion = input("Enter occasion (casual, office, party, wedding, gym): ")
season = input("Enter season (summer, winter, spring, autumn): ")
style = input("Preferred style (minimal, streetwear, luxury, sporty, traditional): ")

# ==============================
# Prompt Engineering
# ==============================

prompt = f"""
A full-body fashion model wearing a stylish {season} outfit for a {occasion}.
The outfit is designed for a {gender} with a {style} fashion style.
Highly detailed clothing, modern trends, fashion magazine shoot,
clean background, professional lighting
"""

# ==============================
# Image Size Options
# ==============================

aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
}

width, height = aspect_ratios["9:16"]  # Vertical for fashion

# ==============================
# Generate Fashion Recommendation
# ==============================

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

# Save Output
image.save("fashion_recommendation.png")

print("✅ Fashion recommendation generated and saved as 'fashion_recommendation.png'")
