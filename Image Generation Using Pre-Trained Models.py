!pip install transformers diffusers torch flax jax jaxlib                                                                                                                                                                   from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

from diffusers import StableDiffusionPipeline
import torch

# Load the Stable Diffusion model and tokenizer
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Text prompt for image generation
prompt = "A futuristic cityscape at sunset with flying cars and neon lights."

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")                                                                                                                                                                                                                                      from PIL import Image
from IPython.display import display

# Assuming 'image' is a PIL Image object
display(image)