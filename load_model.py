import torch
from diffusers import StableDiffusionPipeline
import time

time1 = time.time()

device = "cuda:0"

path = "Lykon/DreamShaper"
safety_pipe = StableDiffusionPipeline.from_pretrained(
path, torch_dtype=torch.bfloat16
).to(device)

print("Time to load (should be very small amount of seconds):", time.time() - time1)