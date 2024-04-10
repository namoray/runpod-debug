import torch
from diffusers import StableDiffusionPipeline
import time

time1 = time.time()

device = "cuda:0"

path = "Lykon/DreamShaper"
safety_pipe = StableDiffusionPipeline.from_pretrained(
path, torch_dtype=torch.bfloat16
).to(device)

print("Time to load (should be very small amount of seconds after the initial download):", time.time() - time1)
print("Look in your console logs. If you see more than ~2s per step to load the model, there is an issue")