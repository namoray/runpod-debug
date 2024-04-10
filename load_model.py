import torch
from diffusers import StableDiffusionPipeline
import time
import os

time1 = time.time()

# Just for testing, nothing helped
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(False)

device = "cuda:0"

path = "Lykon/DreamShaper"
safety_pipe = StableDiffusionPipeline.from_pretrained(
path, torch_dtype=torch.bfloat16
).to(device)

print("Time to load: ", time.time() - time1)
print("If this was your first time running, run it again please as the download tainted it, or:")
print("Look in your console logs. If you see more than ~2s per step to load the model, there is an issue")


print("I actually think this is a cpu issue, as watch, its still really slow even without the device")
path = "Lykon/DreamShaper"
safety_pipe = StableDiffusionPipeline.from_pretrained(
path, torch_dtype=torch.bfloat16
)

print("Basically the model should be loading first into RAM, and then moved onto the GPU after. The problem is in the loading into RAM phase.")