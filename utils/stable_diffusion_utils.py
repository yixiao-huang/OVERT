import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
import numpy as np


def run_safety_checker(model, image, device = 'cuda', dtype = torch.bfloat16):
    assert hasattr(model, "safety_checker"), "Model does not have a safety checker"
    safety_checker_input = model.feature_extractor(image, return_tensors="pt").to(device)
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image

    image, has_nsfw_concept = model.safety_checker(
        images=image_np, clip_input=safety_checker_input.pixel_values.to(dtype)
    )
    return image, has_nsfw_concept

def load_model(
    model_name: str = "stabilityai/stable-diffusion-3.5-large",
    safety_checker_name: str = "CompVis/stable-diffusion-safety-checker",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    prompt = "a photo of a cat holding a sign that says hello world"
    image = pipe(prompt = prompt, generator = torch.manual_seed(1)).images[0]
    image, has_nsfw_concept = run_safety_checker(pipe, image)
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", 
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=dtype,
    ).to(device)
    
    pipe.to(dtype)
    pipe.feature_extractor = CLIPImageProcessor.from_pretrained(safety_checker_name, dtype = dtype, device = device)
    pipe.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_checker_name).to(device)
    return pipe
