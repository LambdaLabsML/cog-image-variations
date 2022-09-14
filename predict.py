import os
from typing import List

import torch
from PIL import Image, ImageFilter
from cog import BasePredictor, Input, Path


from lambda_diffusers import StableDiffusionImageEmbedPipeline

CACHE_DIR = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionImageEmbedPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            cache_dir=CACHE_DIR,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        input_image: Path = Input(description="Input Image", default=None),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 512, 768, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width == height == 1024:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        inp = Image.open(input_image).convert("RGB")
        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            input_image=[inp] * num_outputs,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )
        for i in range(len(output["nsfw_content_detected"])):
            if output["nsfw_content_detected"][i]:
                output["sample"][i] = output["sample"][i].filter(ImageFilter.GaussianBlur(50))

        output_paths = []
        for i, sample in enumerate(output["sample"]):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths