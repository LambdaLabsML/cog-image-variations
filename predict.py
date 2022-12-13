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
            torch_dtype=torch.float16,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        input_image: Path = Input(description="Input Image", default=None),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 2, 3, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=3.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")


        inp = Image.open(input_image).convert("RGB")
        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            input_image=[inp] * num_outputs,
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