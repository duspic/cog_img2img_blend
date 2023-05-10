import os
from typing import List

import utils

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image
from cog import BasePredictor, Input, Path

MODEL_ID = "SG161222/Realistic_Vision_V2.0"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=False,
        ).to("cuda")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, poorly drawn, lowres, bad quality, worst quality, unrealistic, overexposed, underexposed",
        ),
        image: Path = Input(
            description="Image generated with controlnet, overlaid with the original object",
        ),
        noback_image: Path = Input(
            description="The original object image, with removed background",
        ),
        prompt_strength: float = Input(
            description="Prompt strength when providing the image. 1.0 corresponds to full destruction of information in init image",
            default=0.1,
        ),
        num_outputs: int = Input(
            description="Number of images to output. Higher number of outputs may OOM.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=1
        ),
        scheduler: str = Input(
            default="K_EULER_ANCESTRAL",
            choices=["DDIM", "K_EULER", "DPMSolverMultistep", "K_EULER_ANCESTRAL", "PNDM", "KLMS"],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        pipe = self.img2img_pipe
        extra_kwargs = {
            "image": Image.open(image).convert("RGB"),
            "strength": prompt_strength,
        }
        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            
            b_output_path = f"/tmp/out-{i}_blended.png"
            noback = Image.open(noback_image)
            blended = utils.blend(sample, noback)
            blended.save(b_output_path)
            output_paths.append(Path(b_output_path))

        return output_paths
        

def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
