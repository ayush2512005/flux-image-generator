import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

pipe.to("cuda")

prompt = "A futuristic cyberpunk city at night, neon lights, ultra realistic"

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4
).images[0]

image.save("output.png")

print("Image generated and saved as output.png")