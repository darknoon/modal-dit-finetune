import io
import os

import modal

app = modal.App()

@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "sentencepiece", "protobuf", "ftfy"),
    secrets=[modal.Secret.from_name("huggingface-modal-read")],
    gpu="A100",
)
async def run_stable_diffusion(prompt: str):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        use_auth_token=os.environ["HF_TOKEN"],
    ).to("cuda")

    image = pipe(prompt, num_inference_steps=10).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    return img_bytes


@app.local_entrypoint()
def main():
    img_bytes = run_stable_diffusion.remote("Wu-Tang Clan climbing Mount Everest")
    with open("/tmp/output.png", "wb") as f:
        f.write(img_bytes)
