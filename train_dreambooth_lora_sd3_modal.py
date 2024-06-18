import os

from args_dreambooth_lora import parse_args

import modal
app = modal.App()

def cache_model():
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        use_auth_token=os.environ["HF_TOKEN"],
    )
    print(f"Cached model pipeline:\n\n{pipe}")
    return

image = (
    modal.Image
    .debian_slim()
    # just enough to run cache_model
    .pip_install("torch", "diffusers[torch]", "transformers", "sentencepiece", "protobuf", "ftfy")
    .run_function(cache_model, secrets=[modal.Secret.from_name("huggingface-write")])
    # don't have to re-cache model when adjusting dependencies
    .pip_install_from_requirements("requirements.txt")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-write"), modal.Secret.from_name("wandb-darknoon")],
    mounts=[modal.Mount.from_local_python_packages("train_dreambooth_lora_sd3")],
    gpu="H100",
    timeout=4*60*60, # 4 hour max
)
def train(args):
    from train_dreambooth_lora_sdxl import main
    import train_dreambooth_lora_sdxl

    setattr(train_dreambooth_lora_sdxl, "args", args)
    main(args)
    # after training, copy results to modal volume?

if __name__ == "__main__":
    with app.run():
        args = parse_args()
        train.remote(args)

