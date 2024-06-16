This is a simple wrapper around train_dreambooth_sd3.py from diffusers to run on the modal platform, where it's easy to grab an A100 or H100 to finetune your model.

I have already pushed my dataset up to huggingface with a "caption" column, but you could also use a modal volume to sync your data if you don't want to use huggingface.

