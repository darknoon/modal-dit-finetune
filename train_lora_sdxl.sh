# make sure to adjust the column name to match the dataset you are using
#DATASET="your/dataset"
#OUTPUT_NAME="your-output-name"

python train_dreambooth_lora_sdxl_modal.py \
 --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
 --dataset_name=$DATASET \
 --mixed_precision="bf16" \
 --instance_prompt="" \
 --resolution=1024 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=4 \
 --caption_column="caption" \
 --learning_rate=1e-4 \
 --report_to="wandb" \
 --lr_scheduler="constant" \
 --lr_warmup_steps=10 \
 --max_train_steps=500 \
 --validation_prompt="A very simple black and white icon of a man walking a dog on a white background, with nice rounded curves and a minimalist design. The man has a rounded body and head, holding a leash that is attached to the dog's collar. The dog should is rounded and walking alongside the man. Both figures are easily recognizable with clear, distinct, and rounded lines. The style is modern, clean, and easily understandable at a glance." \
 --validation_epochs=1 \
 --seed="0" \
 --push_to_hub \
 --hub_model_id="symbols-sdxl-lora"


