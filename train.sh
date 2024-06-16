# make sure to adjust the column name to match the dataset you are using
$DATASET="darknoon/sf-symbols-captioned"
$OUTPUT_NAME="symbols-sd3-v1"

python train_dreambooth_sd3_modal.py \
 --pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers  \
 --dataset_name="darknoon/sf-symbols-captioned" \
 --mixed_precision="bf16" \
 --instance_prompt="  " \
 --resolution=1024 \
 --train_batch_size=2 \
 --gradient_accumulation_steps=4 \
 --caption_column="caption" \
 --learning_rate=3e-5 \
 --report_to="wandb" \
 --lr_scheduler="constant" \
 --lr_warmup_steps=10 \
 --max_train_steps=500 \
 --validation_prompt="A very simple black and white icon of a man walking a dog, with nice rounded curves and a minimalist design. The man has a rounded body and head, holding a leash attached to the dog's collar. The dog is rounded and walking in front of the man. Both figures are easily recognizable with clear, distinct, and rounded lines. The style is modern, clean, and easily understandable at a glance." \
 --validation_epochs=1 \
 --seed="0" \
 --push_to_hub \
 --hub_model_id=$OUTPUT_NAME


