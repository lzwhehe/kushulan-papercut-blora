#!/bin/bash
# Edit-LoRA on Qwen-Image-Edit-2511 (reported runs): bf16, 1024 px, rank 32, lr 1e-4, two-stage split
# training with DiffSynth-Studio @7539a33. Stage 1 caches prompt embeddings and latents once; stage 2
# trains the LoRA for EPOCHS passes and saves a checkpoint after each, chosen on the development set.
# 1,028 pairs x 3 epochs ~ 3,084 steps, comparable to the 3,000 steps of the SDXL cut-line block.
# DiffSynth needs transformers 4.x (env "ds"); it reuses the local Hugging Face snapshot through a
# symlink, so nothing is downloaded again. Usage: train_qwen_lora.sh <dataset_dir> <work_dir> [epochs]
set -e
DATA=$1; WORK=$2; EPOCHS=${3:-3}
ROOTDISK=${ROOTDISK:-/root/autodl-tmp}
PY=${PY:-$ROOTDISK/envs/ds/bin}
DS=${DS:-$ROOTDISK/DiffSynth-Studio}
SNAP=${SNAP:-$ROOTDISK/hf/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9}
export DIFFSYNTH_MODEL_BASE_PATH=$WORK/models DIFFSYNTH_SKIP_DOWNLOAD=True
mkdir -p $WORK/models/Qwen && ln -sfn $SNAP $WORK/models/Qwen/Qwen-Image-Edit-2511
M="Qwen/Qwen-Image-Edit-2511"
MODELS="$M:transformer/diffusion_pytorch_model*.safetensors,$M:text_encoder/model*.safetensors,$M:vae/diffusion_pytorch_model.safetensors"
COMMON="--max_pixels 1048576 --model_id_with_origin_paths $MODELS --tokenizer_path $SNAP/tokenizer --processor_path $SNAP/processor \
  --learning_rate 1e-4 --remove_prefix_in_ckpt pipe.dit. --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1 \
  --lora_rank 32 --use_gradient_checkpointing --dataset_num_workers 4 --find_unused_parameters --zero_cond_t \
  --data_file_keys image,edit_image --extra_inputs edit_image"
cd $DS
if [ ! -f $WORK/cache/.done ]; then
  echo "== stage 1: cache"; date
  $PY/accelerate launch --mixed_precision bf16 examples/qwen_image/model_training/train.py $COMMON \
    --dataset_base_path $DATA --dataset_metadata_path $DATA/metadata.json --dataset_repeat 1 --num_epochs 1 \
    --output_path $WORK/cache --offload_models "$M:transformer/diffusion_pytorch_model*.safetensors" --task sft:data_process
  touch $WORK/cache/.done
fi
echo "== stage 2: train"; date
$PY/accelerate launch --mixed_precision bf16 examples/qwen_image/model_training/train.py $COMMON \
  --dataset_base_path $WORK/cache --dataset_repeat 1 --num_epochs $EPOCHS --output_path $WORK/lora \
  --offload_models "$M:text_encoder/model*.safetensors,$M:vae/diffusion_pytorch_model.safetensors" --task sft:train
echo "== done"; date; ls -la $WORK/lora
