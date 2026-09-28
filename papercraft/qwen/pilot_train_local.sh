#!/bin/bash
# Pilot LoRA run on the 20 GB H100 slice (not used for reported results): two-stage split training,
# NF4 transformer, 512 px. Checks data format, memory, speed and LoRA export before the bf16 run.
# Usage: pilot_train_local.sh <dataset_dir> <work_dir> [n_pairs]
set -e
DATA=$1; WORK=$2; N=${3:-96}
PY=/home/ubuntu/dsenv/bin  # DiffSynth needs transformers 4.x; inference uses qwenenv (transformers 5.x)
SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9
export DIFFSYNTH_MODEL_BASE_PATH=$WORK/models DIFFSYNTH_SKIP_DOWNLOAD=True
mkdir -p $WORK/models/Qwen && ln -sfn $SNAP $WORK/models/Qwen/Qwen-Image-Edit-2511
$PY/python -c "import json,sys; m=json.load(open('$DATA/metadata.json')); json.dump(m[:$N], open('$WORK/metadata_subset.json','w'))"
M="Qwen/Qwen-Image-Edit-2511"
MODELS="$M:transformer/diffusion_pytorch_model*.safetensors,$M:text_encoder/model*.safetensors,$M:vae/diffusion_pytorch_model.safetensors"
COMMON="--max_pixels 262144 --model_id_with_origin_paths $MODELS --tokenizer_path $SNAP/tokenizer --processor_path $SNAP/processor \
  --learning_rate 1e-4 --num_epochs 1 --remove_prefix_in_ckpt pipe.dit. --lora_base_model dit \
  --lora_target_modules to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1 \
  --lora_rank 32 --use_gradient_checkpointing --dataset_num_workers 2 --find_unused_parameters --zero_cond_t \
  --data_file_keys image,edit_image --extra_inputs edit_image"
cd /home/ubuntu/DiffSynth-Studio
echo "== stage 1: cache"; date
$PY/accelerate launch examples/qwen_image/model_training/train.py $COMMON \
  --dataset_base_path $DATA --dataset_metadata_path $WORK/metadata_subset.json --dataset_repeat 1 \
  --output_path $WORK/cache --offload_models "$M:transformer/diffusion_pytorch_model*.safetensors" --task sft:data_process
echo "== stage 2: train"; date
$PY/accelerate launch examples/qwen_image/model_training/train.py $COMMON \
  --dataset_base_path $WORK/cache --dataset_repeat 1 --output_path $WORK/lora \
  --offload_models "$M:text_encoder/model*.safetensors,$M:vae/diffusion_pytorch_model.safetensors" \
  --quant_options "$M:transformer/diffusion_pytorch_model*.safetensors:bitsandbytes_nf4" --task sft:train
echo "== done"; date; ls -la $WORK/lora
