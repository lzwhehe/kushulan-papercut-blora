#!/bin/bash
# Edit-LoRA on Qwen-Image-Edit-2511 with DiffSynth-Studio, loading the local Hugging Face snapshot
# (no second download from ModelScope). Usage: train_qwen_lora.sh <dataset_dir> <output_dir> [epochs]
# 1,028 pairs (257 images x dense/coarse x flip) x 3 epochs ~ 3,084 steps at batch 1, comparable to
# the 3,000 steps of the SDXL cut-line block. A checkpoint is saved after every epoch; the epoch is
# chosen on the development drawings only.
set -e
DATA=$1; OUT=$2; EPOCHS=${3:-3}
export HF_HOME=/root/autodl-tmp/hf
SNAP=$HF_HOME/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9
DS=/root/autodl-tmp/DiffSynth-Studio
PY=/root/autodl-tmp/envs/qwen/bin
MODEL_PATHS=$($PY/python - "$SNAP" <<'EOF'
import glob, json, sys
s = sys.argv[1]
print(json.dumps([sorted(glob.glob(f"{s}/transformer/*.safetensors")),
                  sorted(glob.glob(f"{s}/text_encoder/*.safetensors")),
                  f"{s}/vae/diffusion_pytorch_model.safetensors"]))
EOF
)
cd $DS
$PY/accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "$DATA" \
  --dataset_metadata_path "$DATA/metadata.json" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "$SNAP/tokenizer" \
  --processor_path "$SNAP/processor" \
  --learning_rate 1e-4 \
  --num_epochs "$EPOCHS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUT" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 4 \
  --find_unused_parameters \
  --zero_cond_t
