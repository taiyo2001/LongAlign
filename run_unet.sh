#!/bin/bash

  #  1. 1回目の学習 (`bash run_unet.sh align ct5f`):
  #     このステップでは、Stable Diffusionモデルが長いテキスト入力を扱えるように学習します。readme.mdには「support long-text
  # inputs」と記載されており、run_unet.shを見るとunet_align_ct5.pyというスクリプトが実行されます。これは、モデルの初期学習フェーズに相当します。

  #  2. 2回目の学習 (`bash run_unet.sh reward test`):
  #     このステップでは、1回目の学習で得られたモデルを使い、選好最適化（Preference
  # Optimization）によって長いテキストと生成される画像との整合性（アライメント）をさらに向上させます。readme.mdには「preference optimization for
  # long-text alignment」と記載されており、run_unet.shを見るとunet_reward.pyが実行されます。この際、1回目の学習の成果物（--ckpt_dir
  # "results/sd15-align-ct5f/"）を読み込んでファインチューニングを行います。

  # 要約すると、1回目で長いテキストを扱えるようにし、2回目でそのテキストに対する画像の忠実度をさらに高める、という流れになります。

  # Train LCM-version Stable Diffusionも同様の2段階の学習を行いますが、こちらはLCM（Latent Consistency
  # Models）に対応したバージョンを生成するためのものです。


export CUDA_LAUNCH_BLOCKING=1

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# export HF_TOKEN="xxx"

# --debug でwandbの起動を抑制
pkill wandb

# ERROR:
# https://discuss.huggingface.co/t/how-to-use-specified-gpus-with-accelerator-to-train-the-model/10967/5


echo $1 $2
if [ $1 = 'align' ]; then
  export MY_ENVIRON_SAMPLE='unipc'
  export MY_ENVIRON_DROP='0.1'
  CUDA_VISIBLE_DEVICES='0' accelerate launch --config_file ./acc_config.yaml unet_align_ct5.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --token_length 240 \
  --resolution 512 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --learning_rate 3e-5 \
  --loss_type "l2" \
  --max_train_epochs 2 \
  --validation_steps 50 \
  --save_steps 100 \
  --output_dir "results/sd15-align-$2" \
  --ckpt_dir "" \
  --resume \
  --debug

elif [ $1 = 'lcm' ]; then
  export MY_ENVIRON_SAMPLE='lcm'
  export MY_ENVIRON_DROP='0.0'
  CUDA_VISIBLE_DEVICES='0,2' accelerate launch --config_file ./acc_config.yaml \
  --num_processes 2 unet_lcm.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --token_length 240 \
  --resolution 512 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --learning_rate 3e-5 \
  --loss_type "huber" \
  --max_train_epochs 2 \
  --validation_steps 150 \
  --save_steps 1250 \
  --output_dir "results/sd15-lcm-$2" \
  --ckpt_dir "results/sd15-align-ct5f/" \
  --resume \
  --debug

elif [ $1 = 'reward' ]; then
  export MY_ENVIRON_SAMPLE='unipc'
  export MY_ENVIRON_DROP='0.0'
  export REWEIGHT=${3:-0.3}
  CUDA_VISIBLE_DEVICES='0' accelerate launch --config_file ./acc_config.yaml unet_reward.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --token_length 240 \
  --resolution 512 \
  --batch_size 1 \
  --accumulation_steps 8 \
  --learning_rate 3e-5 \
  --loss_type "l2" \
  --max_train_epochs 1 \
  --validation_steps 50 \
  --save_steps 100 \
  --output_dir "results/sd15-reward-$2" \
  --ckpt_dir "results/sd15-align-ct5f/" \
  --resume \
  --debug

elif [ $1 = 'reward-lcm' ]; then
  export MY_ENVIRON_SAMPLE='unipc'
  export MY_ENVIRON_DROP='0.0'
  export REWEIGHT=${3:-0.3}
  CUDA_VISIBLE_DEVICES='0,2' accelerate launch --config_file ./acc_config.yaml \
  --num_processes 2 unet_reward_lcm.py \
  --pretrained_decoder="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --mixed_precision="bf16" \
  --token_length 240 \
  --resolution 512 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --learning_rate 3e-5 \
  --loss_type "l2" \
  --max_train_epochs 1 \
  --validation_steps 150 \
  --save_steps 1250 \
  --output_dir "results/sd15-rewardlcm-$2" \
  --ckpt_dir "results/sd15-align-ct5f/, results/sd15-lcm-ct5f/" \
  --resume \
  --debug

fi
