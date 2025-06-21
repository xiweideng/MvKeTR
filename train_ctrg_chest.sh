#!/bin/bash
############################################
export SEED=9223
export EPOCHS=30
export CLIP_TOP_K=16
export BATCH_SIZE=2
############################################
CUDA_VISIBLE_DEVICES=5,nohup python main_train.py \
--image_dir data/CTRG-Chest-548K/ \
--ann_file data/CTRG-Chest-548K/annotation.json \
--image_size 224 \
--spatial_patch_size 28 \
--temporal_patch_size 28 \
--voxel_depth 224 \
--max_seq_length 150 \
--clip_top_k $CLIP_TOP_K \
--dataset_name ct_dataset \
--threshold 3 \
--batch_size $BATCH_SIZE \
--epochs $EPOCHS \
--gpu_id 0 \
--lr_a_ve 5e-5 \
--lr_c_ve 5e-5 \
--lr_s_ve 5e-5 \
--lr_ed 1e-4 \
--step_size 3 \
--gamma 0.8 \
--num_layers 3 \
--topk 32 \
--cmm_size 2048 \
--cmm_dim 512 \
--beam_size 3 \
--d_vf 512 \
--save_dir results \
--log_period 1000 \
--seed $SEED > train.txt 2>&1 &