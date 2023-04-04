#!/bin/bash
#SBATCH --job-name nst_tomogram_8
#SBATCH --qos=gpus2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --output reproduce/nst/%x.out

EXPNAME=nst_tomogram_8 PYTHONHASHSEED=0 python3 -m faket.style_transfer.cli data/shrec2021_extended_dataset/model_8/faket/projections_content.mrc data/shrec2021_extended_dataset/model_0/projections_unbinned.mrc --style-weights 1.0 --content-weight 1.0 --tv-weight 0 --min-scale 1024 --end-scale 1024 --iterations 1 --initial-iterations 1 --save-every 2 --step-size 0.15 --avg-decay 0.99 --style-scale-fac 1.0 --pooling max --devices cuda:0 --seq_start 0 --seq_end 61 --content_layers 8 --content_layers_weights 100 --model_weights pretrained --init data/shrec2021_extended_dataset/model_8/faket/projections_noisy.mrc --output data/shrec2021_extended_dataset/model_8/faket/projections_styled.mrc --random-seed 8