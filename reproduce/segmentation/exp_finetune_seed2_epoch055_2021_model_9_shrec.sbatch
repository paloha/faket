#!/bin/bash
#SBATCH --job-name=exp_finetune_seed2_epoch055_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_finetune_seed2_epoch055_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 55 --DF_weights_path data/results/exp_finetune/seed2 --out_path data/results/exp_finetune/seed2 &

EXPNAME=exp_finetune_seed2_epoch056_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 56 --DF_weights_path data/results/exp_finetune/seed2 --out_path data/results/exp_finetune/seed2 &

EXPNAME=exp_finetune_seed2_epoch057_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 57 --DF_weights_path data/results/exp_finetune/seed2 --out_path data/results/exp_finetune/seed2 &

wait < <(jobs -p)