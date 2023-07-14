#!/bin/bash
#SBATCH --job-name=exp_finetune_seed3_epoch087_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_finetune_seed3_epoch087_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 87 --DF_weights_path data/results/exp_finetune/seed3 --out_path data/results/exp_finetune/seed3 &

EXPNAME=exp_finetune_seed3_epoch088_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 88 --DF_weights_path data/results/exp_finetune/seed3 --out_path data/results/exp_finetune/seed3 &

EXPNAME=exp_finetune_seed3_epoch089_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 89 --DF_weights_path data/results/exp_finetune/seed3 --out_path data/results/exp_finetune/seed3 &

wait < <(jobs -p)