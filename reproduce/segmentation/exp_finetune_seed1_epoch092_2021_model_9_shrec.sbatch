#!/bin/bash
#SBATCH --job-name=exp_finetune_seed1_epoch092_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_finetune_seed1_epoch092_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 92 --DF_weights_path data/results/exp_finetune/seed1 --out_path data/results/exp_finetune/seed1 &

EXPNAME=exp_finetune_seed1_epoch093_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 93 --DF_weights_path data/results/exp_finetune/seed1 --out_path data/results/exp_finetune/seed1 &

EXPNAME=exp_finetune_seed1_epoch094_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 94 --DF_weights_path data/results/exp_finetune/seed1 --out_path data/results/exp_finetune/seed1 &

wait < <(jobs -p)