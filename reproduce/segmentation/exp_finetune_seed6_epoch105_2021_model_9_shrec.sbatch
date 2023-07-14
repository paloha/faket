#!/bin/bash
#SBATCH --job-name=exp_finetune_seed6_epoch105_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_finetune_seed6_epoch105_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 105 --DF_weights_path data/results/exp_finetune/seed6 --out_path data/results/exp_finetune/seed6 &

EXPNAME=exp_finetune_seed6_epoch106_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 106 --DF_weights_path data/results/exp_finetune/seed6 --out_path data/results/exp_finetune/seed6 &

EXPNAME=exp_finetune_seed6_epoch107_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 107 --DF_weights_path data/results/exp_finetune/seed6 --out_path data/results/exp_finetune/seed6 &

wait < <(jobs -p)