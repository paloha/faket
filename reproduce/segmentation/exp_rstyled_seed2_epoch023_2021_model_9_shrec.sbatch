#!/bin/bash
#SBATCH --job-name=exp_rstyled_seed2_epoch023_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_rstyled_seed2_epoch023_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 23 --DF_weights_path data/results/exp_rstyled/seed2 --out_path data/results/exp_rstyled/seed2 &

EXPNAME=exp_rstyled_seed2_epoch024_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 24 --DF_weights_path data/results/exp_rstyled/seed2 --out_path data/results/exp_rstyled/seed2 &

EXPNAME=exp_rstyled_seed2_epoch025_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 25 --DF_weights_path data/results/exp_rstyled/seed2 --out_path data/results/exp_rstyled/seed2 &

wait < <(jobs -p)