#!/bin/bash
#SBATCH --job-name=exp_noisy_seed4_epoch034_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed4_epoch034_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 34 --DF_weights_path data/results/exp_noisy/seed4 --out_path data/results/exp_noisy/seed4 &

EXPNAME=exp_noisy_seed4_epoch035_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 35 --DF_weights_path data/results/exp_noisy/seed4 --out_path data/results/exp_noisy/seed4 &

EXPNAME=exp_noisy_seed4_epoch036_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 36 --DF_weights_path data/results/exp_noisy/seed4 --out_path data/results/exp_noisy/seed4 &

wait < <(jobs -p)