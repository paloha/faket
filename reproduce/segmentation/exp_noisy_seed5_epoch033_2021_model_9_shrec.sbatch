#!/bin/bash
#SBATCH --job-name=exp_noisy_seed5_epoch033_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed5_epoch033_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 33 --DF_weights_path data/results/exp_noisy/seed5 --out_path data/results/exp_noisy/seed5 &

EXPNAME=exp_noisy_seed5_epoch034_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 34 --DF_weights_path data/results/exp_noisy/seed5 --out_path data/results/exp_noisy/seed5 &

EXPNAME=exp_noisy_seed5_epoch035_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 35 --DF_weights_path data/results/exp_noisy/seed5 --out_path data/results/exp_noisy/seed5 &

wait < <(jobs -p)