#!/bin/bash
#SBATCH --job-name=exp_noisy_seed6_epoch062_2021_model_9_baseline
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed6_epoch062_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 62 --DF_weights_path data/results/exp_noisy/seed6 --out_path data/results/exp_noisy/seed6 &

EXPNAME=exp_noisy_seed6_epoch063_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 63 --DF_weights_path data/results/exp_noisy/seed6 --out_path data/results/exp_noisy/seed6 &

EXPNAME=exp_noisy_seed6_epoch064_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 64 --DF_weights_path data/results/exp_noisy/seed6 --out_path data/results/exp_noisy/seed6 &

wait < <(jobs -p)