#!/bin/bash
#SBATCH --job-name=exp_noisy_seed1_epoch010_2021_model_9_baseline
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed1_epoch010_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 10 --DF_weights_path data/results/exp_noisy/seed1 --out_path data/results/exp_noisy/seed1 &

EXPNAME=exp_noisy_seed1_epoch011_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 11 --DF_weights_path data/results/exp_noisy/seed1 --out_path data/results/exp_noisy/seed1 &

EXPNAME=exp_noisy_seed1_epoch012_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 12 --DF_weights_path data/results/exp_noisy/seed1 --out_path data/results/exp_noisy/seed1 &

wait < <(jobs -p)