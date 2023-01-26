#!/bin/bash
#SBATCH --job-name=exp_noisy_seed3_epoch035_2021_model_9_baseline
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed3_epoch035_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 35 --DF_weights_path data/results/exp_noisy/seed3 --out_path data/results/exp_noisy/seed3 &

EXPNAME=exp_noisy_seed3_epoch036_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 36 --DF_weights_path data/results/exp_noisy/seed3 --out_path data/results/exp_noisy/seed3 &

EXPNAME=exp_noisy_seed3_epoch037_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 37 --DF_weights_path data/results/exp_noisy/seed3 --out_path data/results/exp_noisy/seed3 &

wait < <(jobs -p)