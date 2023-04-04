#!/bin/bash
#SBATCH --job-name=exp_noisy_seed6_epoch050_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed6_epoch050_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 50 --DF_weights_path data/results/exp_noisy/seed6 --out_path data/results/exp_noisy/seed6 &

EXPNAME=exp_noisy_seed6_epoch051_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 51 --DF_weights_path data/results/exp_noisy/seed6 --out_path data/results/exp_noisy/seed6 &

EXPNAME=exp_noisy_seed6_epoch052_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 52 --DF_weights_path data/results/exp_noisy/seed6 --out_path data/results/exp_noisy/seed6 &

wait < <(jobs -p)