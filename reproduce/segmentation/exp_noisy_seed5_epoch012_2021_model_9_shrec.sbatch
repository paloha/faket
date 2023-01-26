#!/bin/bash
#SBATCH --job-name=exp_noisy_seed5_epoch012_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed5_epoch012_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 12 --DF_weights_path data/results/exp_noisy/seed5 --out_path data/results/exp_noisy/seed5 &

EXPNAME=exp_noisy_seed5_epoch013_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 13 --DF_weights_path data/results/exp_noisy/seed5 --out_path data/results/exp_noisy/seed5 &

EXPNAME=exp_noisy_seed5_epoch014_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 14 --DF_weights_path data/results/exp_noisy/seed5 --out_path data/results/exp_noisy/seed5 &

wait < <(jobs -p)