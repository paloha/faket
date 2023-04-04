#!/bin/bash
#SBATCH --job-name=exp_noisy_seed4_epoch064_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noisy_seed4_epoch064_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 64 --DF_weights_path data/results/exp_noisy/seed4 --out_path data/results/exp_noisy/seed4 &

EXPNAME=exp_noisy_seed4_epoch065_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 65 --DF_weights_path data/results/exp_noisy/seed4 --out_path data/results/exp_noisy/seed4 &

EXPNAME=exp_noisy_seed4_epoch066_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 66 --DF_weights_path data/results/exp_noisy/seed4 --out_path data/results/exp_noisy/seed4 &

wait < <(jobs -p)