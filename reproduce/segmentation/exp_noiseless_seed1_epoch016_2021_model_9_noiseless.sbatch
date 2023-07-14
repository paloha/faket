#!/bin/bash
#SBATCH --job-name=exp_noiseless_seed1_epoch016_2021_model_9_noiseless
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noiseless_seed1_epoch016_2021_model_9_noiseless PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram noiseless --num_epochs 16 --DF_weights_path data/results/exp_noiseless/seed1 --out_path data/results/exp_noiseless/seed1 &

EXPNAME=exp_noiseless_seed1_epoch017_2021_model_9_noiseless PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram noiseless --num_epochs 17 --DF_weights_path data/results/exp_noiseless/seed1 --out_path data/results/exp_noiseless/seed1 &

EXPNAME=exp_noiseless_seed1_epoch018_2021_model_9_noiseless PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram noiseless --num_epochs 18 --DF_weights_path data/results/exp_noiseless/seed1 --out_path data/results/exp_noiseless/seed1 &

wait < <(jobs -p)