#!/bin/bash
#SBATCH --job-name=exp_noiseless_seed4_epoch049_2021_model_9_noiseless
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noiseless_seed4_epoch049_2021_model_9_noiseless PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram noiseless --num_epochs 49 --DF_weights_path data/results/exp_noiseless/seed4 --out_path data/results/exp_noiseless/seed4 &

EXPNAME=exp_noiseless_seed4_epoch050_2021_model_9_noiseless PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram noiseless --num_epochs 50 --DF_weights_path data/results/exp_noiseless/seed4 --out_path data/results/exp_noiseless/seed4 &

EXPNAME=exp_noiseless_seed4_epoch051_2021_model_9_noiseless PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram noiseless --num_epochs 51 --DF_weights_path data/results/exp_noiseless/seed4 --out_path data/results/exp_noiseless/seed4 &

wait < <(jobs -p)