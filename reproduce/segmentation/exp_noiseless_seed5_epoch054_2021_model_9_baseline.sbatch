#!/bin/bash
#SBATCH --job-name=exp_noiseless_seed5_epoch054_2021_model_9_baseline
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noiseless_seed5_epoch054_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 54 --DF_weights_path data/results/exp_noiseless/seed5 --out_path data/results/exp_noiseless/seed5 &

EXPNAME=exp_noiseless_seed5_epoch055_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 55 --DF_weights_path data/results/exp_noiseless/seed5 --out_path data/results/exp_noiseless/seed5 &

EXPNAME=exp_noiseless_seed5_epoch056_2021_model_9_baseline PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram baseline --num_epochs 56 --DF_weights_path data/results/exp_noiseless/seed5 --out_path data/results/exp_noiseless/seed5 &

wait < <(jobs -p)