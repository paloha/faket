#!/bin/bash
#SBATCH --job-name=exp_gauss_seed5_epoch063_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_gauss_seed5_epoch063_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 63 --DF_weights_path data/results/exp_gauss/seed5 --out_path data/results/exp_gauss/seed5 &

EXPNAME=exp_gauss_seed5_epoch064_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 64 --DF_weights_path data/results/exp_gauss/seed5 --out_path data/results/exp_gauss/seed5 &

EXPNAME=exp_gauss_seed5_epoch065_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 65 --DF_weights_path data/results/exp_gauss/seed5 --out_path data/results/exp_gauss/seed5 &

wait < <(jobs -p)