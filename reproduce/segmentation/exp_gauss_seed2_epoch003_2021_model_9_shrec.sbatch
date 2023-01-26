#!/bin/bash
#SBATCH --job-name=exp_gauss_seed2_epoch003_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_gauss_seed2_epoch003_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 3 --DF_weights_path data/results/exp_gauss/seed2 --out_path data/results/exp_gauss/seed2 &

EXPNAME=exp_gauss_seed2_epoch004_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 4 --DF_weights_path data/results/exp_gauss/seed2 --out_path data/results/exp_gauss/seed2 &

EXPNAME=exp_gauss_seed2_epoch005_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 5 --DF_weights_path data/results/exp_gauss/seed2 --out_path data/results/exp_gauss/seed2 &

wait < <(jobs -p)