#!/bin/bash
#SBATCH --job-name=exp_gauss_seed2_epoch069_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_gauss_seed2_epoch069_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 69 --DF_weights_path data/results/exp_gauss/seed2 --out_path data/results/exp_gauss/seed2 &

EXPNAME=exp_gauss_seed2_epoch070_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 70 --DF_weights_path data/results/exp_gauss/seed2 --out_path data/results/exp_gauss/seed2 &

EXPNAME=exp_gauss_seed3_epoch001_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 1 --DF_weights_path data/results/exp_gauss/seed3 --out_path data/results/exp_gauss/seed3 &

wait < <(jobs -p)