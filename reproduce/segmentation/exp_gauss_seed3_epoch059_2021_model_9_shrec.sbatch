#!/bin/bash
#SBATCH --job-name=exp_gauss_seed3_epoch059_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_gauss_seed3_epoch059_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 59 --DF_weights_path data/results/exp_gauss/seed3 --out_path data/results/exp_gauss/seed3 &

EXPNAME=exp_gauss_seed3_epoch060_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 60 --DF_weights_path data/results/exp_gauss/seed3 --out_path data/results/exp_gauss/seed3 &

EXPNAME=exp_gauss_seed3_epoch061_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 61 --DF_weights_path data/results/exp_gauss/seed3 --out_path data/results/exp_gauss/seed3 &

wait < <(jobs -p)