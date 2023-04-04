#!/bin/bash
#SBATCH --job-name=exp_gauss_seed4_epoch007_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_gauss_seed4_epoch007_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 7 --DF_weights_path data/results/exp_gauss/seed4 --out_path data/results/exp_gauss/seed4 &

EXPNAME=exp_gauss_seed4_epoch008_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 8 --DF_weights_path data/results/exp_gauss/seed4 --out_path data/results/exp_gauss/seed4 &

EXPNAME=exp_gauss_seed4_epoch009_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 9 --DF_weights_path data/results/exp_gauss/seed4 --out_path data/results/exp_gauss/seed4 &

wait < <(jobs -p)