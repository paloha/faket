#!/bin/bash
#SBATCH --job-name=exp_gauss_seed6_epoch029_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_gauss_seed6_epoch029_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 29 --DF_weights_path data/results/exp_gauss/seed6 --out_path data/results/exp_gauss/seed6 &

EXPNAME=exp_gauss_seed6_epoch030_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 30 --DF_weights_path data/results/exp_gauss/seed6 --out_path data/results/exp_gauss/seed6 &

EXPNAME=exp_gauss_seed6_epoch031_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 31 --DF_weights_path data/results/exp_gauss/seed6 --out_path data/results/exp_gauss/seed6 &

wait < <(jobs -p)