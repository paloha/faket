#!/bin/bash
#SBATCH --job-name=exp_rstyled_seed4_epoch052_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_rstyled_seed4_epoch052_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 52 --DF_weights_path data/results/exp_rstyled/seed4 --out_path data/results/exp_rstyled/seed4 &

EXPNAME=exp_rstyled_seed4_epoch053_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 53 --DF_weights_path data/results/exp_rstyled/seed4 --out_path data/results/exp_rstyled/seed4 &

EXPNAME=exp_rstyled_seed4_epoch054_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 54 --DF_weights_path data/results/exp_rstyled/seed4 --out_path data/results/exp_rstyled/seed4 &

wait < <(jobs -p)