#!/bin/bash
#SBATCH --job-name=exp_rstyled_seed1_epoch010_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_rstyled_seed1_epoch010_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 10 --DF_weights_path data/results/exp_rstyled/seed1 --out_path data/results/exp_rstyled/seed1 &

EXPNAME=exp_rstyled_seed1_epoch011_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 11 --DF_weights_path data/results/exp_rstyled/seed1 --out_path data/results/exp_rstyled/seed1 &

EXPNAME=exp_rstyled_seed1_epoch012_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 12 --DF_weights_path data/results/exp_rstyled/seed1 --out_path data/results/exp_rstyled/seed1 &

wait < <(jobs -p)