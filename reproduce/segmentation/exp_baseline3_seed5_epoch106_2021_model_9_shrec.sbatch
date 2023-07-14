#!/bin/bash
#SBATCH --job-name=exp_baseline3_seed5_epoch106_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_baseline3_seed5_epoch106_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 106 --DF_weights_path data/results/exp_baseline3/seed5 --out_path data/results/exp_baseline3/seed5 &

EXPNAME=exp_baseline3_seed5_epoch109_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 109 --DF_weights_path data/results/exp_baseline3/seed5 --out_path data/results/exp_baseline3/seed5 &

EXPNAME=exp_baseline3_seed5_epoch112_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 112 --DF_weights_path data/results/exp_baseline3/seed5 --out_path data/results/exp_baseline3/seed5 &

wait < <(jobs -p)