#!/bin/bash
#SBATCH --job-name=exp_baseline3_seed6_epoch148_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_baseline3_seed6_epoch148_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 148 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

EXPNAME=exp_baseline3_seed6_epoch151_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 151 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

EXPNAME=exp_baseline3_seed6_epoch154_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 154 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

wait < <(jobs -p)