#!/bin/bash
#SBATCH --job-name=exp_baseline3_seed6_epoch076_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_baseline3_seed6_epoch076_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 76 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

EXPNAME=exp_baseline3_seed6_epoch079_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 79 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

EXPNAME=exp_baseline3_seed6_epoch082_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 82 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

wait < <(jobs -p)