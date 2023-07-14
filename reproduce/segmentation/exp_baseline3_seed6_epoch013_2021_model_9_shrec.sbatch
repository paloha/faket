#!/bin/bash
#SBATCH --job-name=exp_baseline3_seed6_epoch013_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_baseline3_seed6_epoch013_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 13 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

EXPNAME=exp_baseline3_seed6_epoch016_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 16 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

EXPNAME=exp_baseline3_seed6_epoch019_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 19 --DF_weights_path data/results/exp_baseline3/seed6 --out_path data/results/exp_baseline3/seed6 &

wait < <(jobs -p)