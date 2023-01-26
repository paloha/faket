#!/bin/bash
#SBATCH --job-name=exp_baseline_seed6_epoch020_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_baseline_seed6_epoch020_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 20 --DF_weights_path data/results/exp_baseline/seed6 --out_path data/results/exp_baseline/seed6 &

EXPNAME=exp_baseline_seed6_epoch021_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 21 --DF_weights_path data/results/exp_baseline/seed6 --out_path data/results/exp_baseline/seed6 &

EXPNAME=exp_baseline_seed6_epoch022_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 22 --DF_weights_path data/results/exp_baseline/seed6 --out_path data/results/exp_baseline/seed6 &

wait < <(jobs -p)