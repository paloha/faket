#!/bin/bash
#SBATCH --job-name=exp_shrec_seed2_epoch048_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_shrec_seed2_epoch048_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 48 --DF_weights_path data/results/exp_shrec/seed2 --out_path data/results/exp_shrec/seed2 &

EXPNAME=exp_shrec_seed2_epoch049_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 49 --DF_weights_path data/results/exp_shrec/seed2 --out_path data/results/exp_shrec/seed2 &

EXPNAME=exp_shrec_seed2_epoch050_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 50 --DF_weights_path data/results/exp_shrec/seed2 --out_path data/results/exp_shrec/seed2 &

wait < <(jobs -p)