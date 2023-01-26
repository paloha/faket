#!/bin/bash
#SBATCH --job-name=exp_shrec_seed6_epoch035_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_shrec_seed6_epoch035_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 35 --DF_weights_path data/results/exp_shrec/seed6 --out_path data/results/exp_shrec/seed6 &

EXPNAME=exp_shrec_seed6_epoch036_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 36 --DF_weights_path data/results/exp_shrec/seed6 --out_path data/results/exp_shrec/seed6 &

EXPNAME=exp_shrec_seed6_epoch037_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 37 --DF_weights_path data/results/exp_shrec/seed6 --out_path data/results/exp_shrec/seed6 &

wait < <(jobs -p)