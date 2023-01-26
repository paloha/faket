#!/bin/bash
#SBATCH --job-name=exp_noiseless_seed3_epoch032_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noiseless_seed3_epoch032_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 32 --DF_weights_path data/results/exp_noiseless/seed3 --out_path data/results/exp_noiseless/seed3 &

EXPNAME=exp_noiseless_seed3_epoch033_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 33 --DF_weights_path data/results/exp_noiseless/seed3 --out_path data/results/exp_noiseless/seed3 &

EXPNAME=exp_noiseless_seed3_epoch034_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 34 --DF_weights_path data/results/exp_noiseless/seed3 --out_path data/results/exp_noiseless/seed3 &

wait < <(jobs -p)