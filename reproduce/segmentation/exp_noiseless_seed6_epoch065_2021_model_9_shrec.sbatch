#!/bin/bash
#SBATCH --job-name=exp_noiseless_seed6_epoch065_2021_model_9_shrec
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_noiseless_seed6_epoch065_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 65 --DF_weights_path data/results/exp_noiseless/seed6 --out_path data/results/exp_noiseless/seed6 &

EXPNAME=exp_noiseless_seed6_epoch066_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 66 --DF_weights_path data/results/exp_noiseless/seed6 --out_path data/results/exp_noiseless/seed6 &

EXPNAME=exp_noiseless_seed6_epoch067_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 67 --DF_weights_path data/results/exp_noiseless/seed6 --out_path data/results/exp_noiseless/seed6 &

wait < <(jobs -p)