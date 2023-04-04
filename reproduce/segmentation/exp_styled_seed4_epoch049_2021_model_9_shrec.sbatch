#!/bin/bash
#SBATCH --job-name=exp_styled_seed4_epoch049_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_styled_seed4_epoch049_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 49 --DF_weights_path data/results/exp_styled/seed4 --out_path data/results/exp_styled/seed4 &

EXPNAME=exp_styled_seed4_epoch050_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 50 --DF_weights_path data/results/exp_styled/seed4 --out_path data/results/exp_styled/seed4 &

EXPNAME=exp_styled_seed4_epoch051_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 51 --DF_weights_path data/results/exp_styled/seed4 --out_path data/results/exp_styled/seed4 &

wait < <(jobs -p)