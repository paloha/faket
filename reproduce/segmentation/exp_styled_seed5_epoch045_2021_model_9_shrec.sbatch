#!/bin/bash
#SBATCH --job-name=exp_styled_seed5_epoch045_2021_model_9_shrec
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/segmentation/%x.out

EXPNAME=exp_styled_seed5_epoch045_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 45 --DF_weights_path data/results/exp_styled/seed5 --out_path data/results/exp_styled/seed5 &

EXPNAME=exp_styled_seed5_epoch046_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 46 --DF_weights_path data/results/exp_styled/seed5 --out_path data/results/exp_styled/seed5 &

EXPNAME=exp_styled_seed5_epoch047_2021_model_9_shrec PYTHONHASHSEED=0 python3 faket/deepfinder/launch_segmentation.py --test_tomo_path data/shrec2021_extended_dataset/ --test_tomo_idx 9 --test_tomogram shrec --num_epochs 47 --DF_weights_path data/results/exp_styled/seed5 --out_path data/results/exp_styled/seed5 &

wait < <(jobs -p)