#!/bin/bash
#SBATCH --job-name=exp_baseline3_seed-05
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/training/%x.out

EXPNAME=exp_baseline3_seed-05 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 6 7 8 --training_tomograms baseline baseline baseline --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 210 --out_path data/results/exp_baseline3/seed5 --save_every 1 --seed 5 --continue_training_path data/results/exp_baseline3/seed5 &

EXPNAME=exp_baseline3_seed-06 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 6 7 8 --training_tomograms baseline baseline baseline --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 210 --out_path data/results/exp_baseline3/seed6 --save_every 1 --seed 6 --continue_training_path data/results/exp_baseline3/seed6 &

wait < <(jobs -p)