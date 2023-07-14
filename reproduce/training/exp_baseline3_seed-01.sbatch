#!/bin/bash
#SBATCH --job-name=exp_baseline3_seed-01
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/training/%x.out

EXPNAME=exp_baseline3_seed-01 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 6 7 8 --training_tomograms baseline baseline baseline --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 210 --out_path data/results/exp_baseline3/seed1 --save_every 1 --seed 1 --continue_training_path data/results/exp_baseline3/seed1 &

EXPNAME=exp_baseline3_seed-02 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 6 7 8 --training_tomograms baseline baseline baseline --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 210 --out_path data/results/exp_baseline3/seed2 --save_every 1 --seed 2 --continue_training_path data/results/exp_baseline3/seed2 &

wait < <(jobs -p)