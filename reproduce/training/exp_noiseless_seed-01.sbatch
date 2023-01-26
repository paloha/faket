#!/bin/bash
#SBATCH --job-name=exp_noiseless_seed-01
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/training/%x.out

EXPNAME=exp_noiseless_seed-01 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 0 1 2 3 4 5 6 7 8 --training_tomograms noiseless noiseless noiseless noiseless noiseless noiseless noiseless noiseless noiseless --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 70 --out_path data/results/exp_noiseless/seed1 --save_every 1 --seed 1 --continue_training_path data/results/exp_noiseless/seed1 &

EXPNAME=exp_noiseless_seed-02 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 0 1 2 3 4 5 6 7 8 --training_tomograms noiseless noiseless noiseless noiseless noiseless noiseless noiseless noiseless noiseless --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 70 --out_path data/results/exp_noiseless/seed2 --save_every 1 --seed 2 --continue_training_path data/results/exp_noiseless/seed2 &

wait < <(jobs -p)