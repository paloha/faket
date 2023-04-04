#!/bin/bash
#SBATCH --job-name=exp_gauss_seed-03
#SBATCH --qos=gpus4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=reproduce/training/%x.out

EXPNAME=exp_gauss_seed-03 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 0 1 2 3 4 5 6 7 8 --training_tomograms gauss gauss gauss gauss gauss gauss gauss gauss gauss --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 150 --out_path data/results/exp_gauss/seed3 --save_every 1 --seed 3 --continue_training_path data/results/exp_gauss/seed3 &

EXPNAME=exp_gauss_seed-04 PYTHONHASHSEED=0 python3 faket/deepfinder/launch_training.py --training_tomogram_ids 0 1 2 3 4 5 6 7 8 --training_tomograms gauss gauss gauss gauss gauss gauss gauss gauss gauss --training_tomo_path data/shrec2021_extended_dataset/ --num_epochs 150 --out_path data/results/exp_gauss/seed4 --save_every 1 --seed 4 --continue_training_path data/results/exp_gauss/seed4 &

wait < <(jobs -p)