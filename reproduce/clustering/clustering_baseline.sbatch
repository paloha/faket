#!/bin/bash
#SBATCH --job-name=clustering_baseline
#SBATCH --qos=cpus150
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-1680  # Must start from 1 until end (including) so 1-3 runs indices 0, 1, 2
#SBATCH --open-mode=append
#SBATCH --output=reproduce/clustering/%x.out

sed -n ${SLURM_ARRAY_TASK_ID}p reproduce/clustering/clustering_baseline.queue | bash
