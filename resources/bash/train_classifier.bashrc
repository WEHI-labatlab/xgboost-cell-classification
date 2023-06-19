#!/bin/bash
#SBATCH --job-name=cell-class
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=56
#SBATCH --nodes=1              # node count
#SBATCH --mem=500G
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<your username>@wehi.edu.au
#SBATCH --output /home/users/allstaff/<your username>/xgboost-cell-classification/resources/slurm-output/%N_%j.out

module load anaconda3
conda activate xgboost-cell-classification
python /home/users/allstaff/<your username>/xgboost-cell-classification/src/train_classifier.py /home/users/allstaff/<your username>/xgboost-cell-classification/resources/json/training.json