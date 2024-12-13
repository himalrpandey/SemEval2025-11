#!/bin/bash
#SBATCH -c 1                # Request 1 CPU core
#SBATCH -t 0-02:00          # Runtime in D-HH:MM
#SBATCH --partition=gpmoo-a # Partition
#SBATCH --mem=10G           # Memory
#SBATCH -o myoutput_%j.out  # Output file
#SBATCH -e myerrors_%j.err  # Error file
#SBATCH --gres=gpu:1      # GPUs requested

~/.conda/envs/cs375final/bin/python dev_emotion_classifiers.py
