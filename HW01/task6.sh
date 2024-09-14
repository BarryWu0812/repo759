#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --cpus-per-task=2
#SBATCH --job-name=FirstSlurm
#SBATCH --output="FirstSlurm.out"
#SBATCH -e FirstSlurm.err
#SBARCH --time=0-00:00:01

./task6 6
