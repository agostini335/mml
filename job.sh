#!/bin/bash
#SBATCH --job-name=mmltest
#SBATCH --cpus-per-task=
#SBATCH --mem=30G
#SBATCH --time=48:00:00
# make sure to select the right partition depending on your needs
#SBATCH --dynamic-a100gpu-32cores-240g
# enable the modules software stack
conda deactivate
conda activate mml39
# load your required software modules

lrs = (0.0001 0.00001 0.001)
seeds = (1 2 3)
policies = ("remove_uncertain" "uncertain_to_negative")
view_positions = ("AP" "PA")

for seed in ${seeds[@]}; do
for policy in ${policies[@]}; do
for lr in ${lrs[@]}; do
for view_position in ${view_positions[@]}; do


python main.py ++model.initial_lr=${lr} ++experiment.seed=${seed} ++experiment.label_policy=${policy} ++experiment.view_position=${view_position}

done
done
done
done