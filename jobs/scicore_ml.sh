#!/bin/bash
source ~/.bashrc
conda activate anomaly_detection

LOG_DIR="../logs"
wandb_project_name=mml

logdir="${LOG_DIR}/${wandb_project_name}"
mkdir -p $logdir
mkdir -p $logdir/outputs/
mkdir -p $logdir/wandb/

# exclude=compute-biomed-01

num_splits=1
epochs=300
config_file="config"

lrs=(1e-4)

models=("resnet18")
lrs=(1e-4 1e-5 1e-3)
seeds=(0 1 2 3 4 0)

for model in ${models[@]}; do
for lr in ${lrs[@]}; do
for seed in ${seeds[@]}; do
    name="${model}_model_${lr}_lr_${seed}_seed_"
    echo $name
    #srun -c 8 -t 4:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=4096\
    # -c 8 -t 16:00:00 -p gpu --gres=gpu:rtx2080ti:1 --mem=16G --wrap \
    sbatch -o "${logdir}/outputs/${name}.log" -J $name  --cpus-per-task=32\
    -c 8 -t 20:00:00 --partition=dynamic-a100gpu-32cores-240g --wrap \
    "python main.py \
    --config-name=${config_file} \
    ++model.epochs=${epochs} \
    ++model.seed=${seed} \
    ++experiment.seed=${seed} \
    ++model.initial_lr=${lr} \
    ++model.name=${model} \
    ++dataset.root_dir_PA=/cluster/work/vogtlab/Projects/CIFAR10 \
    ++dataset.root_dir_AP=/cluster/work/vogtlab/Projects/CIFAR10
    ++dataset.root_dir_AP=/cluster/work/vogtlab/Projects/CIFAR10\"


done
done
done
done
done