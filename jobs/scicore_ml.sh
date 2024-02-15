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
    sbatch -o "${logdir}/outputs/${name}.log" -J $name -x ${exclude[@]} \
    -c 8 -t 20:00:00 -p gpu --gres=gpu:1 --mem=16G --wrap \
    "python main.py \
    --config-name=${config_file} \
    ++epochs=${epochs} \
    ++seed=${seed} \
    ++optimizer.lr=${lr} \
    ++augmentation.s=${aug_s} \
    ++loss.temperature=${temperature} \
    ++loss.similarity_metric=${similarity_metric} \
    ++loss.projection_dim=${proj_dim} \
    ++loss.use_true_negatives=False \
    ++eval_every_n_epochs=256 \
    ++logging.wandb_project_name=${wandb_project_name} \
    ++logging.logdir=${logdir} \
    ++logging.wandb_run_name=${name} \
    ++augmentation.test_time_augmentations=4 \
    ++dataset.normal_class=${normal_class} \
    ++dataset.data_path=/cluster/work/vogtlab/Projects/CIFAR10"
done
done
done
done
done