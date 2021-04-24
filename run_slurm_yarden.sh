#!/bin/bash
#SBATCH --mail-user=yarden.tal1@mail.huji.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --mem=5g
#SBATCH --time=7-0
#SBATCH -c2
#SBATCH --output=/cs/labs/roys/yardental/roberta_mnli_transformers/model_logs/sbatch_%J.out

module load cuda
module load torch

source /cs/labs/roys/yardental/envs/multimodel/bin/activate
cd /cs/labs/roys/yardental/roberta_mnli_transformers

python3 run_glue_no_trainer.py --model_name_or_path roberta-large --per_device_train_batch_size 8 --learning_rate 1e-5 --num_train_epochs 3 --weight_decay 0.1 --output_dir roberta_large --task_name mnli --pad_to_max_length --gradient_accumulation_steps 4 --num_warmup_steps 0.06 --cache_dir cache/roberta
