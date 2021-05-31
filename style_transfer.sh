#!/bin/bash
#SBATCH --mail-user=daniel.afrimi@mail.huji.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --killable
#SBATCH --mem=7g
#SBATCH --time=7-0
#SBATCH -c2
#SBATCH --output=/cs/labs/werman/daniel023/Lab_vision/model_logs/sbatch_%J.out

module load cuda
module load torch

source /cs/labs/werman/daniel023/lab_env/bin/activate
cd /cs/labs/werman/daniel023/Lab_vision/StyleTransferFTN

python3 style_ftn.py train

