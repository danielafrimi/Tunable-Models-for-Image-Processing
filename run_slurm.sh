#!/bin/bash
#SBATCH --mail-user=daniel.afrimi@mail.huji.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1,vmem:3g
#SBATCH --killable
#SBATCH --mem=5g
#SBATCH --time=7-0
#SBATCH -c2
#SBATCH --output=/cs/labs/werman/daniel023/Lav_vision/FTN/sbatch_%J.out

module load cuda
module load torch

source /cs/labs/werman/daniel023/lab_env/bin/activate.csh
cd /cs/labs/werman/daniel023/Lab_vision/FTN

python3 Main.py

