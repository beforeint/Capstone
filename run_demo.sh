#!/bin/bash
#SBATCH --job-name=demo
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/demo_%j.out
#SBATCH --error=logs/demo_%j.err

# 사용 예:
#   sbatch run_demo.sh                    # KG 사용 (기본)
#   sbatch run_demo.sh --no_kg            # baseline 비교
#   sbatch run_demo.sh --epochs 50        # epoch 수 조정

mkdir -p logs

source /home1/pz29075/miniconda3/etc/profile.d/conda.sh
conda activate stockmixer

cd "$(dirname "$0")"   # demo_server.py가 있는 폴더로 이동
python demo_server.py "$@"
