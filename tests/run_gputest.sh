#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=simon_testgpu

module load Python
module load Theano

cd ~/bilingualmultisense/bimu
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3.4 tests/gpu.py