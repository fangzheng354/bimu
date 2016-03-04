#!/bin/bash                                                                                                                                           
#SBATCH --partition=gpu                                                                                                                               
#SBATCH --gres=gpu:1                                                                                                                                  
#SBATCH --job-name=simon_testgpu                                                                                                                      

module load Python
module load Theano

python3.4 hello.py
