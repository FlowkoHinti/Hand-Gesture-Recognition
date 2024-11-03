#!/bin/bash
#
#SBATCH --job-name=train_hagrid                                # Job name
#SBATCH --output=Log/%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:2                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=5-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=8                                  # CPU cores requested per task


eval "$(conda shell.bash hook)"
conda activate /home2/phofmann/miniconda/envs/hagrid/


cd /srv/GadM/Datasets/Tmp/Hand-Gesture-Recognition/2_Modelling/

python TrainClassifier.py --model=vit --setup=frozen --epochs=20 --batch_size=64 --learning_rate=0.001 --patience=3

exit