#!/bin/bash
#
#SBATCH --job-name=download_image                            # Job name
#SBATCH --output=Log/%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:1                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=0-04:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=2G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=1                                 # CPU cores requested per task


wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip

exit