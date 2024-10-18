#!/bin/bash
#
#SBATCH --job-name=unzip_image                            # Job name
#SBATCH --output=Log/%x_%j.out                                  # Output file (includes job name and ID)
#SBATCH --error=Log/%x_%j.err                                   # Error file (includes job name and ID)
#SBATCH --gres=gpu:1                                        # Number of GPUs
#SBATCH --ntasks=1                                          # Number of processes
#SBATCH --time=1-00:00                                      # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4G                                    # Memory per CPU allocated
#SBATCH --cpus-per-task=2                                 # CPU cores requested per task


unzip hagridv2_512.zip
unzip annotations.zip

exit