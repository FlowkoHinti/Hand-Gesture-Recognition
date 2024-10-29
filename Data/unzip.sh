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

cd /srv/GadM/Datasets/Tmp/Hand-Gesture-Recognition/data || { echo "Directory not found!"; exit 1; }

if [ -f "hagridv2_512.zip" ]; then
    unzip hagridv2_512.zip || { echo "Failed to unzip hagridv2_512.zip"; exit 1; }
else
    echo "hagridv2_512.zip not found!"
fi

if [ -f "annotations.zip" ]; then
    unzip annotations.zip || { echo "Failed to unzip annotations.zip"; exit 1; }
else
    echo "annotations.zip not found!"
fi

exit 0