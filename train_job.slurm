#!/bin/bash
#
#SBATCH --partition=1080ti-long    # Partition to submit to <m40-short|m40-long|teslax-short|teslax-long>
#SBATCH --job-name=lstm
#SBATCH -o model/train_log/lstm_res_%j.txt            # output file
#SBATCH -e model/train_log/lstm_res_%j.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --time=4-01:00:00          # D-HH:MM:SS
#SBATCH --gres=gpu:1             # Very Imp. Without this, CUDA does not recognise GPUs
#SBATCH --mem=10GB

source /home/akaruvally/.bashrc
source activate cvpro

hostname

python -u code/train_siamrpn.py --train_path=/home/akaruvally/scratch_dir/vot2013 \
                                    --checkpoint_path=/home/akaruvally/scratch_dir/weights-0690000.pth.tar \
                                    --weight_dir=/home/akaruvally/scratch_dir/weights

sleep 1
exit