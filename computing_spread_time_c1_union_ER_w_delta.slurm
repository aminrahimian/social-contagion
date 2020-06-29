#!/bin/bash
#SBATCH --job-name="computing_spread_time_c1_union_ER_w_delta"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=100G
#SBATCH --partition=sched_mit_sloan_sinana
#SBATCH --time=7-00:00:00


#SBATCH -o /home/rahimian/contagion/computing_spread_time_c1_union_ER_w_delta.out
#SBATCH -e /home/rahimian/contagion/computing_spread_time_c1_union_ER_w_delta.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rahimian@mit.edu

cd /home/rahimian/phase2/contagion/

/home/rahimian/miniconda3/bin/python3.6 computing_spread_time_c1_union_ER_w_delta.py
