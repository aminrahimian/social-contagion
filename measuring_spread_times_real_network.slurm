#!/bin/bash
#SBATCH --job-name="measuring_spread_time_real_networks"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=4-00:00:00
#SBATCH --array=1-2


#SBATCH -o /home/clintonr/contagion/measuring_spread_time_real_networks.out
#SBATCH -e /home/clintonr/contagion/measure_spread_time_real_networks.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clintonr@mit.edu

cd /home/clintonr/contagion/

/home/clintonr/miniconda3/envs/py36/bin/python3.6
measuring_spread_time_real_networks.py ${SLURM_ARRAY_TASK_ID}
