#!/bin/bash
#SBATCH --job-name="measuring_spread_time_real_networks"
#SBATCH --ntasks-per-node=26
#SBATCH --nodes=4
#SBATCH --mem-per-cpu=3G
#SBATCH --cluster=smp
#SBATCH --partition=legacy
#SBATCH --time=12:00
#SBATCH --array=4-19


#SBATCH -o /home/clintonr/contagion/measuring_spread_time_real_networks.out
#SBATCH -e /home/clintonr/contagion/measure_spread_time_real_networks.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=clintonr@mit.edu

module load python/anaconda3.6-5.2.0
#module load sloan/python/modules/3.6
python3 /ihome/arahimian/cah259/contagion/dump_spreading_data.py ${SLURM_ARRAY_TASK_ID}

