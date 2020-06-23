#!/bin/bash
#SBATCH --job-name="measuring_spread_time_real_networks"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=4-00:00:00
#SBATCH --array=1-17


#SBATCH -o /home/clintonr/contagion/measuring_spread_time_real_networks.out
#SBATCH -e /home/clintonr/contagion/measure_spread_time_real_networks.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=clintonr@mit.edu

module load python/3.7.3
module load sloan/python/modules/3.7

python3 /home/clintonr/contagion/measuring_spread_time_real_networks.py ${SLURM_ARRAY_TASK_ID}
