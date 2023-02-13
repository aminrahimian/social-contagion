#!/bin/bash
#SBATCH --job-name="measuring_strcutural_properties_real_networks"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=35
#SBATCH --partition=htc
#SBATCH --cluster=htc
#SBATCH --time=100:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-7

#SBATCH -o /ihome/arahimian/yil305/cai-contagion/output/output.out
#SBATCH -e /ihome/arahimian/yil305/cai-contagion/errors/errors.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YIL305@pitt.edu

module load python/intel-3.6

#module load sloan/python/modules/3.6
python3 /ihome/arahimian/yil305/social-contagion/measuring_strcutural_properties_real_networks.py ${SLURM_ARRAY_TASK_ID}

