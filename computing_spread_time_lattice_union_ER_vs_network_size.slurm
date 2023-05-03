#!/bin/bash
#SBATCH --job-name="computing_spread_time_c_1_union_ER_vs_network_size"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=htc
#SBATCH --cluster=htc
#SBATCH --time=20:30:00
#SBATCH --mem-per-cpu=4G

#SBATCH -o /ihome/arahimian/yil305/social-contagion-master/output/output.out
#SBATCH -e /ihome/arahimian/yil305/social-contagion-master/errors/errors.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YIL305@pitt.edu

module load python/intel-3.6

#module load sloan/python/modules/3.6
python3 /ihome/arahimian/yil305/social-contagion-master/computing_spread_time_lattice_union_ER_vs_network_size.py

