#!/bin/bash
#SBATCH --job-name="computing_spread_time_c1_union_ER_w_rho"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --partition=opa
#SBATCH --cluster=mpi
#SBATCH --mail-user=CAH259@pitt.edu.
#SBATCH --mail-type=END,FAIL #
#SBATCH --time=3-00:00

#SBATCH -o /ihome/arahimian/cah259/contagion/computing_spread_time_c1_union_ER_w_rho.out
#SBATCH -e /ihome/arahimian/cah259/contagion/computing_spread_time_c1_union_ER_w_rho.err


cd /ihome/arahimian/cah259/contagion/

module load python/anaconda3.6-5.2.0 computing_spread_time_c1_union_ER_w_rho.py



