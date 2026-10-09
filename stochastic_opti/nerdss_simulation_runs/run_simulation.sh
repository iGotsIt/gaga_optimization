#!/bin/bash
#SBATCH --job-name=nerdss_opti
#SBATCH --output=run_logs/nerdss_runs_%j.sout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=parallel
#SBATCH --account=mjohn218
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jzhou59@jhu.edu

source ~/.bashrc
conda activate pytorch_env
ml gsl
cd /home/jzhou59/data_mjohn218/jzhou59/gaga_optimization/stochastic_opti/nerdss_simulation_runs
python build_parms_file_and_run.py single