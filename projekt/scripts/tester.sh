#!/usr/bin/env bash
#SBATCH --account=DD-24-108
#SBATCH --job-name=PPP_PROJ01_MPI
#SBATCH -p qcpu
#SBATCH -t 05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --distribution=block:block:block,Pack


## Author: Jakub Vlk
source load_modules.sh

STDOUT_FILEm1="m1_tester.txt"
STDERR_FILEm1="m1_tester_err.txt"
STDERR_FILEm2="m2_tester.txt"
STDERR_FILEm2="m2_tester_err.txt"

STDOUT_FILEm1g="m1g_tester.txt"
STDERR_FILEm1g="m1g_tester_err.txt"
STDOUT_FILEm2g="m2g_tester.txt"
STDERR_FILEm2g="m2g_tester_err.txt"
BINARY_PATH="../build_prof/ppp_proj01"

# Clear the stdout and stderr files
rm -f $STDOUT_FILE $STDERR_FILE

export OMP_NUM_THREADS=8

srun $BINARY_PATH -n 1000000 -t $OMP_NUM_THREADS -m 1 -i input_data_4096.h5 >> $STDOUT_FILEm1 2>> $STDERR_FILEm1
srun $BINARY_PATH -n 1000000 -t $OMP_NUM_THREADS -m 2 -i input_data_4096.h5 >> $STDERR_FILEm2 2>> $STDERR_FILEm2

srun $BINARY_PATH -g -n 1000000 -t $OMP_NUM_THREADS -m 1 -i input_data_4096.h5 >> $STDOUT_FILEm1g 2>> $STDERR_FILEm1g
srun $BINARY_PATH -g -n 1000000 -t $OMP_NUM_THREADS -m 2 -i input_data_4096.h5 >> $STDOUT_FILEm2g 2>> $STDERR_FILEm2g

