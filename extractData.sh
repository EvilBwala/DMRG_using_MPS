#!/bin/bash
#PBS -N Extract4
#PBS -j oe
#PBS -l nodes=1:ppn=24:cpu24a
#PBS -l walltime=72:00:00
#PBS -m abe
#PBS -M agnish2015@gmail.com

module load compiler/intel/2018
module load conda/ananconda3

export I_MPI_PIN_PROCESSOR_LIST=0-23
export I_MPI_FABRICS=shm:tmi
export I_MPI_FALLBACK=0

cd $PBS_O_WORKDIR
echo "My nodefile is $PBS_NODEFILE"
cat $PBS_NODEFILE

python Data_extraction.py > logfile.$PBS_JOBID 2>&1
#time python Recombination.py > logfile.$PBS_JOBID 2>&1

exit
