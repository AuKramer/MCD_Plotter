#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=akramer9@buffalo.edu
#SBATCH --mail-type=all
#SBATCH --cluster=faculty
#SBATCH --partition=jochena
#SBATCH --qos=jochena
#SBATCH --job-name=Pyraz_cd_200_full
#SBATCH --output=Pyraz_cd_200_full.out

export SCRATCH_DIR=$SLURMTMPDIR
export NNODES=`srun -l hostname | uniq | wc -l`
export NPROCS=`srun -l hostname | wc -l`
export | grep SLURM

export LC_COLLATE=C
ulimit -s unlimited

# prior to Dec 2023 we used the next 7 lines
#module use /util/academic/gaussian/g16.A03/modules
#module load g16
#module list
#echo "Sourcing the g16.profile"
#. $g16root/g16/bsd/g16.profile
#echo "G16 variables from sourcing"
#export | grep -i g16

# from Dec. 2023, we use the following 3 lines
# (make sure $HOME/.ccr/modulepaths contains a line
# with the path /projects/academic/jochena/software/easybuild ):
module load gaussian/16.C.02-AVX2
module list
which g16


export GAUSS_SCRDIR=$SLURMTMPDIR
export LC_COLLATE=C

cd $SLURMTMPDIR

g16 << eor
%NProc=$NPROCS
%Mem=8000Mb
%chk=/projects/academic/jochena/akramer9/Favereau_MCD_Collab/MCD_Tests/Pyraz_cd_200_full.chk
#P B3LYP/6-311++G(d,p) TD=(NStates=200) IOp(9/40=5)  gfinput

Pyraz_cd_300_full

0 1
C          1.19200       -0.59648        0.00000
N          1.19397        0.73769        0.00000
C         -0.00007        1.33292        0.00000
C          0.00000       -1.33288        0.00000
C         -1.19200        0.59641        0.00000
N         -1.19416       -0.73781        0.00000
H          2.16611       -1.10214        0.00000
H         -0.01641        2.43033        0.00000
H         -2.16521        1.10265        0.00000
H          0.01727       -2.42974        0.00000

eor

#cp -r * ${SLURM_SUBMIT_DIR}/results/
