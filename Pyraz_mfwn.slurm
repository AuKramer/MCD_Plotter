#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mail-user=akramer9@buffalo.edu
#SBATCH --mail-type=all
#SBATCH --cluster=faculty
#SBATCH --partition=jochena
#SBATCH --qos=jochena
#SBATCH --job-name=Pyrim_mwfn
#SBATCH --output=Pyrim_mwfn.out

export SCRATCH_DIR=$SLURMTMPDIR
export NNODES=`srun -l hostname | uniq | wc -l`
export NPROCS=`srun -l hostname | wc -l`
export | grep SLURM

export LC_COLLATE=C
ulimit -s unlimited
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
export MODULEPATH=/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/compat/usr/lib64:$MODULEPATH
module load  gcccore/11.2.0
module load motif/2.3.8

./Multiwfn_noGUI_Pyraz Pyraz_cd_200_full_ext.fchk < elec_mwfn.in

mv transdipmom.txt transdipmom_elec_Pyrim.txt

./Multiwfn_noGUI_Pyraz Pyraz_cd_200_full_ext.fchk < mag_mwfn.in

mv transdipmom.txt transdipmom_mag_Pyrim.txt
