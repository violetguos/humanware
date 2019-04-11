#!/bin/bash
#PBS -S /bin/bash
#PBS -A colosse-users  # Identifiant Rap; ID
#PBS -l walltime=8:00:00    # Durée en secondes
#PBS -l nodes=1:gpus=1  # Nombre de noeuds.
# do not execute on login nodes

#xcvBS -l advres=MILA2019
module --force purge

PATH=$PATH:/opt/software/singularity-3.0/bin/

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"
export PATH=$PATH:/opt/software/singularity-3.0/bin/

s_exec bash -ex train_loader_b2.sh
