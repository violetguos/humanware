
#!/bin/bash
#PBS -S /bin/bash
#PBS -N TACHE       # Nom de la tâche
#PBS -A colosse-users  # Identifiant Rap; ID
#PBS -l feature=k80
#PBS -l walltime=8:00:00    # Durée en secondes
#PBS -l nodes=1:gpus=1  # Nombre de noeuds.
# do not execute on login nodes
module --force purge

#XXXPBXXXS -l advres=MILA2019
PATH=$PATH:/opt/software/singularity-3.0/bin/

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

# Singularity options 
IMAGE=/rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg
RAP=/rap/jvb-000-aa/COURS2019/etudiants/$USER

source activate humanware
s_exec python maskrcnn-benchmark/tools/train_net.py --config-file maskrcnn-benchmark/configs/humanware_e2e_faster_rcnn_R_101_FPN_1x.yaml