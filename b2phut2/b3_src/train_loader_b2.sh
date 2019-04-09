 #!/bin/bash


export ROOT_DIR=$HOME/humanware/b2phut2
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293'
export DATA_DIR=$SVHN_DIR/train
export TMP_DATA_DIR=$DATA_DIR
export TMP_RESULTS_DIR=$DATA_DIR
export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293/train/avenue_train_metadata_split.pkl'

source /rap/jvb-000-aa/COURS2019/etudiants/common.env

python -u $HOME/humanware/b2phut2/code/test_loader_new_data.py \
    --dataset_dir=$DATA_DIR --metadata_filename=$METADATA_FILENAME \
    --results_dir=$HOME/humanware/results --cfg=$HOME/humanware/b2phut2/code/config/test_b2.yml