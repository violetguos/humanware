 #!/bin/bash


CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../../../../ && pwd`

export B2_ROOT_DIR=$ROOT_DIR/humanware/b2phut2/code
export ELEM_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293'
export DATA_DIR=$ELEM_DIR/train
export TMP_DATA_DIR=$DATA_DIR
export TMP_RESULTS_DIR=$DATA_DIR
export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293/train/avenue_train_metadata_split.pkl'

source /rap/jvb-000-aa/COURS2019/etudiants/common.env

PYTHONPATH=$B2_ROOT_DIR python -u $B2_ROOT_DIR/hyper_param_train.py \
    --dataset_dir=$DATA_DIR --metadata_filename=$METADATA_FILENAME \
    --results_dir=$HOME/humanware/results --cfg=$B2_ROOT_DIR/config/tune.yaml
