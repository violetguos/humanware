 #!/bin/bash

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../ && pwd`

B2_ROOT_DIR=$ROOT_DIR/src
SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293'
DATA_DIR=$SVHN_DIR/train
METADATA_FILENAME=$DATA_DIR/'avenue_train_metadata_split.pkl'

source /rap/jvb-000-aa/COURS2019/etudiants/common.env

PYTHONPATH=$B2_ROOT_DIR python -u $B2_ROOT_DIR/train.py \
    --dataset_dir=$DATA_DIR --metadata_filename=$METADATA_FILENAME \
    --results_dir=$ROOT_DIR/results --cfg=$B2_ROOT_DIR/config/retrain_b2.yaml
