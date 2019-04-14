 #!/bin/bash


CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../../../ && pwd`

B2_ROOT_DIR=$ROOT_DIR/humanware/b2phut2/code
SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293'
TRAIN_DATA_DIR=$SVHN_DIR/train
TRAIN_METADATA_FILENAME=$TRAIN_DATA_DIR/'avenue_train_metadata_split.pkl'
VALIDATION_DIR=$SVHN_DIR/valid
VALIDATION_METADATA_FILENAME=$VALIDATION_DIR/'avenue_valid_metadata_split.pkl'
source /rap/jvb-000-aa/COURS2019/etudiants/common.env

PYTHONPATH=$B2_ROOT_DIR python -u $B2_ROOT_DIR/train_loader_new_data.py \
    --dataset_dir=$TRAIN_DATA_DIR --metadata_filename=$TRAIN_METADATA_FILENAME \
    --results_dir=$ROOT_DIR/humanware/results --cfg=$B2_ROOT_DIR/config/tune_smart.yaml \
    --extra_dataset_dir=$VALIDATION_DIR --extra_metadata_filename=$VALIDATION_METADATA_FILENAME