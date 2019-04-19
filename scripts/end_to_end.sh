#!/bin/bash

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../ && pwd`
EVAL_DIR=$ROOT_DIR/evaluation
RESULTS_DIR=$ROOT_DIR/results

RCNN_BOX_DIR=$ROOT_DIR/data
RCNN_MODEL_NAME=r_101_14400
MODEL_DIR=$RCNN_BOX_DIR/$RCNN_MODEL_NAME/coco_humanware_v1_1553272293_val
BBOX_FILE=$MODEL_DIR/bbox.json
METADATA_FILENAME=$MODEL_DIR/metadata.pkl

ELEM_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293'
VALID_DATA_DIR=$ELEM_DIR/valid
VALID_INSTANCES=$VALID_DATA_DIR/instances_valid.json
VALID_METADATA_FILENAME=$VALID_DATA_DIR/'avenue_valid_metadata_split.pkl'

SAVED_MODELS=$ROOT_DIR/saved_models/RESNET34_val_acc_519
SECOND_STAGE_MODEL=$SAVED_MODELS/checkpoint_0.519.pth
SECOND_STAGE_CONFIG=$SAVED_MODELS/config.yml

function check_error(){
    exit_code=`echo $?`
    if [ $exit_code -ne 0 ]; then
        echo "$1"
        exit $exit_code
    fi
}

## convert bbox.json to metadata.pkl ##
python $EVAL_DIR/converter.py \
    --bbox-file $BBOX_FILE \
    --instance-file $VALID_INSTANCES \
    --output-file $METADATA_FILENAME \
    --original-metadata $VALID_METADATA_FILENAME
check_error "Unable to generate metadata file"

## run eval ##
cd $EVAL_DIR
python eval.py \
    --dataset_dir=$VALID_DATA_DIR \
    --results_dir=$RESULTS_DIR \
    --metadata_filename=$METADATA_FILENAME \
    --model_path=$SECOND_STAGE_MODEL \
    --model_cfg=$SECOND_STAGE_CONFIG
