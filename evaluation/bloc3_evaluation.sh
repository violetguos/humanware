#!/bin/bash

set -e

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../ && pwd`
MASKRCNN_DIR=$ROOT_DIR/"maskrcnn-benchmark"
EVAL_DIR=$SECOND_STAGE_CODE/evaluation
INFERENCE_DIR=$MASKRCNN_DIR/inference/coco_humanware_test

TEST_INSTANCES=$TEST_DIR/instances_test.json
TEST_DIR=$MASKRCNN_DIR/datasets/test_dir

SECOND_STAGE_CODE=$ROOT_DIR/b2phut2/code
BEST_RCNN_MODEL=$ROOT_DIR/saved_models/"r_101_batch_size=2_iter=14400.pth"
BBOX_FILE=$INFERENCE_DIR/bbox.json
METADATA_FILENAME=$INFERENCE_DIR/metadata.pkl

function check_error(){
    exit_code=`echo $?`
    if [ $exit_code -ne 0 ]; then
        echo "$1"
        exit $exit_code
    fi
}


## Activate the conda environment
source activate humanware

### FILL OUT THE REST OF THE FILE ###

## build maskrcnn-bechnmark ##
echo "compiling maskrcnn..."
compile_stm=`cd $MASKRCNN_DIR && python setup.py build develop`
check_error "***** Unable to compile maskrcnn *****"

## run maskrcnn on test data ##

# cleanup stale files
rm -rf $INFERENCE_DIR \
    $TEST_DIR

# set up soft link to DATA_DIR
ln -s $DATA_DIR $TEST_DIR

echo "Running maskrcnn on test set..."
cd $MASKRCNN_DIR
python tools/test_net.py \
    --config-file configs/humanware_best.yaml
    MODEL.WEIGHT $BEST_RCNN_MODEL

check_error "Unable to run maskrcnn on test set"

## convert bbox.json to metadata.pkl ##
python $CURR_DIR/converter.py --bbox-file $BBOX_FILE --instance-file $TEST_INSTANCES --output-file $METADATA_FILENAME
check_error "Unable to generate metadata file"

## run predictions with bbox.json ##
cd $SECOND_STAGE_CODE/evaluation 
python eval.py --dataset_dir=$DATA_DIR --results_dir=$RESULTS_DIR --metadata_filename=$METADATA_FILENAME