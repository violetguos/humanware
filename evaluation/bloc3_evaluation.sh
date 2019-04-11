#!/bin/bash

set -e

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../ && pwd`
MASKRCNN_DIR=$ROOT_DIR/"maskrcnn-benchmark"
INFERENCE_DIR=$MASKRCNN_DIR/inference/coco_humanware_test
TEST_DIR=$MASKRCNN_DIR/datasets/test_dir
BEST_RCNN_MODEL=$ROOT_DIR/saved_models/"r_101_batch_size=2_iter=14400.pth"


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

## run predictions with bbox.json ##


# The final results should be saved to $RESULTS_DIR
# Refer to evaluation_instructions.md for more information