#!/bin/bash

set -e

export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176/lib64/:$LD_LIBRARY_PATH

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR=`cd $CURR_DIR/../ && pwd`
MASKRCNN_DIR=$ROOT_DIR/"maskrcnn-benchmark"
INFERENCE_DIR=$MASKRCNN_DIR/inference/coco_humanware_test

TEST_DIR=$MASKRCNN_DIR/datasets/test_dir
TEST_INSTANCES=$TEST_DIR/instances_test.json

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
compile_stm=`cd $MASKRCNN_DIR && python setup.py build develop --user`
check_error "***** Unable to compile maskrcnn *****"

## run maskrcnn on test data ##

# cleanup stale files
rm -rf $INFERENCE_DIR \
    $TEST_DIR \
    $MASKRCNN_DIR/model_final.pth \
    $MASKRCNN_DIR/last_checkpoint

# set up soft link to DATA_DIR
ln -s $DATA_DIR $TEST_DIR

# set up soft link to best model
ln -s $BEST_RCNN_MODEL $MASKRCNN_DIR/model_final.pth

echo "Running maskrcnn on test set..."
cd $MASKRCNN_DIR
python tools/test_net.py \
    --config-file configs/humanware_best.yaml

check_error "Unable to run maskrcnn on test set"

## convert bbox.json to metadata.pkl ##
python $CURR_DIR/converter.py --bbox-file $BBOX_FILE --instance-file $TEST_INSTANCES --output-file $METADATA_FILENAME
check_error "Unable to generate metadata file"

## run predictions with on test data with metadata.pkl ##
cd $CURR_DIR
python eval.py --dataset_dir=$DATA_DIR --results_dir=$RESULTS_DIR --metadata_filename=$METADATA_FILENAME