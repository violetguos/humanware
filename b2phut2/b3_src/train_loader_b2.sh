 #!/bin/bash


export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176/lib64/:$LD_LIBRARY_PATH
export ROOT_DIR='$HOME/humanware/b2phut2'
export SVHN_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/Humanware_v1_1553272293'
export DATA_DIR=$SVHN_DIR/train
export TMP_DATA_DIR=$DATA_DIR
export TMP_RESULTS_DIR=$DATA_DIR
export METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_metadata.pkl'

source /rap/jvb-000-aa/COURS2019/etudiants/common.env


echo $HOME

echo $PATH

# if [ ! -f $SVHN_DIR'/train.tar.gz' ]; then
if [ 1 -eq 0 ]; then
    echo "Downloading files for the training set!"
    wget -P $SVHN_DIR http://ufldl.stanford.edu/housenumbers/train.tar.gz
fi

# if [ ! -d $TMP_DATA_DIR ]; then
if [ 1 -eq 0 ]; then
    echo "Extracting Files to " $TMP_DATA_DIR
    cp $DATA_DIR'/train.tar.gz' $TMP_DATA_DIR
    tar -xzf $TMP_DATA_DIR'/train.tar.gz' -C $TMP_DATA_DIR
    echo "Extraction finished!"

else
    echo "Train files already present"
fi

python /home/user50/humanware/b2phut2/code/test_loader_new_data.py \
    --dataset_dir=$TMP_DATA_DIR --metadata_filename=$METADATA_FILENAME \
    --results_dir=/home/user50/humanware/results --cfg /home/user50/humanware/b2phut2/code/config/test_b2.yml


# source /rap/jvb-000-aa/COURS2019/etudiants/common.env

# s_exec python $ROOT_DIR'/train.py'  --dataset_dir=$TMP_DATA_DIR --metadata_filename=$METADATA_FILENAME --results_dir=$ROOT_DIR/results --cfg $ROOT_DIR/config/modular_model_config.yml

# echo "Copying files to local hard drive..."
# cp -r $TMP_RESULTS_DIR $ROOT_DIR

# echo "Cleaning up data and results..."
# rm -r $TMP_DATA_DIR
# rm -r $TMP_RESULTS_DIR