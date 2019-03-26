export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx/Compiler/intel2016.4/cuda/9.0.176/lib64/:$LD_LIBRARY_PATH
source activate humanware
python maskrcnn-benchmark/tools/train_net.py \
    --config-file maskrcnn-benchmark/configs/humanware_e2e_faster_rcnn_R_101_FPN_1x.yaml