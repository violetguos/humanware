# Door Number Detection Project

This repository contains the code necessary for the door number detection
project.

The goal of the project is to help blind persons to find their way around by
making sure they are at the right house when they want for example visit a
friend or a family member, go to a specific store, etc.

In developing this project we must keep in mind the different constraints of
this application notably for the selection and development of the models we
will use like the execution time, online vs. offline, the memory usage (in the
case of a mobile application), etc.

## Types of training
Theres 2 types of use of our code. Both use the same arguments. Both support checkpointing.

### Simple training
You can train (or continue) a model with the train.py file

### Hyper-parameters search 
You can do hyper-parameters search  (or continue) with the hyper_param_train.py file

## Quick usage on Helios

To run the code on Helios, you can use the scripts in `scrips/helios/train_on_helios.sh`. 

You can run this directly from the login node using msub: 

`msub -A $GROUP_RAP -l feature=k80,nodes=1:gpus=1,walltime=2:00:00 train_on_helios.sh`

You can easily add this script to a `.pbs` file with your specific settings.

To change the data directories, you can modify the `train_on_helios.sh` script. To change configurations during training, use the `config/modular_model_config.yml` file. This contains tuneable options that can be useful.

To modify the models used, modify the appropriate model declaration in `trainer/trainer.py`.

## Best Model

See the report for more precisions about the best model

## Run the code interactively
For debugging purpose you might want to run your code interactively.

If you want to stop your code in a particular line you can add those
lines there: `import ipdb; ipdb.set_trace()`.
See [ipdb](https://pypi.org/project/ipdb/) for more informations.

## Data
For more information about the data used and its format, consult the `README`
in the `data/` directory.