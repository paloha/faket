import sys
import os
import argparse



import numpy as np
import random
import tensorflow as tf

np.random.seed(123)
tf.random.set_seed(1234)
random.seed(123)

from training import Train
import utils.objl as ol
import produce_objl



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# parse path to config
parser = argparse.ArgumentParser()
parser.add_argument("--path_config", type=str, help="path to the config.ini of the experiment")
args = parser.parse_args()

path_config = args.path_config



# load settings
import configparser

config = configparser.ConfigParser()
config.read(path_config + 'config.ini')
train_tomos = config['training_setting']['training_tomograms'].split()
num_epochs = int(config['training_setting']['num_epochs'])
out_path = config['training_setting']['out_path']
gpu_no = config['training_setting']['gpu_no']
continue_training = bool(int(config['training_setting']['continue_training']))
continue_training_path = config['training_setting']['continue_training_path']


# set GPU id according to config
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no

# create out_path if it does not exist
if os.path.exists(out_path)==False:
        os.makedirs(out_path) 

# create path_data and path target
path_data = []
path_target = []
path_particle_locations = []
for tomo in train_tomos:
    path_reconstruction = 'data/shrec2021_extended_dataset/model_' + \
            tomo[0] + '/faket/reconstruction_' + tomo[2:] + '.mrc'
    path_data.append(path_reconstruction)
    
    path_class_mask = 'data/shrec2021_extended_dataset/model_' + \
            tomo[0] + '/faket/class_mask.mrc'
    path_target.append(path_class_mask)
    
    path_part_loc = 'data/shrec2021_extended_dataset/model_' + \
            tomo[0] + '/particle_locations.txt'
    path_particle_locations.append(path_part_loc)



    
# create objl_train according to path_data
objl_train = produce_objl.create_objl(path_particle_locations)
filename = 'deepfinder/test.xml'


# create objl_valid as in the original Deep-Finder repo
objl_valid = produce_objl.create_objl([path_particle_locations[-1]], int(train_tomos[-1][0]))


#Nclass = 13
Nclass = 16
dim_in = 56 # patch size

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = out_path # output path
trainer.h5_dset_name     = 'dataset' # if training data is stored as h5, you can specify the h5 dataset
trainer.batch_size       = 25
trainer.epochs           = num_epochs
trainer.Nvalid           = 10 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
trainer.class_weights = None # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

# Use following line if you want to resume a previous training session:
if continue_training:
    trainer.net.load_weights(continue_training_path)

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid)
