import sys
import os
import argparse
import json



# parse path to config
parser = argparse.ArgumentParser()
parser.add_argument("--training_tomo_path", type=str, help="main path to the data set of tomograms")
parser.add_argument("--training_tomogram_ids", nargs='*', 
                    choices=['0','1','2','3','4','5','6','7','8'],
                    type=str, help="ids of tomogram within shrec based data set to be used for training of DF")
parser.add_argument("--training_tomograms", nargs='*', type=str,
                    choices=['baseline', 'content', 'noisy', 'styled', 'noiseless'],
                    help="type of tomograms to be used for training of DF")
parser.add_argument("--num_epochs", nargs=1, type=int, help="number of epochs to train DF")
parser.add_argument("--out_path", type=str, help="location where to store the weights of DF")
parser.add_argument("--save_every", type=int, nargs=1, 
                    help="regularly save DF weights after given amount of epochs have been completed", 
                    default=None)
parser.add_argument("--seed1", type=int, nargs=1, help="seed for numpy random", default=123)
parser.add_argument("--seed2", type=int, nargs=1, help="seed for tensorflow random", default=1234)
parser.add_argument("--continue_training_path", type=str, default=None,
                    help="path to DF weights for continuing training")
args = parser.parse_args()



import numpy as np
import random
import tensorflow as tf

np.random.seed(args.seed1[0])
tf.random.set_seed(args.seed2[0])
random.seed(args.seed1[0])

from training import Train
import utils.objl as ol
import produce_objl



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


if not len(args.training_tomogram_ids) == len(args.training_tomograms):
    raise AssertionError()

num_epochs = args.num_epochs[0]
out_path = args.out_path


# set GPU id according to config
#os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no

# create out_path if it does not exist
if os.path.exists(out_path)==False:
        os.makedirs(out_path) 

# create path_data and path target
path_data = []
path_target = []
path_particle_locations = []
for id, tomo in zip(args.training_tomogram_ids, args.training_tomograms):
    path_reconstruction = args.training_tomo_path + 'model_' + \
            id + '/faket/reconstruction_' + tomo + '.mrc'
    path_data.append(path_reconstruction)
    
    path_class_mask = args.training_tomo_path + 'model_' + \
            id + '/faket/class_mask.mrc'
    path_target.append(path_class_mask)
    
    path_part_loc = args.training_tomo_path + 'model_' + \
            id + '/particle_locations.txt'
    path_particle_locations.append(path_part_loc)

    
# create objl_train according to path_data
objl_train = produce_objl.create_objl(path_particle_locations)


# create objl_valid as in the original Deep-Finder repo
objl_valid = produce_objl.create_objl([path_particle_locations[-1]], int(args.training_tomogram_ids[-1][0]))


#Nclass = 13
Nclass = 16
dim_in = 56 # patch size

# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in)
trainer.path_out         = out_path # output path
trainer.h5_dset_name     = 'dataset' # if training data is stored as h5, you can specify the h5 dataset
trainer.batch_size       = 25
trainer.save_every       = args.save_every[0]
trainer.epochs           = num_epochs
trainer.Nvalid           = 10 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
trainer.class_weights = None # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

# Use following line if you want to resume a previous training session:
if args.continue_training_path is not None:
    trainer.net.load_weights(args.continue_training_path)

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid)

# write summary of training to out_path
summary = {
    "training_tomograms" : path_data,
    "num_epochs" : args.num_epochs[0]
}

with open(out_path + 'summary.json', 'w') as fl:
            json.dump(summary, fl, indent=4)