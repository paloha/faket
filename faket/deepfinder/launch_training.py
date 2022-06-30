import os
import sys
import random
import argparse
import numpy as np
import produce_objl
import tensorflow as tf
import utils.objl as ol
from pathlib import Path
from training import Train
from os.path import join as pj
from os.path import splitext, basename

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_tomo_path", type=str, help="main path to the data set of tomograms")
    parser.add_argument("--training_tomogram_ids", nargs='*', 
                        choices=['0','1','2','3','4','5','6','7','8'],
                        type=str, help="ids of tomogram within shrec based data set to be used for training of DF")
    parser.add_argument("--training_tomograms", nargs='*', type=str,
                        # choices=['baseline', 'content', 'noisy', 'styled', 'noiseless'],
                        help="type of tomograms to be used for training of DF")
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train DF")
    parser.add_argument("--out_path", type=str, help="location where to store the weights of DF")
    parser.add_argument("--save_every", type=int, 
                        help="regularly save DF weights after given amount of epochs have been completed", 
                        default=None)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    parser.add_argument("--continue_training_path", type=str, default=None,
                        help="Path to DF weights for continuing training. If path is a dir, use the last weights sorted alphabetically.")
    args = parser.parse_args()
    
    # Tensorflow specific env variables
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if not len(args.training_tomogram_ids) == len(args.training_tomograms):
        raise AssertionError()

    num_epochs = args.num_epochs
    out_path = args.out_path

    # create out_path if it does not exist
    if os.path.exists(out_path)==False:
            os.makedirs(out_path) 

    # create path_data and path target
    path_data = []
    path_target = []
    path_particle_locations = []
    for N, tomo in zip(args.training_tomogram_ids, args.training_tomograms):
        path_reconstruction = Path(f'{args.training_tomo_path}model_{N}/faket/reconstruction_{tomo}.mrc')
        path_data.append(str(path_reconstruction))

        path_class_mask = Path(f'{args.training_tomo_path}model_{N}/faket/class_mask.mrc')
        path_target.append(str(path_class_mask))

        path_part_loc = Path(f'{args.training_tomo_path}model_{N}/particle_locations.txt')
        path_particle_locations.append(str(path_part_loc))

    Nclass = 16
    dim_in = 56 # patch size

    # Initialize training task:
    trainer = Train(Ncl=Nclass, dim_in=dim_in)
    trainer.path_out = Path(out_path) # output path
    trainer.h5_dset_name = 'dataset' # if training data is stored as h5, you can specify the h5 dataset
    trainer.batch_size = 25
    trainer.save_every = args.save_every
    trainer.epochs = num_epochs
    trainer.Nvalid = 10 # steps per validation
    trainer.flag_direct_read = False
    trainer.flag_batch_bootstrap = True
    trainer.Lrnd = 13 # random shifts when sampling patches (data augmentation)
    trainer.class_weights = None # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0
    trainer.seed = args.seed

    # create objl_train according to path_data
    objl_train = produce_objl.create_objl(path_particle_locations)

    # create objl_valid as in the original Deep-Finder repo
    objl_valid = produce_objl.create_objl([path_particle_locations[-1]], int(args.training_tomogram_ids[-1][0]))

    # Use following line if you want to resume a previous training session:
    if args.continue_training_path is not None:
        from losses import tversky_loss
        from tensorflow.keras.models import load_model
        
        if os.path.isdir(args.continue_training_path):
            weights_files = [x for x in os.listdir(args.continue_training_path) if x.endswith('_weights.h5')]
            if (weights_files) == 0:
                raise ValueError(f'No epoch***_weights.h5 file found in dir {args.continue_training_path}')
            args.continue_training_path = pj(args.continue_training_path, sorted(weights_files)[-1])
        
        trainer.net = load_model(args.continue_training_path,
                                 custom_objects={'tversky_loss': tversky_loss})

        # Get the last epoch number from the epoch***_weights.h5 filename
        trainer.restart_from_epoch = \
            int(splitext(basename(args.continue_training_path))[0].split('_')[0][len('epoch'):]) + 1

        # Original DeepFinder code loaded just weights, which is incorrect
        # because it does not load the optimizer state.
        # trainer.net.load_weights(args.continue_training_path)

    # Finally, launch the training procedure:
    trainer.launch(path_data, path_target, objl_train, objl_valid)