import sys

from inference import Segment
import utils.common as cm
import utils.smap as sm


import os


import configparser

config = configparser.ConfigParser()
config.read('deepfinder/config_loc_class.ini')

# load config information
models = config['trained_DF_weights']['models'].split()
num_epochs_list = config['trained_DF_weights']['num_epochs'].split()
test_tomos = config['trained_DF_weights']['test_tomos'].split()
GPU_no = config['GPU_no']['GPU_no']



os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_no

# create output folders if they dont exist already
for model in models:
    path = 'data/deepfinder/localization_classification/'+ model
    if os.path.exists(path)==False:
        os.makedirs(path) 

#sys.exit()


Nclass       = 16
patch_size   = 160 # must be multiple of 4


for test_tomo in test_tomos:

    path_tomo = 'data/shrec2021_extended_dataset/model_9/faket/reconstruction_'+test_tomo+'.mrc'
    
    # Load data:
    tomo = cm.read_array(path_tomo)
    
    for model, num_epochs in zip(models, num_epochs_list):

        # Input parameters:
        path_weights = 'data/deepfinder/trained_models/'+ model + \
                        '/net_weights_epoch'+ str(num_epochs)  +'.h5'

        # Output parameter:
        path_output = 'data/deepfinder/localization_classification/'+ model




        # Initialize segmentation task:
        seg  = Segment(Ncl=Nclass, path_weights=path_weights, patch_size=patch_size)

        # Segment tomogram:
        scoremaps = seg.launch(tomo)

        # Get labelmap from scoremaps:
        labelmap  = sm.to_labelmap(scoremaps)

        # Bin labelmap for the clustering step (saves up computation time):
        scoremapsB = sm.bin(scoremaps)
        labelmapB  = sm.to_labelmap(scoremapsB)

        # Save labelmaps:
        cm.write_array(labelmap , path_output+'/tomo9_'+test_tomo+'_2021_'+ str(num_epochs)  +'epochs_labelmap.mrc')
        cm.write_array(labelmapB, path_output+'/tomo9_'+test_tomo+'_2021_'+ str(num_epochs)  +'epochs_bin1_labelmap.mrc')



