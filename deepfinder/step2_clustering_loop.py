import sys

import os



from inference import Cluster
import utils.common as cm
import utils.objl as ol

import configparser

config = configparser.ConfigParser()
config.read('deepfinder/config_loc_class.ini')

# load config information
models = config['trained_DF_weights']['models'].split()
num_epochs_list = config['trained_DF_weights']['num_epochs'].split()
test_tomos = config['trained_DF_weights']['test_tomos'].split()


for test_tomo in test_tomos:
    for model, num_epochs in zip(models, num_epochs_list):
        

        # Input parameters:
        path_labelmap = 'data/deepfinder/localization_classification/'+ model  +\
                        '/tomo9_'+test_tomo+'_2021_'+ num_epochs +\
                        'epochs_bin1_labelmap.mrc'
        cluster_radius = 5         # should correspond to average radius of target objects (in voxels)
        cluster_size_threshold = 1 # found objects smaller than this threshold are immediately discarded

        # Output parameter:
        path_output = 'data/deepfinder/localization_classification/' + model + '/'


        # Load data:
        labelmapB = cm.read_array(path_labelmap)

        # Initialize clustering task:
        clust = Cluster(clustRadius=5)
        clust.sizeThr = cluster_size_threshold

        # Launch clustering (result stored in objlist): can take some time (37min on i7 cpu)
        objlist = clust.launch(labelmapB)


        # Post-process the object list for evaluation:

        # The coordinates have been obtained from a binned (subsampled) volume, therefore coordinates have to be re-scaled in
        # order to compare to ground truth:
        objlist = ol.scale_coord(objlist, 2)

        # Then, we filter out particles that are too small, considered as false positives. As macromolecules have different
        # size, each class has its own size threshold. The thresholds have been determined on the validation set.
        lbl_list = [ 1,   2,  3,   4,  5,   6,   7,  8,  9, 10,  11, 12, 13, 14, 15 ]
        thr_list = [1000, 1, 1, 1, 1, 20, 20, 20, 1, 1, 1, 1, 1, 1000, 1 ]


        objlist_thr = ol.above_thr_per_class(objlist, lbl_list, thr_list)

        # Save object lists:
        ol.write_xml(objlist    , path_output+'tomo9_'+test_tomo+'_2021_'+ str(num_epochs) +'epoch_bin1_objlist_raw.xml')
        ol.write_xml(objlist_thr, path_output+'tomo9_'+test_tomo+'_2021_'+ str(num_epochs) +'epoch_bin1_objlist_thr.xml')
