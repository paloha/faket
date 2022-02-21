import sys
import argparse
import os



from faket.deepfinder.clustering import Cluster
import faket.deepfinder.utils.common as cm
import faket.deepfinder.utils.objl as ol


def launch_clustering(test_tomogram, test_tomo_idx, num_epochs, label_map_path, out_path):
    
    # Input parameters:
    part_file_name = label_map_path + '/tomo' + str(test_tomo_idx) +\
                            '_' + test_tomogram + '_2021_'+ str(num_epochs) + 'epochs'
    path_labelmap = part_file_name + '_bin1_labelmap.mrc'
    cluster_radius = 5         # should correspond to average radius of target objects (in voxels)
    cluster_size_threshold = 1 # found objects smaller than this threshold are immediately discarded

    # Output parameter:
    part_out_file_name = out_path + '/tomo' + str(test_tomo_idx) +\
                            '_' + test_tomogram + '_2021_'+ str(num_epochs) + 'epoch'


    # Load data:
    labelmapB = cm.read_array(path_labelmap)

    # Initialize clustering task:
    clust = Cluster(clustRadius=5)
    clust.sizeThr = cluster_size_threshold

    # Launch clustering (result stored in objlist): can take some time (37min on i7 cpu)
    objlist = clust.launch(labelmapB)


    # The coordinates have been obtained from a binned (subsampled) volume, 
    # therefore coordinates have to be re-scaled in
    # order to compare to ground truth:
    objlist = ol.scale_coord(objlist, 2)

    # Then, we filter out particles that are too small, 
    # considered as false positives. As macromolecules have different
    # size, each class has its own size threshold. The thresholds have been determined on the validation set.
    lbl_list = [ 1,   2,  3,   4,  5,   6,   7,  8,  9, 10,  11, 12, 13, 14, 15 ]
    thr_list = [1000, 1, 1, 1, 1, 20, 20, 20, 1, 1, 1, 1, 1, 1000, 1 ]


    objlist_thr = ol.above_thr_per_class(objlist, lbl_list, thr_list)

    # Save object lists:
    ol.write_xml(objlist, part_out_file_name + '_bin1_objlist_raw.xml')
    ol.write_xml(objlist_thr, part_out_file_name + '_bin1_objlist_thr.xml')
