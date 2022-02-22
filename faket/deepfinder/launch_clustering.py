import sys
import argparse
import os
from pathlib import Path

from clustering import Cluster
import utils.common as cm
import utils.objl as ol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tomogram", type=str, 
                        help="tomogram to be segmented", default="baseline")
    parser.add_argument("--test_tomo_idx", type=str, 
                        help="folder index of test tomogram")
    parser.add_argument("--num_epochs", type=str, 
                        help="number of epochs deep finder was trained")
    parser.add_argument("--label_map_path", type=str, 
                        help="path to the folder of the label map that results from segmentation")
    parser.add_argument("--out_path", type=str, 
                        help="out path for the xml files resulting from clustering")
    args = parser.parse_args()

    # Input parameters:
    file_name_args = [f'{args.label_map_path}/tomo',
                      f'{args.test_tomo_idx}_',
                      f'{args.test_tomogram}_2021_',
                      f'{args.num_epochs}epochs', 
                      f'_bin1_labelmap.mrc']
    path_labelmap = Path(''.join(file_name_args))
    
    #path_labelmap = Path(f'{part_file_name}_bin1_labelmap.mrc')
    cluster_radius = 5         # should correspond to average radius of target objects (in voxels)
    cluster_size_threshold = 1 # found objects smaller than this threshold are immediately discarded

    # Output parameter:
    part_out_file_name_args = [f'{args.out_path}/tomo',
                               f'{str(args.test_tomo_idx)}_',
                               f'{args.test_tomogram}_2021_',
                               f'{args.num_epochs}epoch_bin1_objlists']
    out_file_name_thr = Path(''.join(part_out_file_name_args) + '_thr.xml')
    out_file_name_raw = Path(''.join(part_out_file_name_args) + '_raw.xml')
    

    # Load data:
    labelmapB = cm.read_array(str(path_labelmap))

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
    # size, each class has its own size threshold.
    # The thresholds have been determined on the validation set.
    lbl_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    thr_list = [1000, 1, 1, 1, 1, 20, 20, 20, 1, 1, 1, 1, 1, 1000, 1 ]

    objlist_thr = ol.above_thr_per_class(objlist, lbl_list, thr_list)

    # Save object lists:
    ol.write_xml(objlist, str(out_file_name_raw))
    ol.write_xml(objlist_thr, str(out_file_name_thr))
