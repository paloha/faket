# Either import the launch_clustering function from within a Python code
# or run this module as a script using `python3 -m faket.deepfinder.launch_clustering` 

import os
import sys
import json
import argparse
from .utils import objl as ol
from .utils import common as cm
from .clustering import Cluster
from os.path import join as pj


def launch_clustering(test_tomogram, test_tomo_idx, num_epochs, label_map_path, 
                      out_path, n_jobs=1, overwrite=False, only_apply_thresholding=False):

    identifier_fname = f'epoch{int(num_epochs):03d}_2021_model_{test_tomo_idx}_{test_tomogram}_bin2'
    
    # Input file name
    labelmap_path = pj(label_map_path, f'{identifier_fname}_labelmap.mrc')
    
    # Output file names
    raw_path = pj(out_path, f'{identifier_fname}_objlist_raw.xml')
    thr_path = pj(out_path, f'{identifier_fname}_objlist_thr.xml')
    
    # Log file names
    logpath = pj(out_path, 'logs', 'clustering')
    summary_fname = pj(logpath, f'{identifier_fname}.json')
    outlog = pj(logpath, f'{identifier_fname}.out')
    errlog = pj(logpath, f'{identifier_fname}.err')

    # Flow control - handling if one or both output files were already computed
    if os.path.exists(raw_path):
        if os.path.exists(thr_path):
            if not overwrite:
                print(f'Already computed! --overwrite not specified, so skipping: {labelmap_path}')
                return
        else:
            if not only_apply_thresholding and not overwrite:
                raise ValueError(f'File {thr_path} does not exit. Either use --overwrite or --only_apply_thresholding.')
    else:
        if only_apply_thresholding:
            raise ValueError(f'File {raw_path} does not exist. Can not apply thresholding only.')
        
        if os.path.exists(thr_path) and not overwrite:
            raise ValueError(f'File {thr_path} already exists. Use --overwrite.')
            
    if only_apply_thresholding and not os.path.exists(summary_fname):
        raise ValueError(f'File {summary_fname} does not exist. Do not use --only_apply_thresholding. Recompute whole clustering.')
    
    cluster_radius = 5         # should correspond to average radius of target objects (in voxels)
    cluster_size_threshold = 1 # found objects smaller than this threshold are immediately discarded
    
    # As macromolecules have different size, each class has its own size threshold (for removal).
    # The thresholds have been determined on the validation set.
    # background 0, 4V94 1, 4CR2 2, 1QVR 3, 1BXN 4, 3CF3 5, 1U6G 6, 3D2F 7, 2CG9 8, 
    # 3H84 9, 3GL1 10, 3QM1 11, 1S3X 12, 5MRC 13, vesicle 14, fiducial 15
    lbl_list = [1,    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,   15]
    thr_list = [1000, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1000, 1]  # FakET all to 1

    # Load data:
    labelmapB = cm.read_array(labelmap_path)
    
    if not only_apply_thresholding:
        if not os.path.exists(raw_path) or overwrite:
            
            # Logging clustering step ###########################
            os.makedirs(logpath, exist_ok=True)

            # Creating summary in log
            summary = {
                "labelmap_path": labelmap_path, 
                "cluster_radius" : cluster_radius, 
                "cluster_size_threshold": cluster_size_threshold, 
                "class_thresholds": dict(zip(lbl_list, thr_list)),
                "n_jobs": n_jobs}
            with open(summary_fname, 'w') as fsum:
                json.dump(summary, fsum, indent=4)

            # Redirect stdout to log
            fout = open(outlog, 'w')
            sys.stdout = fout

            # Redirect stderr to log
            ferr = open(errlog, 'w')
            sys.stderr = ferr
            #####################################################

            print("Initializing clustering task")
    
            # Initialize clustering task:
            clust = Cluster(clustRadius=5)
            clust.sizeThr = cluster_size_threshold

            # Launch clustering (result stored in objlist): can take some time (37min on i7 cpu)
            objlist = clust.launch(labelmapB, n_jobs=n_jobs)

            # The coordinates have been obtained from a binned (subsampled) volume, 
            # therefore coordinates have to be re-scaled in
            # order to compare to ground truth:
            objlist = ol.scale_coord(objlist, 2)
            
            # Save raw object list:
            ol.write_xml(objlist, raw_path)
            
            # Close log files & remove empty log files if any
            fout.close()
            ferr.close()
            for log in [outlog, errlog]:
                if os.path.isfile(log) and os.path.getsize(log) == 0:
                    os.remove(log)
    else:
        # Load raw_path from xml file instead of computing it
        objlist = ol.read_xml(raw_path)
        
        # Load summary_fname from json file for updating
        with open(summary_fname, 'r') as fsum:
            summary = json.load(fsum)
        
        # Update and save the summary file
        summary.update({'class_thresholds': dict(zip(lbl_list, thr_list))})
        with open(summary_fname, 'w') as fsum:
                json.dump(summary, fsum, indent=4)
        
    # Filtering out particles (false positives) that are too small (based on desired thresholds)
    objlist_thr = ol.above_thr_per_class(objlist, lbl_list, thr_list)

    # Save thresholded object lists:
    ol.write_xml(objlist_thr, thr_path)


if __name__ == '__main__':
    
    # Parse arguments
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
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="The number of jobs to use for the MeanShift computation. Computes each of the n_init runs in parallel.")
    parser.add_argument('--overwrite', action='store_true',  # If not provided, means False
                        help='If specified, overwrites previously computed results.')
    parser.add_argument('--only_apply_thresholding', action='store_true',  # If not provided, means False
                        help='If specified, loads raw objlist and applies thresholds.')
   
    args = parser.parse_args()
    
    launch_clustering(args.test_tomogram, args.test_tomo_idx, args.num_epochs, 
                      args.label_map_path, args.out_path, args.n_jobs, args.overwrite, 
                      args.only_apply_thresholding)
