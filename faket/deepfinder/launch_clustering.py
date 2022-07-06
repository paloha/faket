import os
import sys
import json
import argparse
import utils.objl as ol
import utils.common as cm
from clustering import Cluster
from os.path import join as pj

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
    
    args = parser.parse_args()

    identifier_fname = f'epoch{int(args.num_epochs):03d}_2021_model_{args.test_tomo_idx}_{args.test_tomogram}_bin2'
    
    # Input file name
    labelmap_path = pj(args.label_map_path, f'{identifier_fname}_labelmap.mrc')
    
    # Output file names
    thr_path = pj(args.out_path, f'{identifier_fname}_objlist_thr.xml')
    raw_path = pj(args.out_path, f'{identifier_fname}_objlist_raw.xml')
    
    if os.path.exists(raw_path) and os.path.exists(thr_path):
        if not args.overwrite:
            print(f'Already computed! --overwrite not specified, so skipping: {labelmap_path}')
            exit(0)  # Success return code
    
    cluster_radius = 5         # should correspond to average radius of target objects (in voxels)
    cluster_size_threshold = 1 # found objects smaller than this threshold are immediately discarded
    
    # As macromolecules have different size, each class has its own size threshold (for removal).
    # The thresholds have been determined on the validation set.
    lbl_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    thr_list = [1000, 1, 1, 1, 1, 20, 20, 20, 1, 1, 1, 1, 1, 1000, 1]

    # Load data:
    labelmapB = cm.read_array(labelmap_path)

    # Logging ###########################################
    logpath = pj(args.out_path, 'logs', 'clustering')
    logname = identifier_fname
    os.makedirs(logpath, exist_ok=True)

    # Creating summary in log
    summary = {
        "labelmap_path": labelmap_path, 
        "cluster_radius" : cluster_radius, 
        "cluster_size_threshold": cluster_size_threshold, 
        "class_thresholds": dict(zip(lbl_list, thr_list)),
        "n_jobs": args.n_jobs}
    with open(pj(logpath, f'{logname}.json'), 'w') as fsum:
        json.dump(summary, fsum, indent=4)

    # Redirect stdout to log
    outlog = pj(logpath, f'{logname}.out')
    fout = open(outlog, 'w')
    sys.stdout = fout

    # Redirect stderr to log
    errlog = pj(logpath, f'{logname}.err')
    ferr = open(errlog, 'w')
    sys.stderr = ferr
    #####################################################
    
    # Initialize clustering task:
    clust = Cluster(clustRadius=5)
    clust.sizeThr = cluster_size_threshold

    # Launch clustering (result stored in objlist): can take some time (37min on i7 cpu)
    objlist = clust.launch(labelmapB, n_jobs=args.n_jobs)

    # The coordinates have been obtained from a binned (subsampled) volume, 
    # therefore coordinates have to be re-scaled in
    # order to compare to ground truth:
    objlist = ol.scale_coord(objlist, 2)

    # Filtering out particles (false positives) that are too small (based on desired thresholds)
    objlist_thr = ol.above_thr_per_class(objlist, lbl_list, thr_list)

    # Save object lists:
    ol.write_xml(objlist, raw_path)
    ol.write_xml(objlist_thr, thr_path)
    
    # Close log files & remove empty log files if any
    fout.close()
    ferr.close()
    for log in [outlog, errlog]:
        if os.path.isfile(log) and os.path.getsize(log) == 0:
            os.remove(log)
