import os
import sys
import json
import argparse
import utils.smap as sm
import utils.common as cm
from os.path import join as pj
from segmentation import Segment

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tomo_path", type=str, help="path to tomograms to be segmented")
    parser.add_argument("--test_tomogram", type=str, help="tomogram to be segmented", default="baseline")
    parser.add_argument("--test_tomo_idx", type=int, help="folder index of test tomogram")
    parser.add_argument("--num_epochs", type=str, help="number of epochs deep finder was trained")
    parser.add_argument("--DF_weights_path", type=str, help="path to trained weights of deep finder")
    parser.add_argument("--out_path", type=str, help="out path for the mrc file resulting from segmentation")
    parser.add_argument('--overwrite', action='store_true',  # If not provided, means False
                        help='If specified, overwrites previously computed results.')
    args = parser.parse_args()

    # Tensorflow specific env variables
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Create output folders if they dont exist already
    os.makedirs(args.out_path, exist_ok=True)

    # Output file names
    identifier_fname = f'epoch{int(args.num_epochs):03d}_2021_model_{args.test_tomo_idx}_{args.test_tomogram}'
    labelmap_file_name = pj(args.out_path, f'{identifier_fname}_bin1_labelmap.mrc')  # No binning
    labelmapB_file_name = pj(args.out_path, f'{identifier_fname}_bin2_labelmap.mrc')  # 2x binned
    
    if os.path.exists(labelmap_file_name) and not args.overwrite:
        print(f'Already computed! --overwrite not specified, so skipping: {labelmap_file_name}')
        exit(0)  # Success return code

    Nclass = 16
    patch_size = 160 # must be multiple of 4

    # Load data:
    path_tomo = pj(args.test_tomo_path, f'model_{args.test_tomo_idx}', 
                  'faket', f'reconstruction_{args.test_tomogram}.mrc')
    tomo = cm.read_array(str(path_tomo))

    # Model weights:
    path_weights = pj(f'{args.DF_weights_path}', f'epoch{int(args.num_epochs):03d}_weights.h5')

    # Logging ###########################################
    logpath = pj(args.out_path, 'logs', 'segmentation')
    logname = identifier_fname
    os.makedirs(logpath, exist_ok=True)

    # Creating summary in log
    summary = {
        "tomogram" : path_tomo, "model": path_weights, 
        "Nclass": Nclass, "patch_size": patch_size}
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

    # Initialize segmentation task:
    seg = Segment(Ncl=Nclass, path_weights=str(path_weights), patch_size=patch_size)

    # Segment tomogram:
    scoremaps = seg.launch(tomo)

    # Get labelmap from scoremaps:
    labelmap  = sm.to_labelmap(scoremaps)

    # Bin labelmap for the clustering step (saves up computation time):
    scoremapsB = sm.bin(scoremaps)
    labelmapB  = sm.to_labelmap(scoremapsB)

    # Save labelmaps:
    cm.write_array(labelmap , labelmap_file_name)
    cm.write_array(labelmapB, labelmapB_file_name)

    # Close log files & remove empty log files if any
    fout.close()
    # ferr.close()
    for log in [outlog, errlog]:
        if os.path.isfile(log) and os.path.getsize(log) == 0:
            os.remove(log)
