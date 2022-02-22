import os
import sys
import argparse
from pathlib import Path
from segmentation import Segment
import utils.common as cm
import utils.smap as sm

# parse path to config
parser = argparse.ArgumentParser()
parser.add_argument("--test_tomo_path", type=str, help="path to tomograms to be segmented")
parser.add_argument("--test_tomogram", type=str, help="tomogram to be segmented", default="baseline")
parser.add_argument("--test_tomo_idx", type=int, help="folder index of test tomogram")
parser.add_argument("--num_epochs", type=str, help="number of epochs deep finder was trained")
parser.add_argument("--DF_weights_path", type=str, help="path to trained weights of deep finder")
parser.add_argument("--out_path", type=str, help="out path for the mrc file resulting from segmentation")
args = parser.parse_args()

model = args.DF_weights_path
num_epochs = args.num_epochs
test_tomo = args.test_tomogram

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# create output folders if they dont exist already
path = args.out_path
os.makedirs(path, exist_ok=True)

Nclass = 16
patch_size = 160 # must be multiple of 4

path_tomo_args = [f'{args.test_tomo_path}', 
                  f'model_{args.test_tomo_idx}/faket/reconstruction_',
                  f'{test_tomo}.mrc']

path_tomo = Path(''.join(path_tomo_args))
    
# Load data:
tomo = cm.read_array(str(path_tomo))

# Input parameters:
path_weights = Path(f'{args.DF_weights_path}/net_weights_epoch{num_epochs}.h5')

# Initialize segmentation task:
seg  = Segment(Ncl=Nclass, path_weights=str(path_weights), patch_size=patch_size)

# Segment tomogram:
scoremaps = seg.launch(tomo)

# Get labelmap from scoremaps:
labelmap  = sm.to_labelmap(scoremaps)

# Bin labelmap for the clustering step (saves up computation time):
scoremapsB = sm.bin(scoremaps)
labelmapB  = sm.to_labelmap(scoremapsB)

# Save labelmaps:
labelmap_file_name = Path(f'{path}/tomo9_{test_tomo}_2021_{num_epochs}epochs_labelmap.mrc')
labelmapB_file_name = Path(f'{path}/tomo9_{test_tomo}_2021_{num_epochs}epochs_bin1_labelmap.mrc')
cm.write_array(labelmap , str(labelmap_file_name))
cm.write_array(labelmapB, str(labelmapB_file_name))
