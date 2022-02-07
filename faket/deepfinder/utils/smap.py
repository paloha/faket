# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import numpy as np
import h5py

from . import common as cm

def bin(scoremaps):
    """Subsamples the scoremaps by a factor 2. Subsampling is performed by averaging voxel values in 2x2x2 tiles.

    Args:
        scoremaps (4D numpy array): array with index order [class,z,y,x]

    Returns:
        4D numpy array
    """
    dim = scoremaps.shape
    Ncl = dim[3]
    dimB0 = np.int(np.ceil(dim[0] / 2))
    dimB1 = np.int(np.ceil(dim[1] / 2))
    dimB2 = np.int(np.ceil(dim[2] / 2))
    dimB = (dimB0, dimB1, dimB2, Ncl)
    scoremapsB = np.zeros(dimB)
    for cl in range(0,Ncl):
        scoremapsB[:,:,:,cl] = cm.bin_array(scoremaps[:,:,:,cl])
    return scoremapsB
    
def to_labelmap(scoremaps):
    """Converts scoremaps into a labelmap.

    Args:
        scoremaps (4D numpy array): array with index order [class,z,y,x]

    Returns:
        3D numpy array: array with index order [z,y,x]
    """
    labelmap = np.int8( np.argmax(scoremaps,3) )
    return labelmap

def read_h5(filename):
    """Reads scormaps stored in .h5 file.

    Args:
        filename (str): path to file
            This .h5 file has one dataset per class (dataset '/class*' contains scoremap of class *)

    Returns:
        4D numpy array: scoremaps array with index order [class,z,y,x]
    """
    h5file = h5py.File(filename, 'r')
    datasetnames = h5file.keys()
    Ncl = len(datasetnames)
    dim = h5file['class0'].shape 
    scoremaps = np.zeros((dim[0],dim[1],dim[2],Ncl))
    for cl in range(Ncl):
        scoremaps[:,:,:,cl] = h5file['class'+str(cl)][:]
    h5file.close()
    return scoremaps
    
def write_h5(scoremaps, filename):
    """Writes scoremaps in .h5 file

    Args:
        scoremaps (4D numpy array): array with index order [class,z,y,x]
        filename (str): path to file
            This .h5 file has one dataset per class (dataset '/class*' contains scoremap of class *)

    """
    h5file = h5py.File(filename, 'w')
    dim = scoremaps.shape
    Ncl = dim[3]
    for cl in range(Ncl):
        dset = h5file.create_dataset('class'+str(cl), (dim[0], dim[1], dim[2]), dtype='float16' )
        dset[:] = np.float16(scoremaps[:,:,:,cl])
    h5file.close()