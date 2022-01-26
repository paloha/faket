# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import os
import numpy as np
import h5py

import mrcfile
import warnings
warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos with mrcfile

from skimage.measure import block_reduce
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates
#from bm3d import bm3d, BM3DProfileLC, BM3DStages

import matplotlib
matplotlib.use('agg') # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt

from PIL import Image # for reading tif

# Writes an image file containing ortho-slices of the input volume. Generates same visualization as matlab function
# 'tom_volxyz' from TOM toolbox.
# If volume type is int8, the function assumes that the volume is a labelmap, and hence plots in color scale.
# Else, it assumes that the volume is tomographic data, and plots in gray scale.
# INPUTS:
#   vol     : 3D numpy array
#   filename: string '/path/to/file.png'
def plot_volume_orthoslices(vol, filename):
    """Writes an image file containing ortho-slices of the input volume. Generates same visualization as matlab function
    'tom_volxyz' from TOM toolbox.
    If volume type is int8, the function assumes that the volume is a labelmap, and hence plots in color scale.
    Else, it assumes that the volume is tomographic data, and plots in gray scale.

    Args:
        vol (3D numpy array)
        filename (str): '/path/to/file.png'
    """

    # Get central slices along each dimension:
    dim = vol.shape
    idx0 = np.int( np.round(dim[0]/2) )
    idx1 = np.int( np.round(dim[1]/2) )
    idx2 = np.int( np.round(dim[2]/2) )

    slice0 = vol[idx0,:,:]
    slice1 = vol[:,idx1,:]
    slice2 = vol[:,:,idx2]

    # Build image containing orthoslices:
    img_array = np.zeros((slice0.shape[0]+slice1.shape[0], slice0.shape[1]+slice1.shape[0]))
    img_array[0:slice0.shape[0], 0:slice0.shape[1]] = slice0
    img_array[slice0.shape[0]-1:-1, 0:slice0.shape[1]] = slice1
    img_array[0:slice0.shape[0], slice0.shape[1]-1:-1] = np.flipud(np.rot90(slice2))

    # Drop the plot:
    fig = plt.figure(figsize=(10,10))
    if vol.dtype==np.int8:
        plt.imshow(img_array, cmap='CMRmap', vmin=np.min(vol), vmax=np.max(vol))
    else:
        mu  = np.mean(vol) # Get mean and std of data for plot range:
        sig = np.std(vol)
        plt.imshow(img_array, cmap='gray', vmin=mu-5*sig, vmax=mu+5*sig)
    fig.savefig(filename)

# Reads data stored in h5 file, from specified h5 dataset.
# INPUTS:
#   filename : string '/path/to/file.h5'
#   dset_name: string dataset name
# OUTPUT:
#   dataArray: numpy array
def read_h5array(filename, dset_name='dataset'):
    h5file = h5py.File(filename, 'r')
    dataArray = h5file[dset_name][:]
    h5file.close()
    return dataArray

# Writes data in h5 file, to specified h5 dataset. Is also adapted for labelmaps: saved as int8 to gain disk space.
# INPUTS:
#   array    : numpy array
#   filename : string '/path/to/file.h5'
#   dset_name: string dataset name
def write_h5array(array, filename, dset_name='dataset'):
    h5file = h5py.File(filename, 'w')
    if array.dtype == np.int8:
        dset = h5file.create_dataset(dset_name, array.shape, dtype='int8')
        dset[:] = np.int8(array)
    else:
        dset = h5file.create_dataset(dset_name, array.shape, dtype='float16')
        dset[:] = np.float16(array)
    h5file.close()

# Reads array stored as mrc.
# INPUTS:
#   filename: string '/path/to/file.mrc'
# OUTPUT:
#   array: numpy array
def read_mrc(filename):
    with mrcfile.open(filename, permissive=True) as mrc:
        array = mrc.data
    return array

# Writes array as mrc.
# INPUTS:
#   array   : numpy array
#   filename: string '/path/to/file.mrc'
def write_mrc(array, filename):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(array)

def read_tif(filename):
    dataset = Image.open(filename)
    h, w = np.shape(dataset)
    tiffarray = np.zeros((dataset.n_frames, w, h))
    for i in range(dataset.n_frames):
        dataset.seek(i)
        tiffarray[i, :, :] = np.transpose(np.array(dataset))
    return tiffarray.astype(np.single)

# Reads arrays. Handles .h5 and .mrc files, according to what extension the file has.
# INPUTS:
#   filename : string '/path/to/file.ext' with '.ext' either '.h5' or '.mrc'
#   dset_name: string h5 dataset name. Not necessary to specify when reading .mrc
# OUTPUT:
#   array: numpy array
def read_array(filename, dset_name='dataset'):
    """Reads arrays. Handles .h5 and .mrc files, according to what extension the file has.

    Args:
        filename (str): '/path/to/file.ext' with '.ext' either '.h5' or '.mrc'
        dset_name (str, optional): h5 dataset name. Not necessary to specify when reading .mrc

    Returns:
        numpy array
    """
    data_format = os.path.splitext(filename)
    if data_format[1] == '.h5':
        array = read_h5array(filename, dset_name)
    elif data_format[1] == '.mrc' or data_format[1] == '.map' or data_format[1] == '.rec':
        array = read_mrc(filename)
    elif data_format[1] == '.tif' or data_format[1] == '.TIF':
        array = read_tif(filename)
    else:
        print('/!\ DeepFinder can only read datasets in .h5 and .mrc formats')
    return array

# Writes array. Can write .h5 and .mrc files, according to the extension specified in filename.
# INPUTS:
#   array    : numpy array
#   filename : string '/path/to/file.ext' with '.ext' either '.h5' or '.mrc'
#   dset_name: string h5 dataset name. Not necessary to specify when writing .mrc
def write_array(array, filename, dset_name='dataset'):
    """Writes array. Can write .h5 and .mrc files, according to the extension specified in filename.

    Args:
        array (numpy array)
        filename (str): '/path/to/file.ext' with '.ext' either '.h5' or '.mrc'
        dset_name (str, optional): h5 dataset name. Not necessary to specify when reading .mrc
    """
    data_format = os.path.splitext(filename)
    if data_format[1] == '.h5':
        write_h5array(array, filename, dset_name)
    elif data_format[1] == '.mrc':
        write_mrc(array, filename)
    else:
        print('/!\ DeepFinder can only write arrays in .h5 and .mrc formats')

# Subsamples a 3D array by a factor 2. Subsampling is performed by averaging voxel values in 2x2x2 tiles.
# INPUT: numpy array
# OUTPUT: binned numpy array
def bin_array(array):
    """Subsamples a 3D array by a factor 2. Subsampling is performed by averaging voxel values in 2x2x2 tiles.

    Args:
        array (numpy array)

    Returns:
        numpy array: binned array

    """
    return block_reduce(array, (2,2,2), np.mean)

# Rotates a 3D array and uses the same (phi,psi,the) convention as TOM toolbox (matlab) and PyTOM.
# Code based on: https://nbviewer.jupyter.org/gist/lhk/f05ee20b5a826e4c8b9bb3e528348688
# INPUTS:
#   array: 3D numpy array
#   orient: list of Euler angles (phi,psi,the) as defined in PyTOM
# OUTPUT:
#   arrayR: rotated 3D numpy array
def rotate_array(array, orient): # TODO move to core_utils?
    phi = orient[0]
    psi = orient[1]
    the = orient[2]

    # Some voodoo magic so that rotation is the same as in pytom:
    new_phi = -phi
    new_psi = -the
    new_the = -psi

    # create meshgrid
    dim = array.shape
    ax = np.arange(dim[0])
    ay = np.arange(dim[1])
    az = np.arange(dim[2])
    coords = np.meshgrid(ax, ay, az)

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                     coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                     coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

    # create transformation matrix: the convention is not 'zxz' as announced in TOM toolbox
    r = R.from_euler('YZY', [new_phi, new_psi, new_the], degrees=True)
    ##r = R.from_euler('ZXZ', [the, psi, phi], degrees=True)
    mat = r.as_matrix()

    # apply transformation
    transformed_xyz = np.dot(mat, xyz)

    # extract coordinates
    x = transformed_xyz[0, :] + float(dim[0]) / 2
    y = transformed_xyz[1, :] + float(dim[1]) / 2
    z = transformed_xyz[2, :] + float(dim[2]) / 2

    x = x.reshape((dim[1],dim[0],dim[2]))
    y = y.reshape((dim[1],dim[0],dim[2]))
    z = z.reshape((dim[1],dim[0],dim[2])) # reason for strange ordering: see next line

    # the coordinate system seems to be strange, it has to be ordered like this
    new_xyz = [y, x, z]

    # sample
    arrayR = map_coordinates(array, new_xyz, order=1)

    # Remark: the above is equivalent to the below, however the above is faster (0.01s vs 0.03s for 40^3 vol).
    # arrayR = scipy.ndimage.rotate(array, new_phi, axes=(1, 2), reshape=False)
    # arrayR = scipy.ndimage.rotate(arrayR, new_psi, axes=(0, 1), reshape=False)
    # arrayR = scipy.ndimage.rotate(arrayR, new_the, axes=(1, 2), reshape=False)
    return arrayR

# Creates a 3D array containing a full sphere (at center). Is used for target generation.
# INPUTS:
#   dim: list of int, determines the shape of the returned numpy array
#   R  : radius of the sphere (in voxels)
# OUTPUT:
#   sphere: 3D numpy array where '1' is 'sphere' and '0' is 'no sphere'
def create_sphere(dim, R): # TODO move to core_utils?
    C = np.floor((dim[0]/2, dim[1]/2, dim[2]/2))
    x,y,z = np.meshgrid(range(dim[0]),range(dim[1]),range(dim[2]))

    sphere = ((x - C[0])/R)**2 + ((y - C[1])/R)**2 + ((z - C[2])/R)**2
    sphere = np.int8(sphere<=1)
    return sphere

# Denoises 2D image.
# INPUTS:
#   img: 2D numpy array
#   sigma_noise: noise standard deviation
# OUTPUT:
#   img_den: denoised 2D numpy array
#def denoise2D(img, sigma_noise):
#    img_den = bm3d(z=img, sigma_psd=sigma_noise, profile=BM3DProfileLC(), stage_arg=BM3DStages.HARD_THRESHOLDING)
#    return img_den

