import os
import csv
import json
import mrcfile
import zipfile
import radontea
import numpy as np
from numpy.fft import fft2, ifft2


def load_mrc(path):
    """
    Loads the mrc.data from a specified path
    """
    import mrcfile
    with mrcfile.open(path, permissive=True) as mrc:
        return mrc.data.copy()
    
    
def save_mrc(data, path, overwrite=False):
    """
    Saves the data into a mrc file.
    """
    import mrcfile
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(data)


def save_conf(path:str, conf:dict):
    """
    Saves conf as json to the file of the same 
    name as provided in the path just with the
    json extension. E.g. if path is `foo/bar.mrc`,
    saved file will be `foo/bar.json`.
    """
    path = os.path.splitext(path)[0] + '.json'
    with open(path, 'w') as fl:
            json.dump(conf, fl, indent=4)


def get_theta_from_alignment(path):
    """
    Opens the `alignment_simulated.txt` file provided for each
    tomogram in the SHREC2021 dataset and parses the information
    about the tilt angles. 
    """
    with open(path, 'r') as tsv:
        rows = list(csv.reader(tsv, delimiter=' ', skipinitialspace=True))[6:]  # Omitting first rows
        theta = [float(row[2]) for row in rows]
        assert len(theta) == 61, f'Problem in loading theta from {file}. Number of angles is not 61.'
    return theta


def get_theta(data_folder=None, N=None):
    """
    Opens the 'alignment_simulated.txt' file from desired tomogram
    and returns the np.array of tilt angles (floats).
    """
    tomogram_folder = os.path.join(data_folder, f'model_{N}')
    file = os.path.join(tomogram_folder, 'alignment_simulated.txt')
    return get_theta_from_alignment(file)


def get_clim(data, lo=0.01, up=0.99):
    """
    Returns values (clims) between which a specified portion 
    of data lies. Useful for plotting matrices with outliers.
    Can be used as an argument to call np.clip(data, *clims)
    """
    return np.quantile(data, lo), np.quantile(data, up)


def theta_subsampled_indices(theta_len, step):
    """
    Picks every step-th tilt in both directions
    from the 0° tilt. Returns a list of indices
    of chosen tilts assuming len(theta) is odd 
    and the 0° tilt is exactly in the middle.
    Apply to theta with `theta[indices]` to get
    the list of tilt angles instead of indices.
    If the step is too high, returns only the index
    of the 0° tilt.
    """
    assert theta_len % 2 != 0, 'Number of titls must be odd.'
    assert step > 1, 'Only supports int steps > 1.'
    assert isinstance(step, int), 'Only supports int steps.'
    indices = np.concatenate(
    [np.arange(0, theta_len // 2 + 1)[::-step][::-1], 
     np.arange(theta_len // 2, theta_len)[::step][1:]])
    return indices


def slice_to_valid(array, rel_min, rel_max):
    """
    Slices an array in the 0th axis to a region
    specified by the relative boundaries.
    """
    abs_min = round(rel_min * array.shape[0])
    abs_max = round(rel_max * array.shape[0])
    assert abs_min < abs_max
    return array[abs_min:abs_max]


def vol_to_valid(data_folder, N, fname, z_valid, out_fname=None):
    """
    Opens the fname from data_folder of Nth tomogram, 
    slices it in Z dimension according to the 2-tuple 
    z_valid normalized between 0 and 1 and saves it
    as a mrc file with '_valid' suffix.
    """
    print(f'# Slicing {fname} {N}')
    square = load_mrc(os.path.join(data_folder, str(N), f'{fname}.mrc'))
    assert square.shape[0] == square.shape[1], 'Not square'
    valid = slice_to_valid(square, *z_valid)
    out_fname = out_fname or f'{fname}_valid.mrc'
    save_mrc(valid.astype(np.float32), os.path.join(data_folder, str(N), out_fname), overwrite=True)
    print(f'-- DONE slicing {fname} {N} to valid range and saving as mrc file.')

    
def match_mean_std(vol1, vol2=None, means=None, stds=None): 
    """
    Matches mean and std of all arrays in vol1 according
    to the mean and std of respective arrays in vol2 if
    vol2 is not None, else to the means and stds arrays.
    If vol2 is not None, means and stds are ignored.
    """
    n_tilts = vol1.shape[0]
    r = lambda x: x.reshape(n_tilts, -1)
    
    if vol2 is None:
        assert means is not None and stds is not None, \
        'Either provide vol2 or means and stds arrays.'
        
        assert len(means) == len(stds) == n_tilts, \
        'Length of means and stds must match vol1.shape[0].' 
        
        means = np.asarray(means).reshape(-1, 1, 1)
        stds = np.asarray(stds).reshape(-1, 1, 1)
    else:
        assert n_tilts == vol2.shape[0], \
        'Number of tilts in vol1 and vol2 must match.'
    
    vol1s = r(vol1).std(-1).reshape(-1,1,1)
    vol2s = stds if vol2 is None else r(vol2).std(-1).reshape(-1,1,1)
    vol1 = vol1 / vol1s * vol2s
    
    vol1m = r(vol1).mean(-1).reshape(-1,1,1)
    vol2m = means if vol2 is None else r(vol2).mean(-1).reshape(-1,1,1)
    vol1 -= (vol1m - vol2m)
    return vol1 
    

def normalize(x):
    """
    Shifts and scales an array into [0, 1]
    """
    x = x.copy()
    x -= x.min()
    x /= x.max()
    return x


def standardize(x):
    """
    Standardizes an array to mean 0 and variance 1.
    """
    x = x.copy()
    x -= x.mean()
    x /= x.std()
    return x


def standardize_per_tilt(x):
    """
    Standardizes an array to mean 0 and variance 1
    along the first axis. I.e. in an array of shape
    (61, 1024, 1024), each 1024x1024 subarray will
    be standardized separately.
    """
    shape = x.shape
    x = x.copy().reshape(shape[0], -1)
    x -= x.mean(axis=-1).reshape(-1, 1)
    x /= x.std(axis=-1).reshape(-1, 1)
    return x.reshape(shape)

    
def downsample_sinogram_theta(sinogram, theta, step):
    """
    Downsamples first axis of the sinogram according
    to the desired step such that 0 tilt is always present.
    0 tilt is assumed to be the middle tilt in theta. 
    See 'theta_subsampled_indices' for more info.
    """
    indices = theta_subsampled_indices(len(theta), step)
    return sinogram[indices], theta[indices]
    
    
def downsample_sinogram_space(sinogram, n, order):
    """
    Downsamples last two axes of sinogram according to 
    ratio 1/n using interpolation of desired order. 
    Sinogram must be in shape (θ, ?, ?).
    """
    from scipy.ndimage import zoom
    return zoom(sinogram, (1.0, 1 / n, 1 / n), order=order)


def fix_edges_projections(projections, n=0):
    """
    In projections and projections_unbinned provided by SHREC, 
    for unknown reason, there are edge artifacts in the first few
    columns and last few columns of every tilt. The mean of the 
    column is shifted which causes a horizontal stripe through
    origin in fourier space after filtering. To fix this issue
    without cropping the projections we decided to match the
    mean of first frew columns with the mean of nth column.
    Same from the other side but mirrored.
    """
    if n == 0:
        return projections
    p = projections.copy()
    end = p.shape[2]
    for i in range(n):
        p[:,:,i:i+1] -= (
            np.expand_dims(p[:,:,i:i+1].mean(axis=1), axis=1) - 
            np.expand_dims(p[:,:,n:n+1].mean(axis=1), axis=1))
        p[:,:,end-i-1:end-i] -= (
            np.expand_dims(p[:,:,end-i-1:end-i].mean(axis=1), axis=1) - 
            np.expand_dims(p[:,:,end-n-1:end-n].mean(axis=1), axis=1))
    return p


def fix_edges_reconstruction(reconstruction, n=0):
    """
    SHREC attenuates edges of reconstruction. Without it, the 
    reconstructon in Fourier space has a pronounced horizontal
    stripe through origin. Even after fixing edges of projections.
    """
    if n == 0:
        return reconstruction
    z, y, x = reconstruction.shape
    reconstruction[:,:n,:] *= (np.arange(n) / n).reshape(1, -1, 1)
    reconstruction[:,y-n:,:] *= (np.arange(n)[::-1] / n).reshape(1, -1, 1)
    return reconstruction