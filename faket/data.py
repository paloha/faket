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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    square = load_mrc(os.path.join(data_folder, N, f'{fname}.mrc'))
    assert square.shape[0] == square.shape[1], 'Not square'
    valid = slice_to_valid(square, *z_valid)
    out_fname = out_fname or f'{fname}_valid.mrc'
    save_mrc(valid.astype(np.float32), os.path.join(data_folder, N, out_fname), overwrite=True)
    print(f'-- DONE slicing {fname} {N} to valid range and saving as mrc file.')
    
    
def match_mean_std(vol1, vol2): 
    """
    Matches mean and std of all arrays
    in vol1 according to mean and std
    of respective arrays in vol2.
    """
    n_tilts = vol1.shape[0]
    assert n_tilts == vol2.shape[0]
    r = lambda x: x.reshape(n_tilts, -1)
    
    vol1s = r(vol1).std(-1).reshape(-1,1,1)
    vol2s = r(vol2).std(-1).reshape(-1,1,1)
    vol1 = vol1 / vol1s * vol2s
    
    vol1m = r(vol1).mean(-1).reshape(-1,1,1)
    vol2m = r(vol2).mean(-1).reshape(-1,1,1)
    vol1 -= (vol1m - vol2m)
    return vol1 


def normalize(x):
    """
    Shifts and scales an array into [0, 1]
    """
    n = x.copy()
    n -= n.min()
    n /= n.max()
    return n

    
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
