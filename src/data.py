import os
import csv
import mrcfile
import zipfile
import radontea
import numpy as np
from numpy.fft import fft2, ifft2
        
def load_mrc(data_folder, N, mrc_name):
    """
    Loads the mrc.data of a specified mrc file from the
    tomogram folder at index N.
    """
    tomogram_folder = os.path.join(data_folder, f'model_{N}')
    file = os.path.join(tomogram_folder, mrc_name)

    with mrcfile.open(file, permissive=True) as mrc:
        # the tomo data is now accessible via .data, in following order: Z Y X
        print('Loaded data of shape:', mrc.data.shape)
        print('Size of the data:', mrc.data.nbytes / float(1000**3), 'GB')
        data = mrc.data.copy()
    return data


def save_mrc(data, data_folder, N, mrc_name, overwrite=False):
    """
    Saves the data into a mrc file of specified name
    into the tomogram folder at index N.
    """
    tomogram_folder = os.path.join(data_folder, f'model_{N}')
    file = os.path.join(tomogram_folder, mrc_name)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with mrcfile.new(file, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        
        
def get_theta(data_folder, N):
    """
    Opens the 'alignment_simulated.txt' file from desired tomogram
    and returns the np.array of tilt angles (floats).
    """
    tomogram_folder = os.path.join(data_folder, f'model_{N}')
    file = os.path.join(tomogram_folder, 'alignment_simulated.txt')
    with open(file, 'r') as tsv:
        rows = list(csv.reader(tsv, delimiter=' ', skipinitialspace=True))[6:]  # Omitting first rows
        theta = [float(row[2]) for row in rows]
        assert len(theta) == 61, f'Problem in loading theta from {file}. Number of angles is not 61.'
    return np.array(theta)


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
    print(f'# Slicing {fname} {N}' )
    square = load_mrc(data_folder, N, f'{fname}.mrc')
    assert square.shape[0] == square.shape[1], 'Not square'
    valid = slice_to_valid(square, *z_valid)
    out_fname = out_fname or f'{fname}_valid.mrc'
    save_mrc(valid.astype(np.float32), data_folder, N, out_fname, overwrite=True)
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
    indices = theta_subsampled_indices(theta.size, step)    
    return sinogram[indices], theta[indices]
    
    
def downsample_sinogram_space(sinogram, n, order):
    """
    Downsamples last two axes of sinogram according to 
    ratio 1/n using interpolation of desired order. 
    Sinogram must be in shape (θ, ?, ?).
    """
    from scipy.ndimage import zoom
    return zoom(sinogram, (1.0, 1 / n, 1 / n), order=order)


# TODO ADJUST THIS TO HANDLE THE EXTENDED DATASET
# def prepare_shrec21(zip_url=None, zip_fname=None, zip_dir=None):
#     # As of 27.05.2021 available at https://www.shrec.net/cryo-et/ from the Google Drive link
#     zip_url = zip_url or 'https://drive.google.com/file/d/1KcNkdKitxDt_Jhlg6kgFRUncpoEKC1vQ'
#     zip_fname = zip_fname or 'shrec2021_contest_dataset.zip'
#     zip_dir = zip_dir or 'data'
#     zip_path = os.path.join(zip_dir, zip_fname)
#     data_folder = os.path.splitext(zip_path)[0]  # Relative to project root
#     anticipated_size = 7823307210  # Sanity check the zip file
#     anticipated_files = {'misc', 'model_0', 'model_1', 'model_2', 
#                          'model_3', 'model_4', 'model_5', 'model_6',
#                          'model_7', 'model_8', 'model_9', 'readme.txt'}
    
#     if not os.path.isdir(data_folder):        
#         print('Extracted data not found. Searching for zip archive.' )

#         # Getting the zip file from user
#         while not os.path.isfile(zip_path):
#             print(f'{zip_path} does not exist.' )
#             print(f'Provide the data. (Should be available from {zip_url}).')
#             input('Hit Enter to continue.')

#         # Checking the zip size
#         zip_size = os.path.getsize(zip_path)
#         if anticipated_size == zip_size:
#             print('Sanity check OK | Zip archive matches anticipated size.')
#         else:
#             print(f'Zip archive size {zip_size} does not match anticipated size {anticipated_size}.')
#             print('Please make sure you have downloaded the correct version of the dataset.')

#         # Extracting the zip file
#         print(f'Unpacking {zip_fname}. Please wait...')
#         try:
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(data_folder)
#         except Exception as e:
#             print(e)
#         print('Unpacking successful.')
#     else:
#         print('Extracted data found.')

#     # Sanity check folder content
#     if set(os.listdir(data_folder)) != anticipated_files:
#         print(f'Data folder does not contain the anticipated files. {sorted(anticipated_files)}')
#     else:
#         print(f'The data folder contains anticipated files.')
    
#     return data_folder