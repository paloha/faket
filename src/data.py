
# Functions from the FastET project for loading and saving the data
import mrcfile
import os


def load_mrc(data_folder, N, mrc_name):
    """
    Loads the mrc.data of a specified mrc file from the
    tomogram folder at index N.
    """
    tomogram_folder = os.path.join(data_folder, f'model_{N}')
    file = os.path.join(tomogram_folder, mrc_name)

    with mrcfile.open(file, permissive=True) as mrc:
        # the tomo data is now accessible via .data, in following order: Z Y X
        print('Shape of the data:', mrc.data.shape)
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
    with mrcfile.new(file, overwrite=overwrite) as mrc:
        mrc.set_data(data)
