from radontea import backproject_3d
import numpy as np
from numpy.fft import fft2, ifft2
from .data import load_mrc, save_mrc
from .data import get_theta, slice_to_valid
from .data import downsample_sinogram_space
from .data import downsample_sinogram_theta
from .filter import ramp2d, rampShrec


def reconstruct(data_folder, N, conf):
    """
    Uses radontea package 3D filtered backprojection to 
    reconstruct the projections_unbinned according to the config.
    Stores the reconstruction as a mrc file.
    
    Blueprint of the config:
    
    conf = {
    'input_mrc' : 'projections_unbinned.mrc'  # File containing the sinograms (θ, Y, M)
    'output_mrc' : 'reconstruction_bin-1_theta-1.mrc',  # name of the resulting file (X, Y, Z)
    'downsample_angle' : 1,  # Sinogram downsampling in theta dimension (1 = no downsampling)
    'downsample_pre' : 2,  # Sinogram downsampling in space (1 = no downsampling)
    'order' : 2,  # Downsampling in space with spline interpolation of order (0 - 5)
    'filter' : 'ramp',  # Filter userd during reconstruction in FBP algorithm
    'downsample_post' : 1,  # Reconstruction downsampling in space
    'ncpus': 8,  # Number of CPUs to use while reconstructing
    # 2-tuple range of valid pixels in Z dimension normalized from 0 to 1. (0., 1.) or None for all.
    'z_valid': (0.32226, 0.67382) 
    }
    """
    sinogram = load_mrc(data_folder, N, conf['input_mrc'])
    theta = get_theta(data_folder, N)
    print(f'# Processing tomogram {N} | Sinogram shape: {sinogram.shape}' )

    # Downsample in theta dimension such that the 0° angle is 
    # always present and the other angles are centered around it.
    # No interpolation is done here so it is fast and therefore
    # better to do before downsampling in space.
    if conf['downsample_angle'] > 1:
        sinogram, theta = downsample_sinogram_theta(sinogram, theta, conf['downsample_angle'])
        log.info(f'-- Downsampled in theta | Sinogram shape: {sinogram.shape}')
    
    # Downsample the sinogram in space dimension
    if conf['downsample_pre'] > 1:
        sinogram = downsample_sinogram_space(sinogram, conf['downsample_pre'], conf['order'])
        print(f'-- Downsampled in space | Sinogram shape: {sinogram.shape}')

    # Manual filtering
    if conf['filter'] == 'ramp2d':
        sizeX, sizeY = sinogram.shape[1:]
        sinogram = ifft2(fft2(sinogram) * ramp2d(sizeX, sizeY, **conf['filterkwargs'])).real
        filtering = None  # Turning off radontea filtering
    elif conf['filter'] == 'shrecRamp':
        sizeX, sizeY = sinogram.shape[1:]
        sinogram = ifft2(fft2(sinogram) * rampShrec(sizeX, sizeY, **conf['filterkwargs'])).real
        filtering = None  # Turning off radontea filtering
    else:
        filtering = conf['filter']
              

    # On a consumer-grade Intel® Core™ i7-8565U CPU @ 1.80GHz × 8
    # Reconstructing one sinogram 61x512x512 into volume 512x512x512 using 8 cpus takes ~60 seconds.
    # Reconstructing one sinogram 61x1024x1024 into volume 1024x1024x1024 using 8 cpus takes ~10 minutes.
    reconstruction = backproject_3d(sinogram, np.deg2rad(theta), 
                                    filtering=filtering, weight_angles=False, 
                                    padding=False, padval=None, ncpus=conf['ncpus'])[::-1,:,:]

    print(f'-- Reconstructed using FBP | Reconstruction shape: {reconstruction.shape}')

    # Slice the reconstruction to a valid region
    if conf['z_valid'] is not None:
        reconstruction = slice_to_valid(reconstruction, *conf['z_valid'])
        print(f'-- Sliced Z dimension to valid region | Reconstruction shape: {reconstruction.shape}')

    # Downsample the reconstruction in space dimension
    if conf['downsample_post'] > 1:
        from scipy.ndimage import zoom
        ratio = 1 / conf['downsample_post']
        # Ndimage zoom with spline interpolation of desired order
        reconstruction = zoom(reconstruction, (ratio, ratio, ratio), order=conf['order'])
        print(f'-- Downsampled in space | Reconstruction shape: {reconstruction.shape}')

    # Saving the reconstructed volume into an mrc file
    save_mrc(reconstruction.astype(np.float32), data_folder, N, conf['output_mrc'], overwrite=True)
    print(f'-- DONE processing tomogram {N} and saved the reconstruction as mrc file.')
    

def radon_3d(volume, theta, ncpus=None, dose=None, out_shape=None, circle=False, slice_axis=0):
    """
    volume: 3D np array (X, Y, Z)
        To be measured with radon transform.
    theta: 1D np array
        Tilt angles in degrees
    ncpus: int
        How many cpus to use for multiprocessing
    dose: float
        Electrondose per squared pixel (for flipping the values)
    out_shape: int
        Desired length of the vector measured by radon
    circle: bool
        kwarg for radon
    slice_axis: int between 0 and 2
        Specifies which axis contains slices which are going to 
        be processed in parallel by radon transform.
        E.g. 2x512x512 with slice_axis=0 means two radon transforms
    """
    from functools import partial
    from skimage.transform import radon
    from multiprocessing import Pool, cpu_count
    
    ncpus = ncpus or cpu_count()
    func = partial(radon, theta=theta, circle=circle)
    
    # Move the slice axis to front 
    if slice_axis != 0:
        volume = np.moveaxis(volume, slice_axis, 0)
    
    # Output shape will be (?, ?, len(theta))
    with Pool(processes=ncpus) as pool:
        sinogram = np.array(pool.map(func, volume))
        # sinogram = sinogram.swapaxes(0, -1)
        
    if dose is not None:
        # Flipping the values according to the dose
        # In microscope, we measure attenuation not sum
        # If dose = 0, sinogram is just flipped
        sinogram = dose - sinogram
        
    if out_shape is not None:
        # Slice the output to desired shape such that the center is kept 
        start = (sinogram.shape[1] - out_shape) // 2
        assert start >=0, f'out_shape is not <= n measurements of radon {sinogram.shape[1]}'
        end = start + out_shape
        sinogram = sinogram[:,start:end,:]
    
    # Undo the move of the slice axis to front 
    if slice_axis != 0:
        sinogram = np.moveaxis(sinogram, 0, slice_axis)
        
    # Swap theta axis with measurement axis to match SHREC convention
    sinogram = np.swapaxes(sinogram, -1, 0)
    return sinogram  # Output shape (θ, slice_axis, M)