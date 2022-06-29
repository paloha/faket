from radontea import backproject_3d
import numpy as np
from numpy.fft import fft2, ifft2
from .data import load_mrc, save_mrc, save_conf
from .data import get_theta_from_alignment, slice_to_valid
from .data import downsample_sinogram_space
from .data import downsample_sinogram_theta
from .data import match_mean_std, normalize, get_clim
from .filter import ramp2d, rampShrec

def reconstruct_mrc(**kwargs):
    """
    Wrapper around reconstruct function.
    Loads the sinogram from input_mrc.
    Loads the theta from input_mrc parent dir.
    Saves the kwargs as a json file next to the output_mrc.
    
    Parameters
    ----------
    **kwargs: dict
        Keyword arguments to `reconstruct` containing `input_mrc` instead of `sinogram`.
        Where `input_mrc` is a path to a mrc file containing the projections.
        Here, `theta` can be a full path to `alignment_simulated.txt` from where
        the information about tilt angles will be read, or a list (not a numpy array).
    """
    input_mrc = kwargs['input_mrc']
    print(f'# Processing: {input_mrc}')
    sinogram = load_mrc(input_mrc)
    if isinstance(kwargs['theta'], str):
        theta = get_theta_from_alignment(kwargs['theta'])
        kwargs.update({'theta': theta})
    if kwargs['output_mrc'] is not None:
        save_conf(kwargs['output_mrc'], kwargs)
    del kwargs['input_mrc']
    return reconstruct(sinogram, **kwargs)


def reconstruct(sinogram, theta, downsample_angle=1, downsample_pre=1, 
                downsample_post=1, order=3, filtering='ramp', filterkwargs=None, 
                z_valid=None, output_mrc=None, ncpus=None):
    """
    Uses radontea package 3D filtered backprojection to
    reconstruct the provided sinogram measured at theta.
    
    Parameters
    ----------
    
    sinogram:  ndarray, shape (θ, Y, M)
        Three-dimensional array containing the projections.
        Axis 0 contains projections. Axis 1 is the tilt axis.
    theta: list or ndarray, shape (θ,)
        One-dimensional array of tilt angles in degrees. 
    downsample_angle: int, default=1 (no downsampling)
        Sinogram downsampling in theta dimension with int step.
        Always retains the angle in the center of theta.
    downsample_pre: int, default=1 (no downsampling)
        Sinogram downsampling in all space dimensions with int step.
    downsample_post: int, default=1 (no downsampling)
        Reconstruction downsampling in all space dimensions with int step.
    order: int, default=3
        Order (0 - 5) of the spline interpolation during downsampling.
    filtering: str, default='ramp'
        Filter used during reconstruction with FBP algorithm.
        Accepts `ramp2d`, `shrecRamp`, and filters from radontea,
        'ramp', 'shepp-logan', 'cosine', 'hamming', or 'hann'.
    filterkwargs: dict
        Additional kwargs for `ramp2d` or `shrecRamp` filters.
    z_valid: 2-tuple, default=None (no slicing)
        Slices the reconstruction along the 0-axis (Z dimension)
        to a range of valid voxels given by a relative interval from 0 to 1. 
    output_mrc: str, default=None
        Path to the output mrc file. If None, no saving is done and
        reconstruction is just returned instead.
    ncpus: int, default=None
        Number of CPUs used to do the reconstruction. If None, the number
        is set automatically to all CPUs. 
    """
    
    
    print(f'-- Sinogram shape: {sinogram.shape}' )

    # Downsample in theta dimension such that the 0° angle is 
    # always present and the other angles are centered around it.
    # No interpolation is done here so it is fast and therefore
    # better to do before downsampling in space.
    if downsample_angle > 1:
        sinogram, theta = downsample_sinogram_theta(sinogram, theta, downsample_angle)
        print(f'-- Downsampled in theta | Sinogram shape: {sinogram.shape}')
    
    # Downsample the sinogram in space dimension
    if downsample_pre > 1:
        sinogram = downsample_sinogram_space(sinogram, downsample_pre, order)
        print(f'-- Downsampled in space | Sinogram shape: {sinogram.shape}')

    # Manual filtering
    if filtering == 'ramp2d':
        sizeX, sizeY = sinogram.shape[1:]
        sinogram = ifft2(fft2(sinogram) * ramp2d(sizeX, sizeY, **filterkwargs)).real
        filtering = None  # Turning off radontea filtering
    elif filtering == 'shrecRamp':
        sizeX, sizeY = sinogram.shape[1:]
        sinogram = ifft2(fft2(sinogram) * rampShrec(sizeX, sizeY, **filterkwargs)).real
        filtering = None  # Turning off radontea filtering            

    # On a consumer-grade Intel® Core™ i7-8565U CPU @ 1.80GHz × 8
    # Reconstructing one sinogram 61x512x512 into volume 512x512x512 using 8 cpus takes ~60 seconds.
    # Reconstructing one sinogram 61x1024x1024 into volume 1024x1024x1024 using 8 cpus takes ~10 minutes.
    reconstruction = backproject_3d(sinogram, np.deg2rad(theta), 
                                    filtering=filtering, weight_angles=False, 
                                    padding=False, padval=None, ncpus=ncpus)[::-1,:,:]

    print(f'-- Reconstructed using FBP | Reconstruction shape: {reconstruction.shape}')

    # Slice the reconstruction to a valid region
    if z_valid is not None:
        reconstruction = slice_to_valid(reconstruction, *z_valid)
        print(f'-- Sliced Z dimension to valid region | Reconstruction shape: {reconstruction.shape}')

    # Downsample the reconstruction in space dimension
    if downsample_post > 1:
        from scipy.ndimage import zoom
        ratio = 1 / downsample_post
        # Ndimage zoom with spline interpolation of desired order
        reconstruction = zoom(reconstruction, (ratio, ratio, ratio), order=order)
        print(f'-- Downsampled in space | Reconstruction shape: {reconstruction.shape}')

    # Saving the reconstructed volume into an mrc file
    if output_mrc is not None:
        save_mrc(reconstruction.astype(np.float32), output_mrc, overwrite=True)
        print(f'-- Reconstruction saved as mrc file.')
    else:
        return reconstruction.astype(np.float32)


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


def noise_projections(input_mrc, style_mrc, output_mrc, mean=0.0, std=0.4, clip_outliers=(0.0, 1.0), seed=0):
    """
    Adds Gaussian noise of specified mean and std to input_mrc which is then 
    rescaled per tilt to match the mean and std of the tilts in style mrc.
    
    Parameters
    ----------
    input_mrc: str
        Path to a mrc file containig noiseless projections.
    style_mrc: str
        Path to a mrc file containig style projections. Ideally a real sinogram 
        from the data set on which we want to later predict.
    output_mrc: str
        Path to a mrc file where the result is going to be stored.
    mean: float, default=0.0
        Mean of the Gaussian noise.
    std: float, default=0.4
        Standard deviation of the Gaussian noise.
    clip_outliers: 2-tuple of floats, default=(0.0, 1.0)
        Clip data outside of the specified portion.
    seed: 
        Random seed used to generate the noise.
    """
    
    # Load input projections
    volume = load_mrc(input_mrc)
    style = load_mrc(style_mrc)
    
    # Generate random noise
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=mean, scale=std, 
                       size=volume.size).reshape(volume.shape)
    
    # Scaling per tilt based on style (bigger the abs(angle), longer the trajectory)
    volume  = match_mean_std(volume, style)  
    
    # Scale between [0, 1] before adding the noise
    volume = normalize(volume)  
    
    # Add the noise
    volume_noisy = volume + noise
    
    # Remove outliers
    volume_noisy = np.clip(volume_noisy, *get_clim(volume_noisy, *clip_outliers)) 
    
    # Scale back (per tilt) to match style
    volume_noisy = match_mean_std(volume_noisy, style)  
    
    # Save output
    save_mrc(volume_noisy.astype(np.float32), output_mrc, overwrite=True)
    
    
    