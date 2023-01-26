import os
import time
import numpy as np
from shutil import which
import subprocess
from numpy.fft import fft2, ifft2
from .data import fix_edges_projections
from .data import fix_edges_reconstruction
from .data import load_mrc, save_mrc, save_conf
from .data import get_theta_from_alignment, slice_to_valid
from .data import downsample_sinogram_space
from .data import downsample_sinogram_theta
from .data import match_mean_std, normalize, get_clim
from .filter import approxShrec
from radontea import backproject_3d


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
                z_valid=None, output_mrc=None, ncpus=None, 
                fix_edges_proj=0, fix_edges_rec=0, software='radontea'):
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
    filtering: str or None, default=None
        Filter used during reconstruction with FBP algorithm.
        Accepts `approxShrec`, and in case `software='radontea'` accepts
        also filters from radontea {'ramp', 'shepp-logan', 'cosine', 
        'hamming', 'hann'}.
    filterkwargs: dict
        Additional kwargs for `approxShrec` filter.
    z_valid: 2-tuple, default=None (no slicing)
        Slices the reconstruction along the 0-axis (Z dimension)
        to a range of valid voxels given by a relative interval from 0 to 1. 
    output_mrc: str, default=None
        Path to the output mrc file. If None, no saving is done and
        reconstruction is just returned instead.
    ncpus: int, default=None
        If `software='radontea'` this is the number of CPUs used to do
        the reconstruction. If None, the number is set automatically 
        to all CPUs. Ignored if `software='imod'`. 
    fix_edges_proj: int, default=0
        Fixing artifacts as in SHREC baseline projections.
    fix_edges_rec: int, default=0
        Fixing artifacts as in SHREC baseline reconstructions.
    software: str, one of {'radontea', 'imod'}, default 'radontea'
        Specifies which software is used to do the reconstruction. 
        If 'imod', CLI program `tilt` from IMOD must be installed and
        available on PATH and executable.
    """
    
    
    print(f'-- Sinogram shape: {sinogram.shape}' )
    
    theta = np.asarray(theta)

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
    
    # Fixing artifacts in projection space if necessary
    sinogram = fix_edges_projections(sinogram, n=fix_edges_proj)

    # Custom filtering
    if filtering == 'approxShrec':
        sizeX, sizeY = sinogram.shape[1:]
        filterkwargs = filterkwargs or {}
        sinogram = ifft2(fft2(sinogram) * approxShrec(sizeX, sizeY, **filterkwargs)).real
        filtering = None  # Turning off radontea filtering            

    if software == 'radontea':
        # On a consumer-grade Intel® Core™ i7-8565U CPU @ 1.80GHz × 8
        # Reconstructing one sinogram 61x512x512 into volume 512x512x512 using 8 cpus takes ~60 seconds.
        # Reconstructing one sinogram 61x1024x1024 into volume 1024x1024x1024 using 8 cpus takes ~10 minutes.
        reconstruction = backproject_3d(sinogram, np.deg2rad(theta), 
                                        filtering=filtering, weight_angles=False, 
                                        padding=False, padval=None, ncpus=ncpus)[::-1,:,:]
    elif software == 'imod':
        if filtering is not None:
            raise ValueError('')
        reconstruction = backproject_3d_imod(sinogram, theta, z_valid)
    else:
        raise NotImplementedError('Supported choices for software are {"radontea", "imod"}')

    print(f'-- Reconstructed using FBP | Reconstruction shape: {reconstruction.shape}')

    # Slice the reconstruction to a valid region
    if z_valid is not None and software != 'imod':
        reconstruction = slice_to_valid(reconstruction, *z_valid)
        print(f'-- Sliced Z dimension to valid region | Reconstruction shape: {reconstruction.shape}')

    # Downsample the reconstruction in space dimension
    if downsample_post > 1:
        from scipy.ndimage import zoom
        ratio = 1 / downsample_post
        # Ndimage zoom with spline interpolation of desired order
        reconstruction = zoom(reconstruction, (ratio, ratio, ratio), order=order)
        print(f'-- Downsampled in space | Reconstruction shape: {reconstruction.shape}')

    # Fixing artifacts in reconstruction space if necessary
    reconstruction = fix_edges_reconstruction(reconstruction, n=fix_edges_rec)
        
    # Saving the reconstructed volume into an mrc file
    if output_mrc is not None:
        save_mrc(reconstruction.astype(np.float32), output_mrc, overwrite=True)
        print(f'-- Reconstruction saved as mrc file.')
    else:
        return reconstruction.astype(np.float32)

    
def backproject_3d_imod(sinogram, theta, z_valid=None, cutoff=None, sigma=None):
    """
    Reconstructs a 3D volume from sinogram measurements using IMOD tilt program.
    https://bio3d.colorado.edu/imod/betaDoc/man/tilt.html
    
    This function creates two temporary files that are deleted at the end.
    One is for sinogram data that needs to be passed to tilt as a file.
    One is for reconstruction which tilt program creates and we load to RAM.
    """
    
    # Make sure IMOD titl CLI program is available
    assert which('tilt') is not None, 'Please install IMOD and make sure tilt program is in PATH.'
    
    # Tilt angles in tilt program friendly format
    angles = ' '.join(map(str, theta))
    
    # Computing the desired thickness of the reconstruction
    M = sinogram.shape[-1]
    thickness = round((z_valid[1] - z_valid[0]) * M) if z_valid is not None else M
    
    # Config of radial filtering (first value is cutoff, second value is falloff or sigma (when -FalloffIsTrueSigma is set)
    # We wanted to turn off this completely, but it seems not to be possible, this is as close as it gets:
    # - Setting cutoff to 0.5 does not turn off low-pass filtering as the docs say, so we set it to 1px (0px is invalid)
    # - Setting sigma to <1.0000 filters high frequencies, to >1.000 does not change the output so we keep 1.000 set using n pixels
    # - The final setting of RADIAL is: `-RADIAL "1 512" -FalloffIsTrueSigma` if M == 512
    cutoff = 1 if cutoff is None else cutoff
    sigma = M if sigma is None else sigma
    radial = f'{cutoff} {sigma}'
    
    # Use temporary mrc files so the subprocess can read input and write output
    tmp_id = time.time()
    tmp_projections_fname = f'~tmp_projections_{tmp_id}.mrc'
    tmp_reconstruction_fname = f'~tmp_reconstruction_{tmp_id}.mrc'
    save_mrc(sinogram.astype(np.float32), tmp_projections_fname)
    try:
        command = (
            'tilt '
            f'-input {tmp_projections_fname} '
            f'-output {tmp_reconstruction_fname} '
            f'-ANGLES "{angles}" '
            f'-THICKNESS {thickness} '
            f'-RADIAL "{radial}" '
            f'-FalloffIsTrueSigma'
        )
        return_code = subprocess.call(command, shell=True)
        assert return_code == 0, 'Call to IMOD tilt failed.'
        reconstruction = load_mrc(tmp_reconstruction_fname).swapaxes(0, 1)
    finally:
        os.remove(tmp_projections_fname)
        os.remove(tmp_reconstruction_fname)
            
    return reconstruction
    
    
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


# DEPRECATED
# def noise_projections(input_mrc, style_mrc, output_mrc, mean=0.0, std=0.4, clip_outliers=(0.0, 1.0), seed=0):
#     """
#     Adds Gaussian noise of specified mean and std to input_mrc which is then 
#     rescaled per tilt to match the mean and std of the tilts in style mrc.
    
#     Parameters
#     ----------
#     input_mrc: str
#         Path to a mrc file containig noiseless projections.
#     style_mrc: str
#         Path to a mrc file containig style projections. Ideally a real sinogram 
#         from the data set on which we want to later predict.
#     output_mrc: str
#         Path to a mrc file where the result is going to be stored.
#     mean: float, default=0.0
#         Mean of the Gaussian noise.
#     std: float, default=0.4
#         Standard deviation of the Gaussian noise.
#     clip_outliers: 2-tuple of floats, default=(0.0, 1.0)
#         Clip data outside of the specified portion.
#     seed: 
#         Random seed used to generate the noise.
#     """
    
#     # Load input projections
#     volume = load_mrc(input_mrc)
#     style = load_mrc(style_mrc)
    
#     # Generate random noise
#     rng = np.random.default_rng(seed=seed)
#     noise = rng.normal(loc=mean, scale=std, 
#                        size=volume.size).reshape(volume.shape)
    
#     # Scaling per tilt based on style (bigger the abs(angle), longer the trajectory)
#     volume  = match_mean_std(volume, style)  
    
#     # Scale between [0, 1] before adding the noise
#     volume = normalize(volume)  
    
#     # Add the noise
#     volume_noisy = volume + noise
    
#     # Remove outliers
#     volume_noisy = np.clip(volume_noisy, *get_clim(volume_noisy, *clip_outliers)) 
    
#     # Scale back (per tilt) to match style
#     volume_noisy = match_mean_std(volume_noisy, style)  
    
#     # Save output
#     save_mrc(volume_noisy.astype(np.float32), output_mrc, overwrite=True)