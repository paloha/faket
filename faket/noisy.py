import numpy as np
from .data import load_mrc, save_mrc
from .data import match_mean_std
from .data import standardize
from .data import standardize_per_tilt


def find_r_std(noisy, clean, start=0.0, stop=1.0, min_step=0.01, debug=False):
    """
    Find value of `r` that minimizes `f(r)` which is a function that subtracts 
    `clean` with μ=0 and σ=r from `noisy` with μ=0 and σ=1 and returns 
    the r and σ of the result.
    
    The idea is to scale `clean` using r such that it fits the signal in `noisy` as much
    as possible. If it would fit perfectly, `noisy - scaled_clean` has no signal
    at all and contians only noise. That does happen only approximately.
    
    Parameters
    ----------
    noisy: 2D array containing one projection from TEM
    clean: 2D array containing simulated noiseless projection with content matching `noisy`
    start: float, left boundary of the interaval that is searched for plausible r
    stop: float, right boundary of the interval that is searched for plausible r
    min_step: float,  stop contidion for bisection
    debug: bool, if True does not bisect but plots results of r on a grid
    
    Notice
    ------
    Use this function on a 2D tilt data (not the whole 3D sinogram).
    
    Returns
    -------
    r: 
        the factor that scales standardized clean to fit standardized noisy
        I.e. it is the std of clean before the noise will be added
    subtracted.std(): the standard deviation of the estimated noise
    subtracted: the volume containing the estimated noise
    noisy_μ, noisy_σ, clean_μ, clean_σ: statistics of the original volumes
    """
    assert start < stop
    assert 1e-6 < min_step < (stop - start) / 2
    
    # Standardize to μ = 0, σ = 1
    noisy_μ, noisy_σ = noisy.mean(), noisy.std()
    noisy = standardize(noisy)  # (noisy - noisy_μ) / noisy_σ
    
    # Standardize to μ = 0, σ = 1
    clean_μ, clean_σ = clean.mean(), clean.std()
    clean = standardize(clean)  # (clean - clean_μ) / clean_σ
    
    # Compute subtracted_std after scaling the clean
    def f(r):
        # Since noisy and clean are already standardized
        # the μ of clean = 0 and thus we can just multiply
        # it with r to change its std without influencing its μ.
        return (noisy - r * clean).std()
    
    # ---------------------------------------------
    if debug:
        import matplotlib.pyplot as plt
        rs = np.arange(start, stop, min_step)
        stds = []
        for r in rs:
            subtracted = noisy - r * clean
            subtracted_std = subtracted.std()
            stds.append(subtracted_std)
            plt.imshow(subtracted)
            plt.title(f'r: {r}, Subt.σ: {subtracted_std:.4f} Subt.μ: {subtracted.mean():.4f}')
            plt.show()
        plt.plot(rs, stds)
        return
    # ---------------------------------------------
    
    # Search for r that minimizes subtracted_std
    r = search(start, stop, min_step, f)
    subtracted = noisy - r * clean
    return r, subtracted.std(), subtracted, noisy_μ, noisy_σ, clean_μ, clean_σ
    

def search(a, b, e, f):
    """
    a: start of the interval
    b: end of the interval
    e: precision
    f: function 
    """
    def d(x):
        return (f(x + e) - f(x)) / e

    while True:
        new_r = (a + b) / 2
        if (b - new_r) < e:                
            break  # Step too small
        r = new_r
        dr = d(r)
        if dr == 0:
            break
        if dr < 0:
            a = r
        else:
            b = r
    return r

    
def get_curves(projections_style, projections_noiseless):
    """
    Iterates over all tilts of the sinogram and uses find_r_std()
    to estimate the statistics of the noise. Results for each
    tilt are stored in the return dictionary.
    """
    # Find r for each tilt separately in a for loop
    curves = {'rs': [], 'stds': [], 'noisy_μs': [], 'noisy_σs': [], 'clean_μs': [], 'clean_σs': []}
    for i in range(projections_style.shape[0]):
        r, std, noise, noisy_μ, noisy_σ, clean_μ, clean_σ = find_r_std(
            projections_style[i], projections_noiseless[i], start=0, stop=1.0, min_step=0.00001)
        curves['rs'].append(r)
        curves['stds'].append(std)
        curves['noisy_μs'].append(noisy_μ)
        curves['noisy_σs'].append(noisy_σ)
        curves['clean_μs'].append(clean_μ)
        curves['clean_σs'].append(clean_σ)
    return curves


def estimate_noise_curves(projections_noisy_paths, projections_clean_paths):
    """
    Opens mrc files at the provided paths, estimates the noise stats
    for each tilt in each file from projections_noisy_paths using
    the matching noiseless content from projections_clean_paths.
    Returns an array of dictionaries. Each dict contains curves
    relevant to the noise estimation per tilt. Check the docstring
    of get_curves() for more details.
    """
    n_projections = len(projections_noisy_paths)
    assert n_projections == len(projections_clean_paths)
    assert n_projections >= 1
    
    # Computing the noise statistics for each tilt for each tomogram
    all_curves = []
    for n in range(n_projections):
        clean = load_mrc(projections_clean_paths[n])
        noisy = load_mrc(projections_noisy_paths[n])
        assert clean.ndim == 3
        assert clean.shape == noisy.shape
        all_curves.append(get_curves(noisy, clean))
    return all_curves 


def polyfit(curve, order=0):
    """
    Fits a polynomial of desired order to the curve.
    order = -1 returns the original curve.
    """
    if order == -1:
        return curve
    x = np.arange(len(curve))
    p = np.poly1d(np.polyfit(x, curve, order))
    return p(x)


def aggregate(curves, order=0, n='all'):
    """
    Returns function of n
    """
    assert np.asarray(curves).ndim == 2, 'Provide 2D array (n_projections x n_tilts)'
    if n == 'all':
        return np.mean([polyfit(curve, order=order) for curve in curves], axis=0)
    assert isinstance(n, int), 'n must be an integer'
    assert 0 <= n <= len(curves[0]), f'n must be >= 0 and <= {len(curves[0])}'
    return polyfit(curves[n], order=order)


def get_noisy(projections_noiseless, r, std, projections_style=None, 
              style_means=None, style_stds=None, seed=None):
    """
    projections_noiseless will be rescaled to μ=0, σ=r and a Gaussian noise
    with μ=0, σ=std will be added to it. Each tilt is rescaled to match the 
    μ and σ of each tilt in the reference sinogram in projections_style.
    If the projections_style is None the μs and σs of each tilt will be taken
    from means and stds arrays. No noise stats estimation is done in this step.
    
    Parameters
    ----------
    projections_noiseless: 3D sinogram
    r: float or 1D array, result of noise stats estimation
    std: float or 1D array, result of noise stats estimation    
    projections_style: 3D reference sinogram for matching μ and σ tilt-wise
    style_means: 1D array, tilt-wise μ specified exactly not taken from projections_style
    style_stds: 1D array, tilt-wise σ specified exactly not taken from projections_style
    """
    assert projections_noiseless.ndim == 3, 'Inputs must be 3D sinograms'
    r = np.asarray(r)
    std = np.asarray(std)
    assert r.ndim == std.ndim, 'Both r and std must be either floats or 1D arrays'
    if r.ndim > 0:
        r = r.reshape(r.size, 1, 1)
        std = std.reshape(std.size, 1, 1)
    
    x = standardize_per_tilt(projections_noiseless) * r
    rng = np.random.default_rng(seed=seed)
    x += (rng.normal(loc=0, scale=1, size=np.prod(x.shape)).reshape(x.shape) * std)
    
    if projections_style is None:
        assert style_means is not None and style_stds is not None, \
        'Provide either projections_style or style_means and style_stds.'
        return match_mean_std(x, means=style_means, stds=style_stds)
    else:
        assert projections_style.ndim == 3, 'Inputs must be 3D sinograms'
        return match_mean_std(x, projections_style)


def noise_projections(input_mrc, output_mrc, r, std, style_mrc=None, style_means=None, style_stds=None, seed=None):
    """
    Adds Gaussian noise of specified mean and std to input_mrc which is then 
    rescaled per tilt to match the mean and std of the tilts in style mrc.
    
    Parameters
    ----------
    input_mrc: str
        Path to a mrc file containig noiseless projections.
    output_mrc: str
        Path to a mrc file where the result is going to be stored.
    r: float or 1D array
        Scale of the input array before the Gaussian noise is added.
        This is a result of noise estimation from data.
    std: float or 1D array
        Standard deviation of the Gaussian noise (overall or per-tilt).
        This is a result of noise estimation from data.
    style_mrc: str or None
        Path to a mrc file containig style projections. Ideally a real sinogram 
        from the data set on which we want to later predict. If None
        style_means and style_stds must be provided. Used to adjust
        the mean and standard deviation of each output tilt.
    style_means: 1D array or None
        When style_mrc is not None this is ignored. Otherwise an array
        exactly specifying target mean of each tilt of the output.
    style_stds: 1D array or None
        When style_mrc is not None this is ignored. Otherwise an array
        exactly specifying target standard deviation of each tilt of the output.
    seed: int or None
        Random seed used to generate the noise.
    """
    
    # Load input projections
    volume = load_mrc(input_mrc)
    
    # Load style projections
    style = None if style_mrc is None else load_mrc(style_mrc)
    
    # Generate noisy volume
    volume_noisy = get_noisy(
        volume, r=r, std=std,
        projections_style=style, 
        style_means=style_means,
        style_stds=style_stds, 
        seed=seed)

    # Save output
    save_mrc(volume_noisy.astype(np.float32), output_mrc, overwrite=True)