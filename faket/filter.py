import numpy as np


def hann1d(size):
    """
    1D Hann filter as implemented in Radontea package
    https://github.com/RI-imaging/radontea/blob/01fb924b2a241914328c6526ced7248807f3adea/radontea/_alg_bpj.py#L138
    Returns fftshifted filter (high-frequencies in the center).
    """
    kx = 2 * np.pi * np.abs(np.fft.fftfreq(int(size)))  # ramp
    kx[1:] = kx[1:] * (1 + np.cos(kx[1:])) / 2 # hann
    return kx


def ramp1d(size, crowtherFreq=None, radiusCutoff=None):
    """
    Implementation of a 1D ramp filter with support for CrowtherFreq 
    and CircularCutoff. Returns fftshifted (high-frequencies in the center)
    filter ready to be mulitplied with a 1D DFT of an image.
    
    Visualize:
    ----------
    plt.plot(ramp1d(512, 150, 250))
    
    Example:
    --------
    filtered = ifft(fft(image, axis=-1) * ramp1d(sizeX, 50, 220).reshape(1, -1), axis=-1).real
    # If you want to switch axes, in case of a 2d image, set axis=-2 to both fft and ifft 
    # and flip the position of arguments of reshape.
    """
    nyquist = size // 2
    radius = np.abs(np.arange(size) - nyquist)
    unit = crowtherFreq or nyquist
    assert unit <= nyquist, f'Max crowtherFreq is nyquist {nyquist}.'
    
    # Filter
    ramp = radius / unit
    ramp[ramp > 1] = 1
    
    # Circular cut (if radiusCutoff is None, no cut)
    radiusCutoff = radiusCutoff or nyquist * 2
    ramp[radius > radiusCutoff] = 0
    return np.fft.fftshift(ramp)


def circular2d(sizeX, sizeY, radiusCutoff=None):
    """
    Implementation of a 2D circular filter. 
    Returns fftshifted filter (high-frequencies in the center).
    Every value inside the radiusCutoff is 1 and the rest is 0.
    """
    # Centered grids
    shape = np.array([sizeX, sizeY])
    ii = np.abs(np.indices(shape) - (shape // 2).reshape(2, 1, 1))
    radius = np.sqrt(ii[0] ** 2 + ii[1] ** 2)
    f = np.ones(shape)
    nyquist = np.max([sizeX, sizeY]) / 2
    radiusCutoff = radiusCutoff or nyquist
    f[radius >= radiusCutoff + 1] = 0
    return np.fft.fftshift(f)


def gauss2d(sizeX, sizeY, sigmaX, sigmaY, radiusCutoff=None):
    """
    Adjusted from: 
    https://stackoverflow.com/a/27928469/8691571
    https://stackoverflow.com/a/56923189/8691571
    
    Implementation of a 2D Gaussian filter. 
    Returns fftshifted filter (high-frequencies in the center).
    """
    y, x = np.mgrid[-sizeX // 2 + 1:sizeX // 2 + 1, 
                    -sizeY // 2 + 1:sizeY // 2 + 1]
    
    out = np.fft.fftshift(
        np.exp(-(x ** 2. / (2. * sigmaX ** 2.) + y ** 2. / (2. * sigmaY ** 2.))))

    # Circular cut (if radiusCutoff is None, no cut)
    if radiusCutoff is not None:
        c = circular2d(sizeX, sizeY, radiusCutoff=radiusCutoff)
        out *= c
    
    return out


########################################################
# Filters made from products of previous simpler filters
########################################################


def rampShrec(sizeX, sizeY, crowtherFreq=None, radiusCutoff=None):
    """
    Filter SHREC claimed to have used, but it seems it is not
    what they actually used, therefore we implemented `approxShrec`.
    Returns fftshifted filter (high-frequencies in the center).
    """
    ramp = ramp1d(sizeX, crowtherFreq, radiusCutoff=None)
    ramp = np.broadcast_to(ramp, sizeX, sizeY)
    return ramp * circular2d(sizeX, sizeY, radiusCutoff)


def approxShrec(sizeX, sizeY):
    """
    This is a filter made by a product of 2D gaussian filter
    with 1D ramp filter broadcasted to 2D and 2D circular filter.
    Returns fftshifted filter (high-frequencies in the center).
    
    This filter is a reverse-engineered filter from SHREC21.
    A bit odd constant `g_lift` was added to better match the 
    frequency spectrum of SHREC21 reconstructions. The filter
    in SHREC21 is probably a bit different, but this is as close
    as we could get without seeing the source code and config.
    ¯\_(ツ)_/¯
    """

    # Important if sizeX != sizeY
    size = max(sizeX, sizeY)
    nyquist = size // 2
    
    # Empirically estimated config based on a visualization
    # of a SHREC reconstruction in real & Fourier spaces
    sigmaX = int(0.34 * size)
    sigmaY = int(0.20 * size)
    g_lift = 0.33
    f_crowtherFreq = int(0.61 * nyquist)
    
    # 2D Gaussian filter shifted up by `g_lift`
    # Range from `g_lift` to 1 + `g_lift`
    g = gauss2d(size, size, sigmaX, sigmaY) + g_lift
    
    # 1D ramp filter broadcasted to 2D
    # Range from 0 to 1
    f = ramp1d(size, crowtherFreq=f_crowtherFreq)
    f = np.broadcast_to(f, (size, size))
    
    # Circular filter
    # Only zeros and ones
    c = circular2d(size, size, radiusCutoff=nyquist)

    # Filter
    # Range from 0 to ?
    out = f * g * c
    out /= out.max()  # Forcing to range from 0 to 1
    
    # Slice to desired shape anchored to center
    xs, ys = (size - sizeX) // 2, (size - sizeY) // 2
    xe, ye = xs + sizeX, ys + sizeY
    return out[xs:xe, ys:ye]
