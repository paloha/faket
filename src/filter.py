import numpy as np

def ramp2d(sizeX, sizeY, crowtherFreq=None, radiusCutoff=None, angularCutoff=None):
    """
    2D ramp filter with support for Crowther frequency, 
    radius cutoff, and angular cutoff. Returns fftshifted
    2D array containing the filter which can be directly
    mulitplied with a 2D DFT of an image.

    angularCutoff, numerical or 2-tuple
    
    Example of usage:
    filtered = ifft2(fft2(image) * ramp2d(sizeX, sizeY, 20, 180, 82)).real
    filtered = ifft2(fft2(image) * ramp2d(sizeX, sizeY, 20, 180, (8, 82))).real
    """

    # Centered grids
    shape = np.array([sizeX, sizeY])
    ii = np.abs(np.indices(shape) - (shape // 2).reshape(2, 1, 1))

    # Important frequencies
    nyquist = np.max([sizeX, sizeY]) / 2
    unit = crowtherFreq or nyquist
    assert unit <= nyquist, f'Max crowtherFreq is nyquist {nyquist}.'
    
    # Filter
    radius = np.sqrt(ii[0] ** 2 + ii[1] ** 2)
    ramp =  radius / unit
    ramp[ramp > 1] = 1
    
    # Angular cutoff
    if angularCutoff is not None:
        ang = np.arctan2(ii[0], ii[1])
        if isinstance(angularCutoff, tuple):
            a, b =  angularCutoff
            assert a >= 0 and a <= 90, 'Choose angle betwen 0° and 90°'
        else:
            b = angularCutoff
            assert b >= 45 and b <= 90, 'Choose angle between 45° and 90°'
            a = 90 - b 
        ramp[ang < np.deg2rad(a)] = 0
        ramp[ang > np.deg2rad(b)] = 0
        
    
    # Circular cut (if radiusCutoff is None, no cut)
    if radiusCutoff is not None:
        ramp[radius >= radiusCutoff] = 0
    
    return np.fft.fftshift(ramp)


def ramp1d(size, crowtherFreq=None, radiusCutoff=None):
    """
    Implementation of a ramp filter with support for CrowtherFreq 
    and CircularCutoff. Returns fftshifted filter ready to be
    mulitplied with a 1D DFT of an image.
    
    filtered = ifft(fft(image, axis=-1) * ramp1d(sizeX, 50, 220).reshape(1, -1), axis=-1).real
    # If you want to switch axes, in case of a 2d image, set axis=-2 to both fft and ifft 
    # and flip the position of arguments of reshape.
    """
    nyquist = size // 2
    radius = np.abs(np.arange(size) - nyquist)
    unit = crowtherFreq or nyquist
    assert unit <= nyquist, f'Max crowtherFreq is nyquist {nyquist}.'
    
    # Filter
    ramp =  radius / unit
    ramp[ramp > 1] = 1
    
    # Circular cut (if radiusCutoff is None, no cut)
    radiusCutoff = radiusCutoff or nyquist * 2
    ramp[radius >= radiusCutoff] = 0
    return np.fft.fftshift(ramp)


def hannFilter(size):
    """
    Hann filter as implemented in Radontea package
    https://github.com/RI-imaging/radontea/blob/01fb924b2a241914328c6526ced7248807f3adea/radontea/_alg_bpj.py#L138
    """
    kx = 2 * np.pi * np.abs(np.fft.fftfreq(int(size)))  # ramp
    kx[1:] = kx[1:] * (1 + np.cos(kx[1:])) / 2 # hann
    return kx


def rampShrec(sizeX, sizeY, crowtherFreq=None, radiusCutoff=None):
    """
    Filter like in SHREC
    """
    ramp = ramp1d(sizeX, crowtherFreq, radiusCutoff=None)
    ramp = np.broadcast_to(ramp, sizeX, sizeY)
    return ramp * circularFilter(sizeX, sizeY, radiusCutoff)


def circularFilter(sizeX, sizeY, radiusCutoff=None):
    # Centered grids
    shape = np.array([sizeX, sizeY])
    ii = np.abs(np.indices(shape) - (shape // 2).reshape(2, 1, 1))
    radius = np.sqrt(ii[0] ** 2 + ii[1] ** 2)
    f = np.ones(shape)
    nyquist = np.max([sizeX, sizeY]) / 2
    radiusCutoff = radiusCutoff or nyquist
    f[radius >= radiusCutoff] = 0
    return np.fft.fftshift(f)
