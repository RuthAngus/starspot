import numpy as np
import scipy.signal as sps


def filter_sigma_clip(x, y, nsigma=3, window_length=49, polyorder=3):
    """ Sigma clip a light curve using a Savitzky-Golay filter.

    Args:
        x (array): The x-data array.
        y (array): The y-data array.
        nsigma (Optional[float]): The number of sigma to clip on.
            Default = 3.
        window_length (Optional[float]): The filter window length.
            Must be an odd number. Default = 49.
        polyorder (Optional[float]): The polynomial order of the filter.
            Default = 3.

    Returns:
        smooth (array): The smoothed data array.
        mask (array): The mask used for clipping.

    """

    # Smooth the data with a Savitsky-Golay filter.
    smooth = sps.savgol_filter(y, window_length, polyorder)
    resids = y - smooth

    # Clip
    mask = sigma_clip(resids, nsigma=nsigma)
    return mask, smooth


def sigma_clip(x, nsigma=3):
    """
    Sigma clipping for 1D data.

    Args:
        x (array): The data array. Assumed to be Gaussian in 1D.
        nsigma (float): The number of sigma to clip on.

    Returns:
        newx (array): The clipped x array.
        mask (array): The mask used for clipping.
    """

    m = np.ones(len(x)) == 1
    newx = x*1
    oldm = np.array([False])
    while sum(oldm) != sum(m):
        oldm = m*1
        sigma = np.std(newx)
        m &= np.abs(np.median(newx) - x)/sigma < nsigma
        newx = x[m]
    return m
