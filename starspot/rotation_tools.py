"""
A set of functions for measuring rotation periods and plotting the results.
"""


import os
import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import kplr
from astropy.stats import BoxLeastSquares
import scipy.signal as sps

plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)

def transit_mask(self, t0, duration, porb):
    """
    Mask out transits

    Args:
        t0 (float): The reference time of transit. For Kepler data you may
            need to subtract 2454833.0 off this number.
        duration (float): The transit duration.
        porb (float): The planet's orbital period.
    """
    _t0 = float(t0) % porb
    _duration = float(duration) / 24.
    m = np.abs((self.time - _t0 + 0.5*self.porb) \
                % self.porb - 0.5*self.porb) < 1.5*_duration
    return m

def download_light_curves(KID, download_path, lcpath):
    """
    Download Kepler light curves

    Args:
        KID (int): The Kepler ID.
        download_path (str): The path to where the light curves will be
            downloaded.
        lcpath (str): The path to where the light curves are stored (same as
            download_path with /.kplr/data/lightcurves/<KID>, where KID is a 9
            digit number.)

    Returns:
        time (array): The time array.
        flux (array): The flux array.
        flux_err (array): The flux uncertainty array.
    """
    client = kplr.API(data_root=download_path)
    star = client.star(str(int(KID)).zfill(9))
    star.get_light_curves(fetch=True, short_cadence=False);
    time, flux, flux_err = load_kepler_data(lcpath)
    return time, flux, flux_err


def load_kepler_data(LC_DIR):
    """
    Load data and join quarters together.

    Args:
        LC_DIR (str): The path to where the light curves are stored (will
            end in /.kplr/data/lightcurves/<KID>, where KID is a 9 digit
            number.)

    Returns:
        time (array): The time array.
        flux (array): The flux array.
        flux_err (array): The flux uncertainty array.
    """

    # The names of the light curve fits files.
    fnames = sorted(glob.glob(os.path.join(LC_DIR, "*llc*fits")))

    # load and median normalize the first quarter
    time, flux, flux_err = load_and_normalize(fnames[0])

    # Concatenate with the remaining median normalized quarters
    for fname in fnames[1:]:
        x, y, yerr = load_and_normalize(fname)
        time = np.concatenate((time, x))
        flux = np.concatenate((flux, y))
        flux_err = np.concatenate((flux_err, yerr))

    return time, flux, flux_err


def load_and_normalize(fname):
    """
    Load one quarter, remove bad points and median normalize it.

    Args:
        fname (str): The path to and name of a light curve fits file.

    Returns:
        x (array): The time array.
        y (array): The flux array.
        y_err (array): The flux uncertainty array.
    """

    # Load the data and pull out the time and flux arrays.
    hdulist = fits.open(fname)
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]

    # Mask out bad quality points and NaNs.
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
            (q == 0)
    x = time[m]

    # Median normalize
    med = np.median(flux[m])
    y = flux[m]/med - 1
    yerr = flux_err[m]/med

    return x, y, yerr


def dan_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    Args:
        x (array): The time series. If multidimensional, set the time axis
            using the ``axis`` keyword argument and the function will be
            computed for every other axis.
        axis (Optional[int]): The time axis of ``x``. Assumed to be the first
            axis if not specified.
        fast (Optional[bool]): If ``True``, only use the largest ``2^n``
            entries for efficiency. (default: False)

    Returns:
        acf (array): The acf array.
    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0
    return acf / acf[m]


def interp(x_gaps, y_gaps, interval):
    """
    Interpolate the light curve

    Args:
        x_gaps (array): The time array with gaps.
        y_gaps (array): The flux array with gaps.
        interval (float): The grid to interpolate to.

    Returns:
        time (array): The interpolated time array.
        flux (array): The interpolated flux array.
    """
    f = spi.interp1d(x_gaps, y_gaps, kind="zero")
    x = np.arange(x_gaps[0], x_gaps[-1], interval)
    return x, f(x)


def simple_acf(x_gaps, y_gaps, interval):
    """
    Compute an autocorrelation function and a period.

    Applies interpolation, smoothing and peak detection to estimate a
    rotation period.

    Args:
        x_gaps (array): The time array.
        y_gaps (array): The flux array.
        interval (Optional[float]): The time interval between successive
            observations. The default is Kepler cadence.

    Returns:
        lags (array): The array of lag times in days.
        acf (array): The autocorrelation function.
        period (float): The period estimated from the highest peak in the ACF.
    """

    # First of all: interpolate to an evenly spaced grid
    x, y = interp(x_gaps, y_gaps, interval)

    # fit and subtract straight line
    AT = np.vstack((x, np.ones_like(x)))
    ATA = np.dot(AT, AT.T)
    m, b = np.linalg.solve(ATA, np.dot(AT, y))
    y -= m*x + b

    # perform acf
    acf = dan_acf(y)

    # create 'lags' array
    lags = np.arange(len(acf))*interval

    N = len(acf)
    double_acf, double_lags = [np.zeros((2*N)) for i in range(2)]
    double_acf[:N], double_lags[:N] = acf[::-1], -lags[::-1]
    double_acf[N:], double_lags[N:] = acf, lags
    acf, lags = double_acf, double_lags

    # smooth with Gaussian kernel convolution
    Gaussian = lambda x, sig: 1./(2*np.pi*sig**.5) * np.exp(-0.5*(x**2)/
                                                            (sig**2))
    conv_func = Gaussian(np.arange(-28, 28, 1.), 9.)
    acf_smooth = np.convolve(acf, conv_func, mode='same')

    # just use the second bit (no reflection)
    acf_smooth, lags = acf_smooth[N:], lags[N:]

    # cut it in half
    m = lags < max(lags)/2.
    acf_smooth, lags = acf_smooth[m], lags[m]

    # ditch the first point
    acf_smooth, lags = acf_smooth[1:], lags[1:]

    # find all the peaks
    peaks = np.array([i for i in range(1, len(lags)-1)
                     if acf_smooth[i-1] < acf_smooth[i] and
                     acf_smooth[i+1] < acf_smooth[i]])


    # find the highest peak
    if len(peaks):
        m = acf_smooth == max(acf_smooth[peaks])
        highest_peak = acf_smooth[m][0]
        period = lags[m][0]
    else:
        period = 0.

    return lags, acf_smooth, period

def find_and_mask_transits(time, flux, flux_err, periods, durations,
                           nplanets=1):
    """
    Iteratively find and mask transits in the flattened light curve.

    Args:
        time (array): The time array.
        flux (array): The flux array. You'll get the best results
            if this is flattened.
        flux_err (array): The array of flux uncertainties.
        periods (array): The array of periods to search over for BLS.
            For example, periods = np.linspace(0.5, 20, 10)
        durations (array): The array of durations to search over for BLS.
            For example, durations = np.linspace(0.05, 0.2, 10)
        nplanets (Optional[int]): The number of planets you'd like to search for.
            This function will interatively find and remove nplanets. Default is 1.

    Returns:
        transit_masks (list): a list of masks that correspond to the in
            transit points of each light curve. To mask out transits do
            time[~transit_masks[index]], etc.
    """

    cum_transit = np.ones(len(time), dtype=bool)
    _time, _flux, _flux_err = time*1, flux*1, flux_err*1

    t0s, durs, porbs = [np.zeros(nplanets) for i in range(3)]
    transit_masks = []
    for i in range(nplanets):
        bls = BoxLeastSquares(t=_time, y=_flux, dy=_flux_err)
        bls.power(periods, durations)

        print("periods")
        periods = bls.autoperiod(durations, minimum_n_transit=3,
                                 frequency_factor=5.0)
        print("results")
        results = bls.autopower(durations, frequency_factor=5.0)

        # Find the period of the peak
        print("find_period")
        period = results.period[np.argmax(results.power)]

        print("extract")
        # Extract the parameters of the best-fit model
        index = np.argmax(results.power)
        porbs[i] = results.period[index]
        t0s[i] = results.transit_time[index]
        durs[i] = results.duration[index]

        # # Plot the periodogram
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # ax.plot(results.period, results.power, "k", lw=0.5)
        # ax.set_xlim(results.period.min(), results.period.max())
        # ax.set_xlabel("period [days]")
        # ax.set_ylabel("log likelihood")

        # # Highlight the harmonics of the peak period
        # ax.axvline(period, alpha=0.4, lw=4)
        # for n in range(2, 10):
        #     ax.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed")
        #     ax.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")
        # plt.show()

        # plt.plot(_time, _flux, ".")
        # plt.xlim(1355, 1360)

        print("mask")
        in_transit = bls.transit_mask(_time, porbs[i], 2*durs[i], t0s[i])
        transit_masks.append(in_transit)
        _time, _flux, _flux_err = _time[~in_transit], _flux[~in_transit], \
            _flux_err[~in_transit]

    return transit_masks, t0s, durs, porbs

def apply_masks(time, flux, flux_err, transit_masks):
    """
    Apply transit masks to the unflattened light curve.

    Args:
        time (array): The time array.
        flux (array): The flux array
        flux_err (array): The flux_err array.
        transit_masks (list): A list of transit masks.

    Returns:
        masked_time (array): The masked time array.
        masked_flux (array): The masked flux array.
        masked_flux_err (array): The masked flux_err array.

    """
    masked_time = time*1
    masked_flux, masked_flux_err = flux*1, flux_err*1
    for i in range(len(transit_masks)):
        masked_time = masked_time[~transit_masks[i]]
        masked_flux = masked_flux[~transit_masks[i]]
        masked_flux_err = masked_flux_err[~transit_masks[i]]

    return masked_time, masked_flux, masked_flux_err


def butter_bandpass_filter(flux, lowcut, fs, order=3):
    """
    Apply a Butterworth high-pass filter.

    Args:
        flux (array): The flux array.
        lowcut (float): The frequency cut off.
        fs (array): The frequency array.
        order (Optional[int]): The order of the Butterworth filter. Default
            is 3.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sps.butter(order, lowcut, btype='highpass')
    y = sps.lfilter(b, a, flux)
    return y


def get_peak_statistics(x, y, sort_by="height"):
    """
    Get the positions and height of peaks in an array.

    Args:
        x (array): the x array (e.g. period or lag).
        y (array): the y array (e.g. power or ACF).
        sort_by (str): The way to sort the peak array. if "height", sort peaks
            in order of height, if "position", sort peaks in order of
            x-position.

    Returns:
        x_peaks (array): the peak x-positions in descending height order, or
            ascending x-position order.
        y_peaks (array): the peak heights in descending height order, or
            ascending x-position order.
    """

    # Array of peak indices
    peaks = np.array([i for i in range(1, len(y)-1) if y[i-1] <
                      y[i] and y[i+1] < y[i]])

    # extract peak values
    x_peaks = x[peaks]
    y_peaks = y[peaks]

    # sort by height
    if sort_by == "height":
        inds = np.argsort(y_peaks)
        x_peaks, y_peaks = x_peaks[inds][::-1], y_peaks[inds][::-1]

    # sort by position
    elif sort_by == "position":
        inds = np.argsort(x_peaks)
        x_peaks, y_peaks = x_peaks[inds], y_peaks[inds]

    return x_peaks, y_peaks
