"""
Phase dispersion minimisation algorithm, Stellingwerf (1978).
"""

import numpy as np
import scipy.optimize as sco


def sj2(x, meanx, N):
    """
    The variance of a set of data points in one bin.

    Args:
        x (array): The flux array (defined as x in Stellingwerf, 1978).
        meanx (float): The mean of x.
        N (int): The number of data points

    Returns:
        The variance of x.

    """

    return sum((x - meanx)**2)/(N-1)


def s2(nj, sj2, M):
    """
    Overall variance for the binned data. The s2 statistic (equation 2 of
    Stellingwerf, 1978).

    Args:
        nj (array): Number of data points per sample/bin (j = 1 to M).
        sj2 (array): Variance of each sample/bin (j = 1 to M).
        M (int): Number of samples/bins.

    Returns:
        The overall variance of the samples/bins.

    """

    return sum((nj - 1)*sj2)/(sum(nj) - M)


def calc_phase(p, t):
    """
    Calculate the phase array.

    Args:
        t (array): The time array (in days).
        p (float): Period (in days)

    Returns
        phase (array): The phase array.
    """

    return (t % p)/p


def phase_bins(nbins, phase, x):
    """
    Bin data by phase.

    Args:
        nbins (int): The number of bins.
        phase (float): The phase array.
        x (array): The flux array.

    Returns
        phase_bins (array): The phase bin edges (len = nbins + 1).
        x_means (array): The mean flux in each bin.
        Ns (array): The number of points in each bin.
        per_bin_variances (array): The variance in each bin.
        x_binned (list): A list of lists of flux. A list for each bin.
        phase_binned (list): A list of lists of phases. A list for each bin.

    """

    min_phase, max_phase = 0, 1
    phase_bins = np.linspace(min_phase, max_phase, nbins + 1)
    x_binned, phase_binned = [], []
    x_means, Ns, per_bin_variances = [np.empty(nbins) for i in range(3)]
    for j in range(nbins):
        m = (phase_bins[j] < phase) * (phase < phase_bins[j + 1])
        Ns[j] = len(x[m])
        x_means[j] = np.mean(x[m])
        x_binned.append(x[m])
        phase_binned.append(phase[m])
        per_bin_variances[j] = sj2(x[m], x_means[j], Ns[j])

    return x_means, phase_bins, \
        Ns, per_bin_variances, \
        x_binned, phase_binned


def phi(nbins, p, t, x):
    """
    Calculate the phi statistic in Stellingwerf (1978).

    Args:
        nbins (int): The number of bins to use to calculate phase dispersion.
        p (float): The period to calculate the Phi statistic for.
        t (array): The time array.
        x (array): The flux array.

    Returns:
        phi (float): The phi statistic. Ratio of the phase-binned variance to
        the total variance.

    """

    phase = calc_phase(p, t)
    x_means, phase_bs, Ns, sj2s, xb, pb = phase_bins(nbins, phase, x)
    total_binned_variance_s2 = s2(Ns, sj2s, nbins)
    total_variance = sj2(x, np.mean(x), len(x))

    return total_binned_variance_s2/total_variance


def gaussian(pars, x):
    """
    A Gaussian, with a baseline of b.
    """
    A, b, mu, sigma = pars
    # return b + A/(np.sqrt(2*np.pi)*sigma**2) \
    return b + A \
        * np.exp(-.5*(x - mu)**2/sigma**2)


def nll(pars, x, y):
    model = gaussian(pars, x)
    return sum(.5*(y - model)**2)


def estimate_uncertainty(period_grid, phi, best_period):
    """
    Fit a Gaussian to the phase dispersion trough around the minimum and
    report the sigma.

    Args:
        period_grid (array): The period grid.
        phi (array): The phase dispersions over periods.

    Returns:
        mu, sigma (float): The mean and standard deviation of the Gaussian
        fit.

    """

    # Peak finder
    def peaks(y):
        return np.array([i for i in range(1, len(y)-1) if y[i-1] <
                         y[i] and y[i+1] < y[i]])

    # limit to a section around the trough.
    # Find peaks adjacent to the dip.
    pks = peaks(phi)
    peak_pos = period_grid[pks]
    lower_peaks = peak_pos < best_period
    upper_peaks = peak_pos > best_period

    # Catch occasions where no adjacent peaks are found
    if hasattr(peak_pos[lower_peaks], "len"):
        lower_lim = peak_pos[lower_peaks][-1]
    else:
        lower_lim = min(period_grid)
    if hasattr(peak_pos[upper_peaks], "len"):
        upper_lim = peak_pos[upper_peaks][0]
    else:
        upper_lim = max(period_grid)

    result, dip_x, dip_y = fit_gaussian(
        period_grid, phi, upper_lim, lower_lim, best_period)

    # If the uncertainty is close to 50%, the fit might have gone wrong:
    # fit a Gaussian to a window that is a percentage of the period.
    if result.x[-1]/best_period > .4:
        print("fitting to a limited range")
        upper_lim = best_period + .3*best_period
        upper_lim = best_period - .3*best_period
        result, dip_x, dip_y = fit_gaussian(
            period_grid, phi, upper_lim, lower_lim, best_period)

    a, b, mu, sigma = result.x

    return sigma, mu, a, b


def fit_gaussian(period_grid, phi, upper_lim, lower_lim, best_period):
    m = (lower_lim < period_grid) * (period_grid < upper_lim)
    dip_x, dip_y = period_grid[m], phi[m]

    # Amplitude, Baseline, mean, sigma
    bnds = ((None, None), (0, 2), (None, None), (None, None))

    result = sco.minimize(nll, [-.1, 1., best_period, .1*best_period],
                        args=(dip_x, dip_y), bounds=bnds)
    return result, dip_x, dip_y
