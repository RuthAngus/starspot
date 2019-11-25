import numpy as np
from starspot import phase_dispersion_minimization as pdm
import matplotlib.pyplot as plt


def test_sj2():
    np.random.seed(42)
    N = 10000
    t = np.linspace(0, .1, N)
    x = np.random.randn(N)
    sj2 = pdm.sj2(x, 0, N)
    assert np.isclose(sj2, 1, atol=.01)


def test_s2():
    np.random.seed(42)
    N = 10000
    M = 10
    nj = np.ones(M) * N
    sj2 = np.zeros(M)
    for j in range(M):
        t = np.linspace(0, .1, N)
        x = np.random.randn(N)
        sj2[j] = pdm.sj2(x, 0, nj[j])
    s2 = pdm.s2(nj, sj2, M)
    assert np.isclose(s2, 1, atol=.01)


def test_phase():

    # Generate some data
    t = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    x = np.sin(w*t) + np.random.randn(len(t))*1e-2

    phase = pdm.calc_phase(10, t)

    # plt.plot(phase, x, ".")
    # plt.savefig("phase_test")
    # plt.close()


def test_phase_bins():
    """
    Make sure that phased light curves are binned correctly.
    """

    # Generate some data
    t = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    x = np.sin(w*t) + np.random.randn(len(t))*1e-2

    nbins = 10

    # Try a period of 2.5
    phase = pdm.calc_phase(2.5, t)
    x_means, phase_bins, Ns, sj2s, x_binned, phase_binned = \
        pdm.phase_bins(nbins, phase, x)
    mid_phase_bins = np.diff(phase_bins) * .5 + phase_bins[:-1]
    s225 = pdm.s2(Ns, sj2s, nbins)

    # Try a period of 5
    phase = pdm.calc_phase(5, t)
    x_means, phase_bins, Ns, sj2s, x_binned, phase_binned = \
        pdm.phase_bins(nbins, phase, x)
    mid_phase_bins = np.diff(phase_bins) * .5 + phase_bins[:-1]
    s25 = pdm.s2(Ns, sj2s, nbins)

    # Try a period of 10
    phase = pdm.calc_phase(10, t)
    x_means, phase_bins, Ns, sj2s, x_binned, phase_binned = \
        pdm.phase_bins(nbins, phase, x)
    mid_phase_bins = np.diff(phase_bins) * .5 + phase_bins[:-1]
    s210 = pdm.s2(Ns, sj2s, nbins)

    # Plot each bin
    for j in range(nbins):
        plt.plot(phase_binned[j], x_binned[j], ".", alpha=.1, zorder=3)

    # Make sure that a period of 10 has the smallest s2 value.
    assert s210 < s25
    assert s210 < s225

    # Calculate the total variance and phi statistic and test that too.
    total_variance = pdm.sj2(x, np.mean(x), len(x))
    phi10 = s210/total_variance
    phi5 = s25/total_variance
    phi25 = s225/total_variance

    assert phi10 < phi5
    assert phi10 < phi25

    assert pdm.phi(10, 10, t, x) == s210/total_variance
    assert pdm.phi(10, 5, t, x) == s25/total_variance
    assert pdm.phi(10, 2.5, t, x) == s225/total_variance

    # plt.plot(phase, x, ".", zorder=0)
    # plt.errorbar(mid_phase_bins, x_means, fmt=".", yerr=sj2s, zorder=1)
    # plt.savefig("phase_test2")
    # plt.close()

    assert np.isclose(max(x_means), 1, atol=.02)


def test_phi():

    # Generate some data
    t = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    x = np.sin(w*t) + np.random.randn(len(t))*1e-2

    # Generate some data
    # t = np.linspace(0, 100, 1000)
    # p = 10
    # w = 2*np.pi/p
    # x1 = np.sin(w*t)
    # x2 = .4*np.sin(w*t + np.pi/2)
    # x3 = .3*np.sin(w*t + np.pi/3)
    # x = x1 #+ x2 + x3
    # x += np.random.randn(len(x)) * .1

    # plt.plot(t, x1)
    # plt.plot(t, x2)
    # plt.plot(t, x3)
    x += np.random.randn(len(x))*.1
    # plt.plot(t, x)
    # plt.savefig("test")

    # Calculate the Phi statistic over a range of periods
    nperiods = 200
    nbins = 10
    periods = np.linspace(1, 20, nperiods)
    phis = np.zeros(nperiods)
    for i, p in enumerate(periods):
        phis[i] = pdm.phi(nbins, p, t, x)

    # Find period with the lowest Phi
    ind = np.argmin(phis)
    pplot = periods[ind]
    # pplot = 10

    # Get variances for that period
    phase = pdm.calc_phase(pplot, t)
    x_means, phase_bs, Ns, sj2s, xb, pb = pdm.phase_bins(nbins, phase, x)
    mid_phase_bins = np.diff(phase_bs)*.5 + phase_bs[:-1]

    # Calculate the phase at that period (for plotting)
    phase = pdm.calc_phase(pplot, t)

    # Make the plot
    # fig = plt.figure(figsize=(16, 9))
    # ax1 = fig.add_subplot(311)
    # ax1.plot(t, x, ".")
    # ax1.set_xlabel("Time")
    # ax1.set_ylabel("Flux")

    # ax2 = fig.add_subplot(312)
    # ax2.plot(phase, x, ".")
    # ax2.errorbar(mid_phase_bins, x_means, yerr=sj2s, fmt=".")
    # ax2.set_xlabel("Phase")
    # ax2.set_ylabel("Flux")

    # ax3 = fig.add_subplot(313)
    # ax3.plot(periods, phis)  # *pplot)
    # ax3.set_xlabel("Period [days]")
    # ax3.set_ylabel("Dispersion")
    # ax3.axvline(periods[ind], color="C1")

    # fig.savefig("phi_test")

    assert np.isclose(periods[ind], 10, atol=.1)


if __name__ == "__main__":
    test_sj2()
    test_s2()

    # Generate some data
    t = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    x = np.sin(w*t) + np.random.randn(len(t))*1e-2

    test_phase()
    test_phase_bins()
    test_phi()
