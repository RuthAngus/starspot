import numpy as np
import matplotlib.pyplot as plt
from starspot.rotation_tools import filter_sigma_clip, sigma_clip


def test_sigma_clip():
    np.random.seed(42)
    N, Nout = 1000, 20
    t0 = np.linspace(0, 100, N)
    p = 10
    w = 2*np.pi/p
    y0 = np.sin(w*t0) + np.random.randn(N)*.1
    inds = np.random.choice(np.arange(len(y0)), Nout)
    y0[inds] += np.random.randn(Nout)*10.

    # Initial removal of extreme outliers.
    m = sigma_clip(y0, nsigma=7)
    t, y = t0[m], y0[m]

    # Sigma clip
    smooth, mask = filter_sigma_clip(t, y, polyorder=2)
    resids = y - smooth

    # Plot results
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    plt.plot(t0, y0, ".", label="Original")
    plt.plot(t, y, ".", label="initial clip")
    plt.plot(t, smooth, label="smoothed")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, resids, ".", label="Whole lc")
    plt.plot(t[~mask], resids[~mask], ".", label="Detected outliers")
    plt.legend()
    plt.savefig("test.png")


if __name__ == "__main__":
    test_sigma_clip()
