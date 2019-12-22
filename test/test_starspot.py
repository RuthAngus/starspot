import numpy as np
from starspot import phase_dispersion_minimization as pdm
import matplotlib.pyplot as plt
import starspot as ss


def test_big_plot():
    # Generate some data
    time = np.linspace(0, 100, 1000)
    p = 10
    w = 2*np.pi/p
    flux = np.sin(w*time) + np.random.randn(len(time))*1e-2
    flux_err = np.ones_like(flux)*1e-2

    rotate = ss.RotationModel(time, flux, flux_err)
    ls_period = rotate.ls_rotation()
    acf_period = rotate.acf_rotation(interval=0.02043365)
    pdm_period, period_err = rotate.pdm_rotation(rotate.lags, pdm_nbins=10)
    fig1 = rotate.big_plot()
    fig1.savefig("big_plot_test")


if __name__ == "__main__":
    test_big_plot()
