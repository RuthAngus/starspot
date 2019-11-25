starspot
====================================

.. image:: https://github.com/RuthAngus/starspot/blob/master/docs/logo.pdf
   :width: 600

Check out the `documentation <https://starspot.readthedocs.io/en/latest/>`_!

*starspot* is a tool for measuring stellar rotation periods using
Lomb-Scargle (LS) periodograms, autocorrelation functions (ACFs), phase
dispersion minimization (PDM) and Gaussian processes (GPs).
It uses the `astropy <http://www.astropy.org/>`_ implementation of
`Lomb-Scargle periodograms
<http://docs.astropy.org/en/stable/stats/lombscargle.html>`_, and the
`exoplanet <https://exoplanet.dfm.io/en/stable/>`_ implementation of
fast `celerite <https://celerite.readthedocs.io/en/latest/?badge=latest>`_
Gaussian processes.

*starspot* is compatible with any light curve with time, flux and flux
uncertainty measurements, including Kepler, K2 and TESS light curves.
If your light curve is has evenly-spaced (or close to evenly-spaced)
observations, all three of these methods: LS periodograms, ACFs and GPs will
be applicable.
For unevenly spaced light curves like those from the Gaia, or ground-based
observatories, LS periodograms and GPs are preferable to ACFs.

Example usage
-------------
::

    import numpy as np
    import starspot as ss

    # Generate some data
    time = np.linspace(0, 100, 10000)
    period = 10
    w = 2*np.pi/period
    flux = np.sin(w*time) + np.random.randn(len(time))*1e-2 + \
        np.random.randn(len(time))*.01
    flux_err = np.ones_like(flux)*.01

    rotate = ss.RotationModel(time, flux, flux_err)

    # Calculate the Lomb Scargle periodogram period (highest peak in the periodogram).
    lomb_scargle_period = rotate.ls_rotation()

    # Calculate the autocorrelation function (ACF) period (highest peak in the ACF).
    # This is for evenly sampled data only -- time between observations is 'interval'.
    acf_period = rotate.acf_rotation(interval=np.diff(time)[0])

    # Calculate the phase dispersion minimization period (period of lowest dispersion).
    period_grid = np.linspace(5, 20, 1000)
    pdm_period = rotate.pdm_rotation(period_grid)

    print(lomb_scargle_period, acf_period, pdm_period)
    >> 9.99892010582963 10.011001100110011 10.0

    # Calculate a Gaussian process rotation period
    gp_period = rotate.GP_rotation()


License & attribution
---------------------

Copyright 2018, Ruth Angus.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
