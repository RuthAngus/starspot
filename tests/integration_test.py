# import numpy as np
# import lightkurve as lk
# import starspot as ss
# import starspot.rotation_tools as rt
# import matplotlib.pyplot as plt

# def test_run():

#     starname = "TIC 10863087"
#     lcf = lk.search_lightcurvefile(starname).download()
#     lc = lcf.PDCSAP_FLUX
#     no_nan_lc = lc.remove_nans()
#     clipped_lc = no_nan_lc.remove_outliers(sigma=3)
#     clipped_lc.scatter(alpha=.5, s=.5);

#     rotate = ss.RotationModel(clipped_lc.time, clipped_lc.flux, clipped_lc.flux_err)
#     rotate.lc_plot()
#     ls_period = rotate.ls_rotation()
#     rotate.ls_plot()
#     tess_cadence = 1./24./30.  # This is a TESS 2 minute cadence star.
#     acf_period = rotate.acf_rotation(tess_cadence)
#     # rotate.acf_plot()

#     period_grid = np.linspace(.1, 2, 1000)
#     pdm_period, period_err = rotate.pdm_rotation(
#         period_grid, pdm_nbins=10)
#     print(pdm_period, period_err)

#     rotate.pdm_plot();

#     # Lomb-Scargle periodogram
#     period_array = 1./rotate.freq
#     power_array = rotate.power

#     # Autocorrelation function
#     ACF_array = rotate.acf
#     lag_array = rotate.lags

#     # Phase-dispersion minimization
#     phi_array = rotate.phis  # The 'dispersion' plotted in the lower panel above.
#     period_grid = period_grid  # We already defined this above.

#     # Get peak positions and heights, in order of highest to lowest peak.
#     peak_positions, peak_heights = rt.get_peak_statistics(1./rotate.freq,
#                                                           rotate.power)
#     print(peak_positions[0])

#     # Get peak positions and heights, in order of highest to lowest peak.
#     acf_peak_positions, acf_peak_heights = rt.get_peak_statistics(
#         rotate.lags, rotate.acf, sort_by="height")
#     print(acf_peak_positions[0])

#     # Get peak positions and heights, in order of lags.
#     acf_peak_positions, acf_peak_heights = rt.get_peak_statistics(
#         rotate.lags, rotate.acf, sort_by="position")
#     print(acf_peak_positions[0])


# # def test_gp():
# #     starname = "TIC 10863087"
# #     lcf = lk.search_lightcurvefile(starname).download()
# #     lc = lcf.PDCSAP_FLUX
# #     no_nan_lc = lc.remove_nans()
# #     clipped_lc = no_nan_lc.remove_outliers(sigma=3)
# #     clipped_lc.scatter(alpha=.5, s=.5);

# #     rotate = ss.RotationModel(clipped_lc.time, clipped_lc.flux, clipped_lc.flux_err)
# #     gp_results = rotate.gp_rotation()
# #     print("GP period = {0:.2f} + {1:.2f} - {2:.2f}".format(
# #         rotate.gp_period, rotate.errp, rotate.errm))
# #     rotate.plot_posterior()

# #     plt.hist(rotate.period_samples);
# #     plt.xlabel("Period [days]")
# #     plt.ylabel("Unnormalized probability")
# #     rotate.plot_prediction()

# if __name__ == "__main__":
#     test_run()
    ## test_gp()
