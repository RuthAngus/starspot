import numpy as np
import matplotlib.pyplot as plt
import starspot.stitch as sps


def test_stitching():

    # Simulate data
    N = 100
    t = np.linspace(0, 20, N)
    y = np.zeros_like(t) + np.random.randn(N)*.1
    yerr = np.ones_like(y)*.1
    y_orig = y*1

    # 3 gaps
    gap_times, steps = [5, 12, 17], [1, 2, 1]
    mu = sps.step_model(t, gap_times, steps)
    y += mu

    star = sps.StitchModel(t, y, yerr, gap_times, steps, 2.0)
    star.model_offsets()
    map_soln = star.find_optimum()
    mu_gp, var = star.evaluate_model(t)

    step1, step2, step3 = map_soln[0]["step1"], map_soln[0]["step2"], \
        map_soln[0]["step3"]

    # plt.subplot(3, 1, 1)
    # plt.errorbar(t, y, yerr=yerr, fmt="k.")
    # plt.plot(t, mu_gp)
    # plt.subplot(3, 1, 2)
    # stitched = y - sps.step_model(t, gap_times, [step1, step2, step3])
    # plt.errorbar(t, stitched, yerr=yerr, fmt=".", alpha=.5)
    # plt.errorbar(t, y_orig, yerr=yerr, fmt=".", alpha=.5)
    # plt.subplot(3, 1, 3)
    # plt.errorbar(t, y - mu_gp, yerr=yerr, fmt="k.")
    # plt.savefig("test_stitch")

    assert np.isclose(map_soln[0]["step1"], 1, atol=.1)
    assert np.isclose(map_soln[0]["step2"], 2, atol=.1)
    assert np.isclose(map_soln[0]["step3"], 1, atol=.1)

    # 13 gaps
    # Simulate data
    N = 500
    t = np.linspace(0, 100, N)
    y = np.zeros_like(t) + np.random.randn(N)*.1
    y_orig = y*1
    yerr = np.ones_like(y)*.1

    gap_times = [5, 12, 17, 20, 30, 35, 40, 45, 50, 55, 60, 70, 80]
    steps = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    mu = sps.step_model(t, gap_times, steps)
    y += mu

    star = sps.StitchModel(t, y, yerr, gap_times, steps, 2.0)
    star.model_offsets()
    map_soln = star.find_optimum()
    mu_gp13, var = star.evaluate_model(t)

    staps = [map_soln[0]["step1"], map_soln[0]["step2"], map_soln[0]["step3"],
             map_soln[0]["step4"], map_soln[0]["step5"], map_soln[0]["step6"],
             map_soln[0]["step7"], map_soln[0]["step8"], map_soln[0]["step9"],
             map_soln[0]["step10"], map_soln[0]["step11"],
             map_soln[0]["step12"], map_soln[0]["step13"]]

    # plt.subplot(2, 1, 1)
    # plt.errorbar(t, y, yerr=yerr, fmt="k.")
    # plt.plot(t, mu_gp13)
    # plt.subplot(2, 1, 2)
    # stitched = y - sps.step_model(t, gap_times, staps)
    # plt.errorbar(t, stitched, yerr=yerr, fmt=".", alpha=.5)
    # plt.errorbar(t, y_orig, yerr=yerr, fmt=".", alpha=.5)
    # plt.savefig("test_stitch13")

    assert np.isclose(map_soln[0]["step1"], 1, atol=.1)
    assert np.isclose(map_soln[0]["step2"], 2, atol=.1)
    assert np.isclose(map_soln[0]["step3"], 1, atol=.1)


if __name__ == "__main__":
    test_stitching()
