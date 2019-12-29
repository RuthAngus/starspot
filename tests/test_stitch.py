import numpy as np
import matplotlib.pyplot as plt
import starspot.stitch as sps


def test_stitching():

    # Simulate data
    N = 100
    t = np.linspace(0, 20, N)
    y = np.zeros_like(t) + np.random.randn(N)*.1
    yerr = np.ones_like(y)*.1

    # 3 gaps
    gap_times, steps = [5, 12, 17], [1, 2, 1]
    mu = sps.step_model(t, gap_times, steps)
    y += mu

    star = sps.StitchModel(t, y, yerr, gap_times, steps, 2.0)
    star.model_offsets()
    map_soln = star.find_optimum()
    mu, var = star.evaluate_model(t)

    # plt.subplot(2, 1, 1)
    # plt.errorbar(t, y, yerr=yerr, fmt="k.")
    # plt.plot(t, mu)
    # plt.subplot(2, 1, 2)
    # plt.errorbar(t, y - mu, yerr=yerr, fmt="k.")
    # plt.savefig("test_stitch")

    print(map_soln["step1"], map_soln["step2"], map_soln["step3"])
    assert np.isclose(map_soln["step1"], 1, atol=.1)
    assert np.isclose(map_soln["step2"], 2, atol=.1)
    assert np.isclose(map_soln["step3"], 1, atol=.1)

    # 13 gaps
    # Simulate data
    N = 500
    t = np.linspace(0, 100, N)
    y = np.zeros_like(t) + np.random.randn(N)*.1
    yerr = np.ones_like(y)*.1

    gap_times = [5, 12, 17, 20, 30, 35, 40, 45, 50, 55, 60, 70, 80]
    steps = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    mu = sps.step_model(t, gap_times, steps)
    y += mu

    star = sps.StitchModel(t, y, yerr, gap_times, steps, 2.0)
    star.model_offsets()
    map_soln = star.find_optimum()
    mu, var = star.evaluate_model(t)

    # plt.subplot(2, 1, 1)
    # plt.errorbar(t, y, yerr=yerr, fmt="k.")
    # plt.plot(t, mu)
    # plt.subplot(2, 1, 2)
    # plt.errorbar(t, y - mu, yerr=yerr, fmt="k.")
    # plt.savefig("test_stitch13")

    print(map_soln["step1"], map_soln["step2"], map_soln["step3"])
    assert np.isclose(map_soln["step1"], 1, atol=.1)
    assert np.isclose(map_soln["step2"], 2, atol=.1)
    assert np.isclose(map_soln["step3"], 1, atol=.1)


if __name__ == "__main__":
    test_stitching()
