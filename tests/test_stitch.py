import numpy as np
import matplotlib.pyplot as plt
import starspot.stitch as sps


def test_stitching():

    # Simulate data
    N = 100
    t = np.linspace(0, 20, N)
    y = np.zeros_like(t) + np.random.randn(N)*.1
    yerr = np.ones_like(y)*.1

    gap_times, steps = [5, 12, 17], [1, 2, 1]
    mu = sps.step_model(t, gap_times, steps)
    y += mu

    plt.subplot(2, 1, 1)
    plt.errorbar(t, y, yerr=yerr, fmt="k.")

    star = sps.StitchModel(t, y, yerr, steps, gap_times)
    star.model_offsets()
    map_soln = star.find_optimum()
    mu, var = star.evaluate_model(t)
    plt.plot(t, mu)

    plt.subplot(2, 1, 2)
    plt.errorbar(t, y - mu, yerr=yerr, fmt="k.")
    plt.savefig("test_stitch")

    print(map_soln["step1"], map_soln["step2"], map_soln["step3"])
    assert np.isclose(map_soln["step1"], 1, atol=.1)
    assert np.isclose(map_soln["step2"], 2, atol=.1)
    assert np.isclose(map_soln["step3"], 1, atol=.1)


if __name__ == "__main__":
    test_stitching()
