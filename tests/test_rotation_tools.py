# import numpy as np
# import matplotlib.pyplot as plt

# import starspot.rotation_tools as rt

# def test_transit_mask():
#     N = 1000
#     x = np.linspace(0, 100, N)
#     y = np.random.randn(N)*.1
#     t0, dur, porb = 12, 3, 20

#     mask = ((x - (t0 - .5*dur)) % porb) < dur
#     y[mask] -= 2

#     for i in range(int(100/porb)):
#         plt.axvline(t0 + porb*i)

#     mask = rt.transit_mask(x, t0, dur*24, porb)
#     assert np.isclose(np.mean(y[mask]), 0, atol=.01)

#     # plt.plot(x, y, ".")
#     # plt.plot(x[mask], y[mask], ".")
#     # plt.savefig("test")

# if __name__ == "__main__":
#     test_transit_mask()
