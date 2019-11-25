import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rotation_tools import transit_mask, load_kepler_data

def test_transit_mask():

    # Load dataframe of kois
    full = pd.read_csv("../../KOI_rotation/kane_cks_tdmra_dr2.csv")
    df = full.iloc[0]
    print(df.kepid, df.koi_time0, df.koi_duration, df.koi_period)

    # Load and plot the first lc
    starname = str(int(df.kepid)).zfill(9)
    lcpath = "/Users/rangus/.kplr/data/lightcurves/{}".format(starname)
    x, y, yerr = load_kepler_data(lcpath)

    tmask = transit_mask(x, df.koi_time0, df.koi_duration,
                                df.koi_period)
    ms = 2
    plt.plot(x, y, ".", color="C0", ms=ms, zorder=0)
    plt.plot(x[~tmask], y[~tmask], ".", color="C1", ms=ms, zorder=0)
    plt.xlim(x[0], x[0] + 100)
    plt.savefig("test")

if __name__ == "__main__":
    test_transit_mask()
