"""
Tools for stitching light curves together.
"""

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from exoplanet.gp import terms, GP
import exoplanet as xo


class StitchModel(object):
    """
    Code for stitching light curves together with a GP.

    Args:
        t (array): The time array in days.
        y (array): The flux array.
        yerr (array): The array of flux uncertainties.

        gap_times (array): The times of the gaps between light curves.
        steps (array): The flux differences between the 1st light curve, and
            all following light curves.

    """

    def __init__(self, t, y, yerr, steps, gap_times):

        self.t = t
        self.y = y
        self.yerr = yerr
        self.steps = steps
        self.gap_times = gap_times

    def model_offsets(self):
        """
        Define the GP offset model.

        """

        nsteps = len(self.steps)
        with pm.Model() as model:

            # Parameters
            logsigma = pm.Normal("logsigma", mu=0.0, sd=15.0)
            logrho = pm.Normal("logrho", mu=0.0, sd=5.0)

            # Define step variables
            step1 = pm.Normal("step1", mu=self.steps[0], sd=2.0)
            steps = step1
            if nsteps > 1:
                step2 = pm.Normal("step2", mu=self.steps[1], sd=2.0)
                steps = [step1, step2]
            if nsteps > 2:
                step3 = pm.Normal("step3", mu=self.steps[2], sd=2.0)
                steps = [step1, step2, step3]

            # The step model
            mu = step_model(self.t, self.gap_times, steps)

            # The likelihood function assuming known Gaussian uncertainty
            pm.Normal("obs", mu=mu, sd=self.yerr, observed=self.y)

            # Set up the kernel an GP
            kernel = terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)
            gp = GP(kernel, self.t, self.yerr ** 2)

            # Add a custom "potential" (log probability function) with the GP
            # likelihood
            pm.Potential("gp", gp.log_likelihood(self.y))

        self.gp = gp
        self.model = model
        return model

    def find_optimum(self):
        """
        Optimize to find the MAP solution.

        Returns:
            map_soln (dict): a dictionary containing the optimized parameters.
        """

        with self.model:
            map_soln = xo.optimize(start=self.model.test_point)

        self.map_soln = map_soln
        return map_soln

    def evaluate_model(self, test_t):
        """
        Evaluate the best-fit GP offset model.

        Args:
            test_t (array): The ordinate values to plot the prediction at.

        Returns:
            mu (array): The best-fit mean-function.
            var (array): The variance of the model
        """

        with self.model:
            mu, var = xo.eval_in_model(
                self.gp.predict(test_t, return_var=True), self.map_soln)
        return mu, var


def step_model(t, gap_times, steps):
    """
    The step model. A model that starts at zero for the 1st light curve, then
    models the offsets between subsequent light curves and the 1st.

    Args:
        t (array): The concatenated time array of all light curves.
        gap_times (array): The times of the gaps between light curves.
        steps (array): The flux differences between the 1st light curve, and
            all following light curves.

    Returns:
        mu (array): A model that is zero for the 1st light curve, with step
            functions for the offsets of all subsequent light curves.

    """
    mu = np.zeros(len(t))
    for i in range(len(gap_times)-1):
        mu += (t > gap_times[i]) * (t < gap_times[i+1]) * steps[i]
    mu += (t > gap_times[-1]) * steps[-1]
    return mu
