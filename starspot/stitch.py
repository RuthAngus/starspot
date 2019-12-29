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
        stdev (Optional[float]): The standard deviation of the Gaussian prior
            for flux differences between light curves.

    """

    def __init__(self, t, y, yerr, gap_times, steps, stdev):

        self.t = t
        self.y = y
        self.yerr = yerr
        self.steps = steps
        self.gap_times = gap_times
        self.stdev = stdev

    def model_offsets(self):
        """
        Define the GP offset model.

        """

        stdev = self.stdev
        # stdev = 2.0
        nsteps = len(self.steps)
        with pm.Model() as model:

            # Parameters
            logsigma = pm.Normal("logsigma", mu=0.0, sd=15.0)
            logrho = pm.Normal("logrho", mu=0.0, sd=5.0)

            # Define step variables
            step1 = pm.Normal("step1", mu=self.steps[0], sd=stdev)
            steps = step1
            if nsteps > 1:
                step2 = pm.Normal("step2", mu=self.steps[1], sd=stdev)
                steps = [step1, step2]
            if nsteps > 2:
                step3 = pm.Normal("step3", mu=self.steps[2], sd=stdev)
                steps = [step1, step2, step3]
            if nsteps > 3:
                step4 = pm.Normal("step4", mu=self.steps[3], sd=stdev)
                steps = [step1, step2, step3, step4]
            if nsteps > 4:
                step5 = pm.Normal("step5", mu=self.steps[4], sd=stdev)
                steps = [step1, step2, step3, step4, step5]
            if nsteps > 5:
                step6 = pm.Normal("step6", mu=self.steps[5], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6]
            if nsteps > 6:
                step7 = pm.Normal("step7", mu=self.steps[6], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7]
            if nsteps > 7:
                step8 = pm.Normal("step8", mu=self.steps[7], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8]
            if nsteps > 8:
                step9 = pm.Normal("step9", mu=self.steps[8], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8, step9]
            if nsteps > 9:
                step10 = pm.Normal("step10", mu=self.steps[9], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8, step9, step10]
            if nsteps > 10:
                step11 = pm.Normal("step11", mu=self.steps[10], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8, step9, step10, step11]
            if nsteps > 11:
                step12 = pm.Normal("step12", mu=self.steps[11], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8, step9, step10, step11, step12]
            if nsteps > 12:
                step13 = pm.Normal("step13", mu=self.steps[12], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8, step9, step10, step11, step12, step13]
            if nsteps > 13:
                step14 = pm.Normal("step14", mu=self.steps[13], sd=stdev)
                steps = [step1, step2, step3, step4, step5, step6, step7,
                         step8, step9, step10, step11, step12, step13, step14]


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
    if len(gap_times) > 1:
        for i in range(len(gap_times)-1):
            mu += (t >= gap_times[i]) * (t < gap_times[i+1]) * steps[i]
        mu += (t >= gap_times[-1]) * steps[-1]
    else:
        mu = (t >= gap_times) * steps
    return mu
