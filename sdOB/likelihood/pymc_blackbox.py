'''
[PyMC](https://docs.pymc.io/index.html) is a great tool for doing Bayesian inference and parameter estimation. It has a load of [in-built probability distributions](https://docs.pymc.io/api/distributions.html) that you can use to set up priors and likelihood functions for your particular model. You can even create your own [custom distributions](https://docs.pymc.io/prob_dists.html#custom-distributions).

Here are the class used to build external likelihood, one can find the detail usage from the [web](https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html)
'''

import aesara
import aesara.tensor as at
import arviz as az
import numpy as np


class LogLike(at.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, arguments):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike: [function]
            The log-likelihood (or whatever) function we've defined
        arguments: [list]
            a list contains all arguments except (theta, which contains MCMC parameters) of likelihood (probability density) function,
            e.g. arguments = [x_obs, y_obs, sigma]
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.arguments = arguments

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, *self.arguments)

        outputs[0][0] = np.array(logl)  # output the log-likelihood


def gradients(vals, func, releps=1e-3, abseps=None, mineps=1e-9, reltol=1e-3,
              epsscale=0.5):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    vals: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    func:
        A function that takes in an array of values.
    releps: float, array_like, 1e-3
        The initial relative step size for calculating the derivative.
    abseps: float, array_like, None
        The initial absolute step size for calculating the derivative.
        This overrides `releps` if set.
        `releps` is set then that is used.
    mineps: float, 1e-9
        The minimum relative step size at which to stop iterations if no
        convergence is achieved.
    epsscale: float, 0.5
        The factor by which releps if scaled in each iteration.

    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.zeros(len(vals))

    # maximum number of times the gradient can change sign
    flipflopmax = 10.

    # set steps
    if abseps is None:
        if isinstance(releps, float):
            eps = np.abs(vals)*releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
            teps = releps*np.ones(len(vals))
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
            teps = releps
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")
    else:
        if isinstance(abseps, float):
            eps = abseps*np.ones(len(vals))
        elif isinstance(abseps, (list, np.ndarray)):
            if len(abseps) != len(vals):
                raise ValueError("Problem with input absolute step sizes")
            eps = np.array(abseps)
        else:
            raise RuntimeError("Absolute step sizes are not a recognised type!")
        teps = eps

    # for each value in vals calculate the gradient
    count = 0
    for i in range(len(vals)):
        # initial parameter diffs
        leps = eps[i]
        cureps = teps[i]

        flipflop = 0

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5*leps  # change forwards distance to half eps
        bvals[i] -= 0.5*leps  # change backwards distance to half eps
        cdiff = (func(fvals)-func(bvals))/leps

        while 1:
            fvals[i] -= 0.5*leps  # remove old step
            bvals[i] += 0.5*leps

            # change the difference by a factor of two
            cureps *= epsscale
            if cureps < mineps or flipflop > flipflopmax:
                # if no convergence set flat derivative (TODO: check if there is a better thing to do instead)
                warnings.warn("Derivative calculation did not converge: setting flat derivative.")
                grads[count] = 0.
                break
            leps *= epsscale

            # central difference
            fvals[i] += 0.5*leps  # change forwards distance to half eps
            bvals[i] -= 0.5*leps  # change backwards distance to half eps
            cdiffnew = (func(fvals)-func(bvals))/leps

            if cdiffnew == cdiff:
                grads[count] = cdiff
                break

            # check whether previous diff and current diff are the same within reltol
            rat = (cdiff/cdiffnew)
            if np.isfinite(rat) and rat > 0.:
                # gradient has not changed sign
                if np.abs(1.-rat) < reltol:
                    grads[count] = cdiffnew
                    break
                else:
                    cdiff = cdiffnew
                    continue
            else:
                cdiff = cdiffnew
                flipflop += 1
                continue

        count += 1

    return grads



# define a theano Op for our likelihood function
class LogLikeWithGrad(at.Op):

    itypes = [at.dvector]  # expects a vector of parameter values when called
    otypes = [at.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, arguments):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike: [function]
            The log-likelihood (or whatever) function we've defined
        arguments: [list]
            a list contains all arguments except (theta, which contains MCMC parameters) of likelihood (probability density) function,
            e.g. arguments = [x_obs, y_obs, sigma]
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.arguments = arguments

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, *self.arguments)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(at.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [at.dvector]
    otypes = [at.dvector]

    def __init__(self, loglike, wave_obs, flux_obs, fluxerr_obs, wavelength, flux1, flux2, arm='R', slmodel=None):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        flux_obs:
            The "observed" flux that our log-likelihood function takes in
        wavelength:
            The dependent variable (aka 'wavelength') that our model requires (for slam model or interpolate)
        flux1:
            The slam interpolate flux of star1
        flux2:
            The slam interpolate flux of star2
        arm: [str] 'R' or 'B'
        slmodel: slam model
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.flux_obs = flux_obs
        self.wave_obs = wave_obs
        self.wavelength = wavelength
        self.fluxerr_obs = fluxerr_obs
        self.flux1 = flux1
        self.flux2 = flux2
        self.arm = arm
        self.slmodel = slmodel


        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(theta, *self.arguments)

        # calculate gradients
        grads = gradients(theta, lnlike)

        outputs[0][0] = grads


