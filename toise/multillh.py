# Adapted from [multillh](http://code.icecube.wisc.edu/svn/sandbox/nwhitehorn/multillh)

import logging
import numpy as np
import scipy.optimize


class NuisanceParam:
    """
    NuisanceParam is a simple class implementing a free parameter in the
    fit, optionally constrained with either flat or Gaussian priors. This
    class adds only a knob and prior: it does not contribute in any way
    to the Poisson mean in any bin.
    """

    def __init__(self, prior=1, gauserror=None, min=None, max=None):
        """
        Seeds the minimizer with the value of prior, optionally
        constrained by a Gaussian prior around this value (gauserror)
        and/or by half- or fully-closed box constraints (min/max)
        """
        self.seed = prior
        self.uncertainty = gauserror
        self.min = min
        self.max = max

    def prior(self, value, **kwargs):
        if self.uncertainty == None:
            return 0
        return -((value - self.seed) ** 2) / (2.0 * self.uncertainty**2)


class Combination(object):
    def __init__(self, components):
        """
        :param components: a dict of str: (component, livetime)
        """
        self._components = components
        for label, (component, livetime) in self._components.items():
            if hasattr(component, "min"):
                self.min = component.min
            if hasattr(component, "max"):
                self.max = component.max
            if hasattr(component, "seed"):
                self.seed = component.seed
        if not hasattr(self, "seed"):
            self.seed = 1.0

    def prior(self, value, **kwargs):
        v = list(self._components.values())[0][0]
        if hasattr(v, "prior"):
            return v.prior(value, **kwargs)
        else:
            return 0.0

    def expectations(self, *args, **kwargs):
        exes = dict()
        for label, (component, livetime) in self._components.items():
            if hasattr(component.expectations, "__call__"):
                subex = component.expectations(*args, **kwargs)
            else:
                subex = component.expectations
            exes[label] = livetime * subex
        return exes

    @property
    def bin_edges(self):
        edges = dict()
        for label, (component, livetime) in self._components.items():
            edges[label] = component.bin_edges

        return edges

    def _get_energy_range(self):
        """Return the true energy range where all the components have support"""

        def energy_range(i):
            return (
                component[0].energy_range[i] for component in self._components.values()
            )

        return max(energy_range(0)), min(energy_range(1))

    def apply(self, func):
        """Return a copy with the function applied to each component"""
        return Combination(
            {
                k: (func(component), livetime)
                for k, (component, livetime) in self._components.items()
            }
        )

    def differential_chunks(self, *args, **kwargs):
        # create chunk generators and sort by energy range
        generators = sorted(
            [
                {
                    "label": label,
                    "livetime": livetime,
                    "emin": max(
                        component.energy_range[0], kwargs.get("emin", -np.inf)
                    ),
                    "emax": min(
                        component.energy_range[1], kwargs.get("emax", np.inf)
                    ),
                    "chunks": component.differential_chunks(
                        *args, exclusive=True, **kwargs
                    ),
                }
                for label, (component, livetime) in self._components.items()
            ],
            key=lambda item: (item["emin"], item["emax"]),
        )
        # merged chunks
        all_done = False
        while not all_done:
            components = dict()
            eranges = []
            ecenters = []
            e_center = None
            for item in generators:
                # since the generators are sorted by emin, the first is guaranteed to have
                # the smallest (i.e. next) available e_center
                if e_center is None:
                    try:
                        e_center, component = next(item["chunks"])
                    except StopIteration:
                        continue
                # add chunks from remaining generators whose energy ranges overlap the current e_center
                elif e_center > item["emin"] and e_center < item["emax"]:
                    e_center, component = next(item["chunks"])
                else:
                    continue
                if e_center is not None:
                    components[item["label"]] = (component, item["livetime"])
                    eranges.append(component.energy_range)
                    ecenters.append(e_center)
                    logging.getLogger().debug(
                        "label: %s, enu: %.2g" % (item["label"], e_center)
                    )
            if not components:
                all_done = True
            else:
                combo = Combination(components)
                combo.energy_range = eranges[np.argmax(ecenters)]
                combo.energy_center = max(ecenters)
                yield max(ecenters), combo


class LLHEval(object):
    """
    Object containing a Poisson likelihood fit
    """

    def __init__(self, data, unbinned=False):
        """
        Initialize fit with data. This should be a dictionary from
        keys -- of any type -- shared with the fit components to
        np arrays of arbitrary dimensionality containing histograms.

        If the keyword argument unbinned is set to True, data should
        instead be a dictionary mapping keys to a python list of np
        arrays, each describing the probability momemt function for one
        event, binned the same way as the simulated distributions.
        """
        self.components = dict()
        self.data = data
        self.unbinned = unbinned

    def add_component(self, name, component):
        self.components[name] = component

    def bounds(self):
        """
        Returns dictionary of tuples of the bounds on each parameter
        (None if unbounded)
        """
        retval = dict()
        for c in self.components:
            min = 0
            max = None
            if hasattr(self.components[c], "min"):
                min = self.components[c].min
            if hasattr(self.components[c], "max"):
                max = self.components[c].max
            retval[c] = (min, max)
        return retval

    def expectations(self, **kwargs):
        """
        Returns dictionary of np arrays in the same format as the
        input data given the parameter values specified in keyword
        arguments
        """
        lamb = dict()
        for param in kwargs:
            c = self.components[param]
            if not hasattr(c, "expectations"):
                continue
            if callable(c.expectations):
                expec = c.expectations(**kwargs)
            else:
                expec = c.expectations
            for prop in expec:
                llh_bit = kwargs[param] * expec[prop]
                if prop in lamb:
                    lamb[prop] += llh_bit
                else:
                    lamb[prop] = llh_bit
        return lamb

    def llh(self, **kwargs):
        """
        Evaluates log-likelihood for parameter values specified in
        keyword arguments
        """
        lamb = self.expectations(**kwargs)
        llh = 0
        for param in kwargs:
            if hasattr(self.components[param], "prior"):
                llh += self.components[param].prior(kwargs[param], **kwargs)

        for prop in lamb:
            with np.errstate(divide="ignore"):
                log_lambda = np.log(lamb[prop])
                log_data = np.log(self.data[prop])
            log_lambda[np.isinf(log_lambda)] = 0
            log_data[np.isinf(log_data)] = 0
            if self.unbinned:
                norm = np.sum(lamb[prop])
                for event in self.data[prop]:
                    llh += np.sum(event * lamb[prop]) / norm
                llh -= norm
            else:
                llh += np.sum(self.data[prop] * (log_lambda - log_data)) - np.sum(
                    lamb[prop] - self.data[prop]
                )

        return llh

    def llh_contributions(self, **kwargs):
        """
        Evaluates log-likelihood contributions for parameter values specified in
        keyword arguments. Priors are not included.
        """
        lamb = self.expectations(**kwargs)
        llh = dict()
        for prop in lamb:
            log_lambda = np.log(lamb[prop])
            log_lambda[np.isinf(log_lambda)] = 0
            llh[prop] = self.data[prop] * log_lambda - lamb[prop]
        return llh

    def sat_llh(self, **kwargs):
        """
        Evaluates LLH for a saturated Poisson model. Note
        that since this does not involve the parameters of the regular
        model, the resulting values do not include priors
        """
        if self.unbinned:
            raise ValueError("Saturated LLH does not work in unbinned mode")

        llh = 0
        dof = 0
        for prop in self.data:
            log_lambda = np.log(self.data[prop])
            log_lambda[np.isinf(log_lambda)] = 0
            llh += np.sum(self.data[prop] * log_lambda - self.data[prop])
            dof += np.sum(self.data[prop] != 0)
        return (llh, dof + 1)

    def fit(self, minimizer_params=dict(), **fixedparams):
        """
        Return dictionary of best-fit values for all parameters using
        scipy's L-BFGS-B fitter. Keyword arguments are optional and
        are interpreted as assigning fixed values to the named
        parameters in the fit. All parameters not passed as keyword
        arguments will be free to float.

        If a parameter is fixed to an iterable value, it will be treated
        as a discrete parameter and minimized on a grid.
        """
        from collections.abc import Iterable
        import itertools

        freeparams = list(self.components.keys())
        discrete_params = []
        bounds = []
        seeds = []
        for param in fixedparams:
            if isinstance(fixedparams[param], Iterable) and not isinstance(
                fixedparams[param], str
            ):
                discrete_params.append(param)
            if param in freeparams:
                freeparams.remove(param)
        for p in freeparams:
            bounds.append(self.bounds()[p])
            if hasattr(self.components[p], "seed"):
                seeds.append(self.components[p].seed)
            else:
                seeds.append(0)

        def minllh(x):
            pdict = dict(zip(freeparams, x))
            pdict.update(fixedparams)
            return -self.llh(**pdict)

        if len(discrete_params) == 0:
            if freeparams:
                bestfit = scipy.optimize.minimize(
                    minllh,
                    seeds,
                    bounds=bounds,
                    **minimizer_params,
                )
            else:
                bestfit = {"x": dict(fixedparams)}
            fixedparams.update(dict(zip(freeparams, bestfit["x"])))
            return fixedparams
        else:
            bestllh = np.inf
            bestparams = dict(fixedparams)
            for points in itertools.product(
                *tuple(fixedparams[k] for k in discrete_params)
            ):
                for k, p in zip(discrete_params, points):
                    fixedparams[k] = p
                if freeparams:
                    bestfit = scipy.optimize.minimize(
                        minllh,
                        seeds,
                        bounds=bounds,
                        **minimizer_params,
                    )
                else:
                    bestfit = {"x": dict(freeparams), "fun": -self.llh(**fixedparams)}
                if bestfit["fun"] < bestllh:
                    bestparams = dict(fixedparams)
                    bestparams.update(dict(zip(freeparams, bestfit["x"])))
                    bestllh = bestfit["fun"]
            return bestparams

    def profile1d(self, param, values, minimizer_params=dict(), **fixedparams):
        """
        Return a named np array with best-fit nuisance values and
        likelihood values for the values of param passed. Additional
        keywords are interpreted as fixed values of nuisance parameters.
        """
        llhpoints = []
        for val in values:
            params = {param: val}
            params.update(fixedparams)
            if len(params) - len(self.components) == 0:
                fit = params
            else:
                fit = self.fit(minimizer_params=minimizer_params, **params)
            mlh = self.llh(**fit)
            llhpoints.append(tuple(list(fit.values()) + [mlh]))
        dkeys = list(fit.keys()) + ["LLH"]
        dtypes = [float] * (len(list(fit.keys())) + 1)
        for i in range(len(dtypes)):
            if isinstance(llhpoints[-1][i], str):
                dtypes[i] = "|S32"
        return np.array(llhpoints, dtype=list(zip(dkeys, dtypes)))

    def profile2d(self, param1, values1, param2, values2, **fixedparams):
        """
        Returns a 2D scan (see profile1d) of the likelihood space at
        all combinations of values passed.
        """
        llhpoints = []
        for val1 in values1:
            for val2 in values2:
                params = {param1: val1, param2: val2}
                params.update(fixedparams)
                if len(params) - len(self.components) == 0:
                    fit = params
                else:
                    fit = self.fit(**params)
                mlh = self.llh(**fit)
                llhpoints.append(list(fit.values()) + [mlh])
        return np.asarray(llhpoints).view(
            dtype=list(
                zip(list(fit.keys()) + ["LLH"], [float] * (len(list(fit.keys())) + 1))
            )
        )


def _pseudo_llh(components, poisson, **nominal):
    allh = LLHEval(None)
    allh.components = components
    for k in allh.components.keys():
        if not k in nominal:
            try:
                nominal[k] = getattr(components[k], "seed")
            except AttributeError:
                nominal[k] = 1
    expectations = allh.expectations(**nominal)
    if poisson:
        pseudodata = dict()
        for tag in expectations:
            pseudodata[tag] = np.random.poisson(expectations[tag])
    else:
        pseudodata = expectations
    allh.data = pseudodata
    return allh


def pseudodata_llh(components, **hypothesis):
    """
    Create a likelihood on a realization of the hypothesis
    """

    return _pseudo_llh(components, True, **hypothesis)


def asimov_llh(components, **nominal):
    """
    Create a likelihood on the average over all realizations of the hypothesis
    *nominal*, also known as the Asimov_ dataset. Likelihood ratios obtained
    this way are asymptotically equivalent to the median likelihood ratio over
    repeated realizations of the hypothesis.

    _Asimov: http://dx.doi.org/10.1140/epjc/s10052-011-1554-0
    """

    return _pseudo_llh(components, False, **nominal)


def get_expectations(llh, **nominal):
    exes = dict()
    for k, comp in llh.components.items():
        if k in nominal:
            continue
        if hasattr(comp, "seed"):
            nominal[k] = comp.seed
        else:
            nominal[k] = 1
    for k, comp in llh.components.items():
        if hasattr(comp, "expectations"):
            if callable(comp.expectations):
                ex = comp.expectations(**nominal)
            else:
                ex = comp.expectations
            exes[k] = {klass: nominal[k] * values for klass, values in ex.items()}
    return exes
