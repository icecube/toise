
Likelihood analysis
*******************

Nearly every analysis in IceCube is based on a likelihood test. Given data
:math:`\vec{x}`, one calculates the likelihood of that data given a hypothesis
:math:`H`. If the data points are independent, the global likelihood is simply
the product of the likelihoods of all the data points,

.. math::
	
	L(\vec{x} | H) = \prod_i L(x_i | H) . 

Typically :math:`H` has some free parameters that can be fit to the data, e.g.
in the diffuse analysis, the normalization and spectral index of the
quasi-diffuse astrophysical neutrino flux. The hypothesis :math:`\hat{H}` that
maximizes :math:`L` is the best fit hypothesis. It is usually more convenient
to work with the logarithm of the likelihood,

.. math::
	
	\ln L(\vec{x} | H) = \sum_i \ln L(x_i | H) .

Since the logarithm is a monotonic mapping, the :math:`\hat{H}` that maximizes
the log-likelihood also maximizes the likelihood itself. 

Given a best-fit hypothesis :math:`\hat{H}`, it is pertinent to ask whether the
data favor it significantly over a null hypothesis :math:`H_0`. This question
is best formulated as a likelihood ratio test. The test statistic

.. math::
	
	t \equiv 2\left[ \ln L(\vec{x} | \hat{H}) - \ln L(\vec{x} | H_0 ) \right]

is a measure of the significance of the distinction between :math:`\hat{H}` and
:math:`H_0`. This is especially useful in situations where :math:`H_0` and
:math:`\hat{H}` are *nested models*, i.e. :math:`H_0` is simply :math:`\hat{H}`
with :math:`N` of its parameters fixed to certain values, e.g. the number of
"signal" events or the "signal" flux fixed to 0. In such cases, :math:`t` is
expected to follow a :math:`\chi^2` distribution with :math:`N` degrees of
freedom. The p-value of the test is the probability that :math:`t` would take
on a value greater than the one observed if :math:`H_0` were true. If the
p-value is below some threshold, e.g. :math:`P(\chi_1^2 > M^2)`, we say that
:math:`H_0` is rejected at :math:`M \sigma`.

When optimizing analyses without access to real data (either because the data
are kept blind to avoid bias or because the detector hasn't been built yet), we
are typically interested in two quantities for each kind of analysis:

- *sensitivity*, defined as the median 90% upper limit expected in an ensemble
  of experiments if :math:`H_0` is true, and
- *discovery potential*, defined as the signal level required to reject
  :math:`H_0` at more than :math:`5\sigma` in 50% of the experiments.

In the general case we would derive quantities like this by sampling
realizations of the models :math:`H_0` and :math:`H_1` ("pseudodata"), fitting
each realization to build up the distribution of :math:`t` and find the
appropriate quantile. The median, however, is a special and extremely
convenient case. For some families of likelihood functions there exists an
artificial dataset, known as the [Asimov]_ dataset, with the property that
:math:`\hat{H} = H_1` by construction.

The utility of this artificial dataset is that the contours of the test
statistic :math:`t` are equivalent to the median of contours that one would
obtain from repeated sampling, i.e. instead of fitting thousands of fluctuated
realizations of the underlying hypothesis to obtain the median test statistic,
one simply has to fit the average realization directly. For the case we are
most interested in, the binned Poisson likelihood, the Asimov dataset is the
one where :math:`x = \mu` in every bin. Since every :math:`n`-dimensional
unbinned likelihood is equivalent to a Poisson likelihood with :math:`\mu =
\int_{\rm bin} {\rm pdf}(\vec{y} | H) dy^n` in the limit of infinitessimally
small bins, we will approximate all analyses with a Poisson likelihood.


Model analyses
**************

Point source sensitivity with throughgoing muons
------------------------------------------------

.. plot:: plots/ps_sensitivity_demo.py demo_plot

.. [Asimov] G. Cowan, K. Cranmer, E. Gross, and O. Vitells. Asymptotic formulae for likelihood-based tests of new physics. The European Physical Journal C, 71(2):1554, 2011. ISSN 1434-6044. `doi: 10.1140/epjc/s10052-011-1554-0 <http://dx.doi.org/10.1140/epjc/s10052-011-1554-0>`_.

Gotchas
*******

- Requires SciPy >= 0.15 (so use py2-v2)

Depends on external projects:

- phys-services
- MuonGun
- photospline
- NewNuFlux
- AtmosphericSelfVeto