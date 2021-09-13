import numpy as np
from scipy.stats import poisson

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


class PoissonUnivariate(ScipyModel):
    """Wrapper around scipy.stats.gamma.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.SEMI_BOUNDED
    MODEL_CLASS = poisson

    def _fit_constant(self, X):
        self._params = {
            'loc': np.unique(X)[0],
            'mu': 0.0,
        }

    def _fit(self, X):
        self._params = {
            'loc': np.min(X),
            'mu': np.mean(X)-np.min(X),
        }

    def _is_constant(self):
        return self._params['mu'] == 0

    def _extract_constant(self):
        return self._params['loc']
