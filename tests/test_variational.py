from typing import Callable, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from bayinx import Model
from bayinx.dists import normal
from bayinx.machinery import MeanField
from bayinx.machinery.flows.affine import ElementwiseAffine
from bayinx.machinery.normalizing_flow import NormalizingFlow


# Tests ----
def test_meanfield(benchmark):
    # Construct model
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array(0.0)}
            self.constraints = {}

        @eqx.filter_jit
        def eval(self, data: dict):
            # Get constrained parameters
            params = self.constrain()

            # Evaluate mu ~ N(10,1)
            return jnp.sum(
                normal.logprob(x=params["mu"], mu=jnp.array(10.0), sigma=jnp.array(1.0))
            )

    model = NormalDist()

    # Construct meanfield variational
    vari = MeanField(model)

    # Optimize variational distribution
    benchmark(vari.fit, 10000)
    vari = vari.fit(10000)

    # Assert parameters are roughly correct
    assert abs(10.0 - vari.var_params["mean"]) < 0.01 and abs(0.0 - vari.var_params['log_std']) < 0.01


def test_normalizingflow(benchmark):
    # Construct model
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array(0.0)}
            self.constraints = {}

        @eqx.filter_jit
        def eval(self, data: dict):
            # Get constrained parameters
            params = self.constrain()

            # Evaluate mu ~ N(10,1)
            return jnp.sum(
                normal.logprob(x=params["mu"], mu=jnp.array(10.0), sigma=jnp.array(1.0))
            )

    model = NormalDist()

    # Construct normalizing flow variational
    vari = NormalizingFlow(MeanField(model), [ElementwiseAffine(1)], model)

    # Optimize variational distribution
    benchmark(vari.fit, 10000)
    vari = vari.fit(10000)

    # Assert parameters are roughly correct
    assert abs(10.0 - vari.flows[0].params["shift"]) < 0.01 and abs(0.0 - vari.flows[0].params["scale"]) < 0.01
