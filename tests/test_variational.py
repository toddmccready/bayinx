from typing import Callable, Dict

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from bayinx import Model
from bayinx.dists import normal
from bayinx.machinery.variational import MeanField, NormalizingFlow, Standard
from bayinx.machinery.variational.flows import Affine


# Tests ----
def test_meanfield(benchmark):
    # Construct model
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array([0.0, 0.0])}
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
    def benchmark_fit():
        vari.fit(10000)

    benchmark(benchmark_fit)
    vari = vari.fit(10000)

    # Assert parameters are roughly correct
    assert all(abs(10.0 - vari.var_params["mean"]) < 0.1) and all(
        abs(0.0 - vari.var_params["log_std"]) < 0.1
    )


def test_normalizingflow(benchmark):
    # Construct model
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array([0.0, 0.0])}
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
    vari = NormalizingFlow(Standard(model), [Affine(2)], model)

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000)

    benchmark(benchmark_fit)
    vari = vari.fit(10000)

    params = vari.flows[0].constrain()
    assert (
        all(abs(10.0 - vari.flows[0].params["shift"]) < 0.1)
        and (abs(jnp.eye(2) - params["scale"]) < 0.1).all()
    )
