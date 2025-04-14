from typing import Callable, Dict

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from bayinx import Model
from bayinx.dists import normal
from bayinx.mhx.vi import MeanField, NormalizingFlow, Standard
from bayinx.mhx.vi.flows import FullAffine, Planar, Radial


# Tests ----
@pytest.mark.parametrize("var_draws", [1, 10, 100])
def test_meanfield(benchmark, var_draws):
    # Construct model definition
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array([0.0, 0.0])}
            self.constraints = {}

        @eqx.filter_jit
        def eval(self, data: dict):
            # Get constrained parameters
            params = self.constrain_pars()

            # Evaluate mu ~ N(10,1)
            return jnp.sum(
                normal.logprob(x=params["mu"], mu=jnp.array(10.0), sigma=jnp.array(1.0))
            )

    # Construct model
    model = NormalDist()

    # Construct meanfield variational
    vari = MeanField(model)

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000, var_draws=var_draws)

    benchmark(benchmark_fit)
    vari = vari.fit(20000)

    # Assert parameters are roughly correct
    assert all(abs(10.0 - vari.var_params["mean"]) < 0.1) and all(
        abs(0.0 - vari.var_params["log_std"]) < 0.1
    )


@pytest.mark.parametrize("var_draws", [1, 10, 100])
def test_affine(benchmark, var_draws):
    # Construct model definition
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array([0.0, 0.0])}
            self.constraints = {}

        @eqx.filter_jit
        def eval(self, data: dict):
            # Get constrained parameters
            params = self.constrain_pars()

            # Evaluate mu ~ N(10,1)
            return jnp.sum(
                normal.logprob(x=params["mu"], mu=jnp.array(10.0), sigma=jnp.array(1.0))
            )

    # Construct model
    model = NormalDist()

    # Construct normalizing flow variational
    vari = NormalizingFlow(Standard(model), [FullAffine(2)], model)

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000, var_draws=var_draws)

    benchmark(benchmark_fit)
    vari = vari.fit(20000)

    params = vari.flows[0].constrain_pars()
    assert (abs(10.0 - vari.flows[0].params["shift"]) < 0.1).all() and (
        abs(jnp.eye(2) - params["scale"]) < 0.1
    ).all()


@pytest.mark.parametrize("var_draws", [1, 10, 100])
def test_flows(benchmark, var_draws):
    # Construct model definition
    class NormalDist(Model):
        params: Dict[str, Array]
        constraints: Dict[str, Callable[[Array], Array]]

        def __init__(self):
            self.params = {"mu": jnp.array([0.0, 0.0])}
            self.constraints = {}

        @eqx.filter_jit
        def eval(self, data: dict):
            # Get constrained parameters
            params = self.constrain_pars()

            # Evaluate mu ~ N(10,1)
            return jnp.sum(
                normal.logprob(x=params["mu"], mu=jnp.array(10.0), sigma=jnp.array(1.0))
            )

    # Construct model
    model = NormalDist()

    # Construct normalizing flow variational
    vari = NormalizingFlow(
        Standard(model), [FullAffine(2), Planar(2), Radial(2)], model
    )

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000, var_draws=var_draws)

    benchmark(benchmark_fit)
    vari = vari.fit(20000)

    mean = vari.sample(1000).mean(0)
    var = vari.sample(1000).var(0)
    assert (abs(10.0 - mean) < 0.1).all() and (abs(var - 1.0) < 0.1).all()
