from typing import Dict

import jax.numpy as jnp
import pytest
from jaxtyping import Array

from bayinx import Model, Parameter, define
from bayinx.dists import normal
from bayinx.mhx.vi import MeanField, NormalizingFlow, Standard
from bayinx.mhx.vi.flows import FullAffine, Planar, Radial


class NormalDist(Model):
    x: Parameter[Array] = define(shape = (2,))
    something: int = 1

    def eval(self, data: Dict[str, Array]):
        # Constrain parameters
        self, target = self.constrain_params()

        # Evaluate x ~ Normal(10.0, 1.0)
        target += normal.logprob(self.x(), 10.0, 1.0).sum()

        return target


# Tests ----
@pytest.mark.parametrize("var_draws", [1, 4])
def test_meanfield(benchmark, var_draws):
    # Construct model
    model = NormalDist()

    # Construct meanfield variational
    vari = MeanField(model)

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000, var_draws=var_draws)

    vari = vari.fit(20000, var_draws=var_draws)
    benchmark(benchmark_fit)

    # Assert parameters are roughly correct
    assert all(abs(10.0 - vari.mean) < 0.1) and all(abs(0.0 - vari.log_std) < 0.1)


@pytest.mark.parametrize("var_draws", [1, 4])
def test_affine(benchmark, var_draws):
    # Construct model
    model = NormalDist()

    # Construct normalizing flow variational
    vari = NormalizingFlow(Standard(model), [FullAffine(2)], model)

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000, var_draws=var_draws)

    vari = vari.fit(20000, var_draws=var_draws)
    benchmark(benchmark_fit)

    params = vari.flows[0].transform_params()
    assert (abs(10.0 - vari.flows[0].params["shift"]) < 0.1).all() and (
        abs(jnp.eye(2) - params["scale"]) < 0.1
    ).all()


@pytest.mark.parametrize("var_draws", [1, 4])
def test_flows(benchmark, var_draws):
    # Construct model
    model = NormalDist()

    # Construct normalizing flow variational
    vari = NormalizingFlow(
        Standard(model), [FullAffine(2), Planar(2), Radial(2)], model
    )

    # Optimize variational distribution
    def benchmark_fit():
        vari.fit(10000, var_draws=var_draws)

    vari = vari.fit(20000, var_draws=var_draws)
    benchmark(benchmark_fit)

    mean = vari.sample(1000).mean(0)
    var = vari.sample(1000).var(0)
    assert (abs(10.0 - mean) < 0.1).all() and (abs(var - 1.0) < 0.1).all()
