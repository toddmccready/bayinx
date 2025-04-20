from typing import Dict

import jax.numpy as jnp
from jaxtyping import Array

from bayinx import Model, Parameter
from bayinx.dists import normal
from bayinx.mhx.vi import Standard


class NormalDist(Model):
    x: Parameter[Array]

    def __init__(self):
        self.x = Parameter(jnp.array([0.0, 0.0]))

    def eval(self, data: Dict[str, Array]):
        # Constrain parameters
        self, target = self.constrain_params()

        # Evaluate x ~ Normal(10.0, 1.0)
        target += jnp.sum(normal.logprob(self.x(), jnp.array(10.0), jnp.array(1.0)))

        return target


def test_predictive():
    # Construct model
    model = NormalDist()

    # Construct meanfield variational
    vari = Standard(model)

    # Posterior predictive function
    def extract_first_x(model: NormalDist) -> Array:
        return model.x()[0]

    # Sample from posterior
    first_x_samples: Array = vari.posterior_predictive(extract_first_x, 1000)

    assert abs(first_x_samples.mean()) < 0.1 and abs(first_x_samples.var() - 1.0) < 0.1
