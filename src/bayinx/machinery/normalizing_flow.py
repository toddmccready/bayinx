# Imports ----
import jax
import equinox as eqx
import optax as opx
import jax.lax as lax, jax.numpy as jnp, jax.random as jr
from jax.flatten_util import ravel_pytree
from bayinx import Model
from bayinx.dists import normal
from functools import partial

# Typing ----
from typing import Dict, Callable, Tuple, Any
from jaxtyping import Array, Key, Scalar
from optax import OptState



class NormalizingFlow(eqx.Module):
    dmorphs: list