import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import gammaincc
from jaxtyping import Array, ArrayLike, Float

from bayinx.dists import gamma2


def prob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    nu: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The mixed probability mass/density function (PMF/PDF) for a (mean-inverse dispersion parameterized) Gamma distribution.

    # Parameters
    - `x`:  Value(s) at which to evaluate the PMF/PDF.
    - `mu`: The positive mean.
    - `nu`: The positive inverse dispersion.

    # Returns
    The PMF/PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `nu`.
    """
    evals: Array = jnp.zeros_like(x * 1.0) # ensure float dtype

    # Construct boolean masks
    uncensored: Array = jnp.array(jnp.logical_and(0.0 < x, x < censor)) # pyright: ignore
    censored: Array = jnp.array(x == censor) # pyright: ignore

    # Evaluate mixed probability (?) function
    evals = jnp.where(uncensored, gamma2.prob(x, mu, nu), evals)
    evals = jnp.where(censored, gammaincc(nu, x * nu / mu), evals) # pyright: ignore

    return evals


def logprob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    nu: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log-transformed mixed probability mass/density function (log PMF/PDF) for a (mean-inverse dispersion parameterized) Gamma distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log PMF/PDF.
    - `mu`:     The positive mean/location.
    - `nu`:     The positive inverse dispersion.

    # Returns
    The log PMF/PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `nu`.
    """
    evals: Array = jnp.full_like(x * 1.0, -jnp.inf) # ensure float dtype

    # Construct boolean masks
    uncensored: Array = jnp.array(jnp.logical_and(0.0 < x, x < censor)) # pyright: ignore
    censored: Array = jnp.array(x == censor) # pyright: ignore

    evals = jnp.where(uncensored, gamma2.logprob(x, mu, nu), evals)
    evals = jnp.where(censored, lax.log(gammaincc(nu, x * nu / mu)), evals) # pyright: ignore

    return evals
