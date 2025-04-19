import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from bayinx.dists import posnormal


def prob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The mixed probability mass/density function (PMF/PDF) for a censored positive Normal distribution.

    # Parameters
    - `x`:  Value(s) at which to evaluate the PMF/PDF.
    - `mu`: The mean.
    - `sigma`: The positive standard deviation.
    - `censor`: The positive censor value.

    # Returns
    The PMF/PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, `sigma`, and `censor`.
    """
    # Cast to Array
    x, mu, sigma, censor = jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(censor)

    # Construct boolean masks
    uncensored: Array = jnp.logical_and(0.0 < x, x < censor)
    censored: Array = x == censor

    # Evaluate probability mass/density function
    evals = jnp.where(uncensored, posnormal.prob(x, mu, sigma), 0.0)
    evals = jnp.where(censored, posnormal.ccdf(x,mu,sigma), evals)

    return evals


def logprob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log-transformed mixed probability mass/density function (log PMF/PDF) for a censored positive Normal distribution.

    # Parameters
    - `x`: Where to evaluate the log PMF/PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.
    - `censor`: The censor.

    # Returns
    The log PMF/PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, `sigma`, and `censor`.
    """
    # Cast to Array
    x, mu, sigma, censor = jnp.array(x), jnp.array(mu), jnp.array(sigma), jnp.array(censor)

    # Construct boolean masks
    uncensored: Array = jnp.logical_and(jnp.array(0.0) < x, x < censor)
    censored: Array = x == censor

    # Evaluate log probability mass/density function
    evals = jnp.where(uncensored, posnormal.logprob(x, mu, sigma), -jnp.inf)
    evals = jnp.where(censored, posnormal.logccdf(x, mu, sigma), evals)

    return evals
