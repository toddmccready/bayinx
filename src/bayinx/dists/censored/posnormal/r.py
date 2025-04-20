import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Float, Key

from bayinx.dists import posnormal


def prob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The mixed probability mass/density function (PMF/PDF) for a right-censored positive Normal distribution.

    # Parameters
    - `x`:  Value(s) at which to evaluate the PMF/PDF.
    - `mu`: The mean.
    - `sigma`: The positive standard deviation.
    - `censor`: The positive censor value.

    # Returns
    The PMF/PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, `sigma`, and `censor`.
    """
    # Cast to Array
    x, mu, sigma, censor = (
        jnp.asarray(x),
        jnp.asarray(mu),
        jnp.asarray(sigma),
        jnp.asarray(censor),
    )

    # Construct boolean masks
    uncensored: Array = jnp.logical_and(0.0 < x, x < censor)
    censored: Array = x == censor

    # Evaluate probability mass/density function
    evals = jnp.where(uncensored, posnormal.prob(x, mu, sigma), 0.0)
    evals = jnp.where(censored, posnormal.ccdf(x, mu, sigma), evals)

    return evals


def logprob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The log-transformed mixed probability mass/density function (log PMF/PDF) for a right-censored positive Normal distribution.

    # Parameters
    - `x`: Where to evaluate the log PMF/PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.
    - `censor`: The censor.

    # Returns
    The log PMF/PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, `sigma`, and `censor`.
    """
    # Cast to Array
    x, mu, sigma, censor = (
        jnp.asarray(x),
        jnp.asarray(mu),
        jnp.asarray(sigma),
        jnp.asarray(censor),
    )

    # Construct boolean masks for censoring
    uncensored: Array = jnp.logical_and(jnp.asarray(0.0) < x, x < censor)
    censored: Array = x == censor

    # Evaluate log probability mass/density function
    evals = jnp.where(uncensored, posnormal.logprob(x, mu, sigma), -jnp.inf)
    evals = jnp.where(censored, posnormal.logccdf(x, mu, sigma), evals)

    return evals


def sample(
    n: int,
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
    censor: Float[ArrayLike, "..."] = jnp.inf,
    key: Key = jr.PRNGKey(0),
) -> Float[Array, "..."]:
    """
    Sample from a right-censored positive Normal distribution.

    # Parameters
    - `n`: Number of draws to sample per-parameter.
    - `mu`: The mean.
    - `sigma`: The standard deviation.
    - `censor`: The censor.

    # Returns
    Draws from a right-censored positive Normal distribution. The output will have the shape of (n,) + the broadcasted shapes of `mu`, `sigma`, and `censor`.
    """
    # Cast to Array
    mu, sigma, censor = (
        jnp.asarray(mu),
        jnp.asarray(sigma),
        jnp.asarray(censor),
    )

    # Derive shape
    shape = (n,) + jnp.broadcast_shapes(mu.shape, sigma.shape, censor.shape)

    # Draw from positive normal
    draws = jr.truncated_normal(key, 0.0, jnp.inf, shape) * sigma + mu

    # Censor values
    draws = jnp.where(censor <= draws, censor, draws)

    return draws
