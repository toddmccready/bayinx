import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, Float, UInt

__PI = 3.141592653589793

def __binom(x, y):
    """
    Helper function for the Binomial coefficient.
    """
    return jnp.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))


def prob(
    x: UInt[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    phi: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The probability mass function (PMF) for a (mean-inverse overdispersion parameterized) Negatvie Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the PMF.
    - `mu`: The mean.
    - `phi`: The inverse overdispersion.

    # Returns
    The PMF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `phi`.
    """
    # Cast to Array
    x, mu, phi = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(phi)

    return jnp.exp(logprob(x,mu,phi))


def logprob(
    x: UInt[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    phi: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The log-transformed probability mass function (PMF) for a (mean-inverse overdispersion parameterized) Negatvie Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the log PMF.
    - `mu`: The mean.
    - `phi`: The inverse overdispersion.

    # Returns
    The log PMF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `phi`.
    """
    # Cast to Array
    x, mu, phi = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(phi)

    # Evaluate log PMF
    evals: Array = jnp.where(
        x < 0,
        -jnp.inf,
        (
            gammaln(x + phi) - gammaln(x + 1) - gammaln(phi)
            + x * (jnp.log(mu) - jnp.log(mu + phi))
            + phi * (jnp.log(phi) - jnp.log(mu + phi))
        )
    )

    return evals


def cdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    pass


def logcdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    pass


def ccdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    pass


def logccdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    pass
