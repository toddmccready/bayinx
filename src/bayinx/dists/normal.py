import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Array, ArrayLike, Float

__PI = 3.141592653589793


def prob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return lax.exp(-0.5 * lax.square((x - mu) / sigma)) / (sigma * lax.sqrt(2.0 * __PI))


def logprob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The log of the probability density function (log PDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the log PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return -lax.log(sigma * lax.sqrt(2.0 * __PI)) - 0.5 * lax.square((x - mu) / sigma)


def uprob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return lax.exp(-0.5 * lax.square((x - mu) / sigma)) / sigma


def ulogprob(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    """
    The log of the unnormalized probability density function (log uPDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The log uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return -lax.log(sigma) - 0.5 * lax.square((x - mu) / sigma)


def cdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jss.ndtr((x - mu) / sigma)


def logcdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jss.log_ndtr((x - mu) / sigma)


def ccdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jss.ndtr((mu - x) / sigma)


def logccdf(
    x: Float[ArrayLike, "..."],
    mu: Float[ArrayLike, "..."],
    sigma: Float[ArrayLike, "..."],
) -> Float[Array, "..."]:
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jss.log_ndtr((mu - x) / sigma)
