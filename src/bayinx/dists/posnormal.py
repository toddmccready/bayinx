import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from bayinx.dists import normal


def prob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = jnp.array(0.0) <= x

    # Evaluate PDF
    evals = jnp.where(
        non_negative,
        normal.prob(x, mu, sigma) / normal.cdf(mu/sigma, 0.0, 1.0),
        jnp.array(0.0))

    return evals


def logprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the probability density function (log PDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the log PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = jnp.array(0.0) <= x

    # Evaluate log PDF
    evals = jnp.where(
        non_negative,
        normal.logprob(x, mu, sigma) - normal.logcdf(mu/sigma, 0.0, 1.0),
        -jnp.inf)

    return evals


def uprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the uPDF.
    - `mu`: The mean/location parameter(s).
    - `sigma`: The positive standard deviation parameter(s).

    # Returns
    The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = jnp.array(0.0) <= x

    # Evaluate PDF
    evals = jnp.where(
        non_negative,
        normal.prob(x, mu, sigma),
        jnp.array(0.0))

    return evals


def ulogprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the unnormalized probability density function (log uPDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the log uPDF.
    - `mu`: The mean/location parameter(s).
    - `sigma`: The non-negative standard deviation parameter(s).

    # Returns
    The log uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = jnp.array(0.0) <= x

    # Evaluate log PDF
    evals = jnp.where(
        non_negative,
        normal.logprob(x, mu, sigma),
        -jnp.inf)

    return evals


def cdf(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The cumulative density function (CDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the log uPDF.
    - `mu`: The mean/location parameter(s).
    - `sigma`: The non-negative standard deviation parameter(s).

    # Returns
    The CDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.

    # Notes
    Not numerically stable for small `x`.
    """
    # Cast to Array
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = jnp.array(0.0) <= x

    # Compute intermediates
    A: Array = normal.cdf(x, mu, sigma)
    B: Array = normal.cdf(- mu / sigma, 0.0, 1.0)
    C: Array = normal.cdf(mu / sigma, 0.0, 1.0)

    # Evaluate CDF
    evals = jnp.where(
        non_negative,
        (A - B) / C,
        jnp.array(0.0))

    return evals

# TODO: make numerically stable
def logcdf(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log-transformed cumulative density function (log CDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the log uPDF.
    - `mu`: The mean/location parameter(s).
    - `sigma`: The non-negative standard deviation parameter(s).

    # Returns
    The log CDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.

    # Notes
    Not numerically stable for small `x`.
    """
    # Cast to Array
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = jnp.array(0.0) <= x

    A: Array = normal.logcdf(x, mu, sigma)
    B: Array = normal.logcdf(- mu/sigma, 0.0, 1.0)
    C: Array = normal.logcdf(mu/sigma, 0.0, 1.0)

    # Evaluate log CDF
    evals = jnp.where(
        non_negative,
        A + jnp.log1p(-jnp.exp(B - A)) - C,
        -jnp.inf)

    return evals

def ccdf(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The complementary cumulative density function (cCDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the log uPDF.
    - `mu`: The mean/location parameter(s).
    - `sigma`: The non-negative standard deviation parameter(s).

    # Returns
    The cCDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.

    # Notes
    Not numerically stable for small `x`.
    """
    # Cast to arrays
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = 0.0 <= x

    # Compute intermediates
    A: Array = normal.cdf(-x, -mu, sigma)
    B: Array = normal.cdf(mu/sigma, 0.0, 1.0)

    # Evaluate cCDF
    evals = jnp.where(non_negative, A / B, jnp.array(1.0))

    return evals


def logccdf(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log-transformed complementary cumulative density function (log cCDF) for a positive Normal distribution.

    # Parameters
    - `x`: Value(s) at which to evaluate the log uPDF.
    - `mu`: The mean/location parameter(s).
    - `sigma`: The non-negative standard deviation parameter(s).

    # Returns
    The log cCDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.

    # Notes
    Not numerically stable for small `x`.
    """
    # Cast to arrays
    x, mu, sigma = jnp.array(x), jnp.array(mu), jnp.array(sigma)

    # Construct boolean mask for non-negative elements
    non_negative: Array = 0.0 <= x

    # Compute intermediates
    A: Array = normal.logcdf(-x, -mu, sigma)
    B: Array = normal.logcdf(mu/sigma, 0.0, 1.0)

    # Evaluate log cCDF
    evals = jnp.where(non_negative, A - B, jnp.array(0.0))

    return evals
