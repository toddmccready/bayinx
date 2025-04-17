import jax.lax as lax
from jax.scipy.special import gamma, gammaln
from jaxtyping import Array, ArrayLike, Float, Real


def prob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], nu: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a (mean-precision parameterized) Gamma distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `mu`:     The mean.
    - `nu`:     The positive inverse dispersion.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `nu`.
    """


    return 1 / gamma(nu) * (nu / mu)**nu * x**(nu-1.0) * lax.exp(- (x * nu / mu)) # pyright: ignore


def logprob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], nu: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log-transformed probability density function (log PDF) for a (mean-precision parameterized) Gamma distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log PDF.
    - `mu`:     The mean/location.
    - `nu`:     The positive inverse dispersion.

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `nu`.
    """

    return - gammaln(nu) + nu * (lax.log(nu) - lax.log(mu)) + (nu - 1.0) * lax.log(x) - (x * nu / mu) # pyright: ignore


def uprob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], sigma: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the uPDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return lax.exp(-0.5 * lax.square((x - mu) / sigma)) / sigma  # pyright: ignore


def ulogprob(
    x: Real[ArrayLike, "..."], mu: Real[ArrayLike, "..."], sigma: Real[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the unnormalized probability density function (log uPDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log uPDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The log uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return -lax.log(sigma) - 0.5 * lax.square((x - mu) / sigma)  # pyright: ignore
