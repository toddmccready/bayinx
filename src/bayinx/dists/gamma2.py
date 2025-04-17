import jax.lax as lax
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, Float


def prob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], nu: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a (mean-precision parameterized) Gamma distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `mu`:     The positive mean.
    - `nu`:     The positive inverse dispersion.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `nu`.
    """

    return lax.exp(logprob(x, mu, nu))


def logprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], nu: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log-transformed probability density function (log PDF) for a (mean-precision parameterized) Gamma distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log PDF.
    - `mu`:     The positive mean/location.
    - `nu`:     The positive inverse dispersion.

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `nu`.
    """

    return - gammaln(nu) + nu * (lax.log(nu) - lax.log(mu)) + (nu - 1.0) * lax.log(x) - (x * nu / mu) # pyright: ignore
