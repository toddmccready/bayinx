import jax.lax as lax
from jaxtyping import Array, ArrayLike, Float

__PI = 3.141592653589793


def prob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The probability density function (PDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the PDF.
    - `mu`:     The mean/location.
    - `sigma`:  The positive standard deviation.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return lax.exp(-0.5 * lax.square((x - mu) / sigma)) / (  # pyright: ignore
        sigma * lax.sqrt(2.0 * __PI)
    )


def logprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The log of the probability density function (log PDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the log PDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The non-negative standard deviation parameter(s).

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return -lax.log(sigma * lax.sqrt(2.0 * __PI)) - 0.5 * lax.square(
        (x - mu) / sigma  # pyright: ignore
    )


def uprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
) -> Float[Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a Normal distribution.

    # Parameters
    - `x`:      Value(s) at which to evaluate the uPDF.
    - `mu`:     The mean/location parameter(s).
    - `sigma`:  The positive standard deviation parameter(s).

    # Returns
    The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """

    return lax.exp(-0.5 * lax.square((x - mu) / sigma)) / sigma  # pyright: ignore


def ulogprob(
    x: Float[ArrayLike, "..."], mu: Float[ArrayLike, "..."], sigma: Float[ArrayLike, "..."]
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
