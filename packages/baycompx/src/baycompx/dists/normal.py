# MARK: Imports ----
import jax.lax as _lax

## Typing
import jaxtyping as _t


# MARK: Constants
_PI = 3.141592653589793


# MARK: Functions ----
def prob(x: _t.Real[_t.ArrayLike, "..."],
         mu: _t.Real[_t.ArrayLike, "..."],
         sigma: _t.Real[_t.ArrayLike, "..."]) -> _t.Real[_t.Array, "..."]:
    """
    The probability density function (PDF) for a Normal distribution.

    Parameters
    ----------
    - `x` (Real):       Value(s) at which to evaluate the PDF.
    - `mu` (Real):      The mean/location parameter(s).
    - `sigma` (Real):   The non-negative standard deviation parameter(s).

    Returns
    -------
    - (Real):   The PDF evaluated at `x`. The output will have the broadcasted 
                shapes of `x`, `mu`, and `sigma`.
    """

    
    return _lax.exp( -0.5 * _lax.square( (x - mu) / sigma ) ) / ( sigma * _lax.sqrt(2.0 * _PI) )


def logprob(x: _t.Real[_t.ArrayLike, "..."],
            mu: _t.Real[_t.ArrayLike, "..."],
            sigma: _t.Real[_t.ArrayLike, "..."]) -> _t.Real[_t.Array, "..."]:
    """
    The log of the probability density function (log PDF) for a Normal distribution.
    
    Parameters
    ----------
    - `x` (Real): Value(s) at which to evaluate the log PDF.
    - `mu` (Real): The mean/location parameter(s).
    - `sigma` (Real): The non-negative standard deviation parameter(s).
        
    Returns
    -------
    - (Real): The log of the PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    
    return - _lax.log(sigma * _lax.sqrt(2.0 * _PI)) - 0.5 * _lax.square( (x - mu) / sigma )



def uprob(x: _t.Real[_t.ArrayLike, "..."],
          mu: _t.Real[_t.ArrayLike, "..."],
          sigma: _t.Real[_t.ArrayLike, "..."]) -> _t.Real[_t.Array, "..."]:
    """
    The unnormalized probability density function (uPDF) for a Normal distribution.
        
    Parameters
    ----------
    - `x` (Real): Value(s) at which to evaluate the uPDF.
    - `mu` (Real): The mean/location parameter(s).
    - `sigma` (Real): The non-negative standard deviation parameter(s).
        
    Returns
    -------
    - (Real): The uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    
    return _lax.exp( -0.5 * _lax.square( (x - mu) / sigma ) ) / sigma


def ulogprob(x: _t.Real[_t.ArrayLike, "..."],
             mu: _t.Real[_t.ArrayLike, "..."],
             sigma: _t.Real[_t.ArrayLike, "..."]) -> _t.Real[_t.Array, "..."]:
    """
    The log of the unnormalized probability density function (log uPDF) for a Normal distribution.
        
    Parameters
    ----------
    - `x` (Real): Value(s) at which to evaluate the log uPDF.
    - `mu` (Real): The mean/location parameter(s).
    - `sigma` (Real): The non-negative standard deviation parameter(s).
        
    Returns
    -------
    - (Real): The log uPDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    
    return - _lax.log(sigma) - 0.5 * _lax.square( (x - mu) / sigma )
