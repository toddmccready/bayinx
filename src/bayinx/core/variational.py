from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Key


class Variational(eqx.Module):
    """
    A variational approximation.
    """

    @abstractmethod
    def sample(self, n: int, key: Key) -> Array:
        """
        Sample from the variational distribution `n` times.
        """

        pass

    @abstractmethod
    def eval(self, samples: Array) -> Array:
        """
        Evaluate the variational density at `samples`.
        """

        pass
