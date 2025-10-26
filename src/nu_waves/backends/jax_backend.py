import jax.numpy as jnp
import jax.random as jr


class _JaxRandom:
    def __init__(self, seed=0):
        self.key = jr.key(seed)

    def _split(self):
        self.key, subkey = jr.split(self.key)
        return subkey

    def standard_normal(self, shape):
        return jr.normal(self._split(), shape)

    def uniform(self, shape, minval=0.0, maxval=1.0):
        return jr.uniform(self._split(), shape, minval=minval, maxval=maxval)


class JaxBackend:
    def __init__(self, device=None):
        self.xp = jnp
        import jax
        self._device = device or jax.device("cpu")
        self.random = _JaxRandom(seed=0)

        print("[JAX] Using device:", self._device)
        pass

    """Subset of Array-API interface mapped to torch."""
    def __getattr__(self, name):
        # delegate to torch if it exists
        if hasattr(self.xp, name):
            return getattr(self.xp, name)
        raise AttributeError(f"[JAX]: jax.numpy has no attribute '{name}'")

    @property
    def device(self):
        return self._device

    def random(self, *shape):
        import jax.random as jr
        self.key, subkey = jr.split(self.key)
        return jr.normal(subkey, shape)
