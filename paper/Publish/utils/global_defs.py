import jax.numpy as jnp
from typing import Union, Callable

PRNGKey = jnp.ndarray
CNumber = Union[float, complex]
VariationalParameters = jnp.ndarray
SpinConfiguration = jnp.ndarray
MCSample = jnp.ndarray
MCState = jnp.ndarray
VariationalLogPsi = Callable[[VariationalParameters, SpinConfiguration], CNumber]

LocalOperator = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
ParametricLocalOperator = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], CNumber]