import jax.numpy as jnp


def running_means(chain):
    """
    Calculate the running mean of a Markov chain.

    Parameters:
    chain (numpy.ndarray): A 2D array representing the Markov chain,
                           where each row is a sample and each column is a degree of freedom.

    Returns:
    numpy.ndarray: A 2D array of the running mean for each degree of freedom.
    """
    cumulative_sum = jnp.cumsum(chain, axis=0)
    sample_counts = jnp.arange(1, chain.shape[0] + 1).reshape(-1, 1)
    return cumulative_sum / sample_counts


def autocorrelation(chain, max_lag=None):
    """
    Calculate the autocorrelation function for each degree of freedom in a Markov chain.

    Parameters:
    chain (numpy.ndarray): A 2D array representing the Markov chain,
                           where each row is a sample and each column is a degree of freedom.
    max_lag (int): The maximum lag to compute the autocorrelation for. Defaults to 10% of the chain length.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to a lag and each column to a degree of freedom.
    """
    if max_lag is None:
        max_lag = chain.shape[0] // 10  # default to 10% of the chain length

    mean = jnp.mean(chain, axis=0)
    var = jnp.var(chain, axis=0)

    acors = jnp.zeros((max_lag + 1, chain.shape[1]))

    for lag in range(max_lag + 1):
        if lag == 0:
            acors = acors.at[lag].set(1)
        else:
            shifted_chain = chain[:-lag] - mean
            lagged_chain = chain[lag:] - mean
            acors = acors.at[lag].set(
                jnp.sum(shifted_chain * lagged_chain, axis=0)
                / (var * (chain.shape[0] - lag))
            )

    return acors


def autocorrelation_time(chain, max_lag=None):
    if max_lag is None:
        max_lag = chain.shape[0] // 10  # default to 10% of the chain length

    acor_time = jnp.zeros(chain.shape[1])

    for i in range(chain.shape[1]):
        acor_sum = 0.0
        for lag in range(1, max_lag + 1):
            autocorr = jnp.corrcoef(chain[:-lag, i], chain[lag:, i])[0, 1]
            if autocorr < 0:
                break
            acor_sum += autocorr

        acor_time = acor_time.at[i].set(1 + 2 * acor_sum)

    return acor_time
