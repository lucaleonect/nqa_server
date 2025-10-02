import numpy as np

def are_close(num1, num2, tolerance=1e-10):
    """
    Check if two numbers differ by less than a specified tolerance.

    Args:
    num1 (float): The first number.
    num2 (float): The second number.
    tolerance (float): The tolerance level for difference (default is 1e-6).

    Returns:
    bool: True if the difference is less than the tolerance, False otherwise.
    """
    return abs(num1 - num2) < tolerance


def is_symmetric(matrix):
    is_symmetric = np.allclose(matrix, matrix.T, atol=1e-08)
    return is_symmetric

def QUBO_to_Ising(Q):
    """
    QUBO: E = - x.T Q x \\ sure about the - sign?
    ISING: H = - 0.5 * sigma.T J_Q sigma - h.sigma
    E = 1/2 H - 1/2 C_Q
    With the paper's notation: gamma = 1, delta = 0
    Thus: alpha = 1/2, beta = - 1/2 C_Q

    Explicitly: energy_QUBO==energy_Ising

    energy_QUBO = - x_QUBO.T @ Q @ x_QUBO,
    energy_Ising = - 0.25 * sigma_Ising.T @ J_Q @ sigma_Ising - 0.5*np.dot(sigma_Ising, h) -0.5*C_Q
        with C_Q = np.sum(Q_triangle_upper) + np-diag(Q) = 0.5 * (np.sum(Q) + np.diag(Q))

    """

    assert is_symmetric(Q), "Non symmetric input matrix Q: stop"
    J_Q = np.copy(Q)  # you MUST use np.copy to shallow copy it or you modify Q
    np.fill_diagonal(J_Q, 0)

    n = J_Q.shape[0]
    h_Q = np.zeros(n)
    for j in range(n):
        h_Q[j] = np.sum(Q[:, j])

    C_Q = 0.5 * (np.sum(Q) + np.sum(np.diag(Q)))

    return J_Q/4, h_Q/2, C_Q/2
