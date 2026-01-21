# Credits to: Pietro Torta

import numpy as np
import random

from scipy.linalg import eigvalsh


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


def compute_absVal_matrix(matrix):
    """
    # NB: it computes |matrix|
    matrix must be symmetric, not necessarily PSD
    """
    if not is_symmetric(matrix):
        print("\nERROR: matrix is not symmetric\n")

    print(f"\nBounds on M: {matrix.min(), matrix.max()}")

    s, u = np.linalg.eigh(matrix)
    s_min, s_max = s.min(), s.max()
    print(f"Bounds on spectrum for M: {s_min, s_max}")

    if s_min < 0:
        print(f"M is NOT a PSD matrix with {np.sum(s < 0)} negative eigenvalues\n")
    print("\n")

    s = np.clip(np.abs(s), 1e-13, 1e13)

    return np.real(np.dot(u * s, u.T))
    # return np.dot(u, np.dot(np.diag(s), vh))


def generate_qubo_matrix(n, dist="norm", scale_term=1, PSD=False, linear_term=True):
    """
    Generate a random QUBO matrix of size n x n.

    Args:
    n (int): The size of the QUBO matrix.

    dist = "norm", "unif", "bern"
    *** Some parameters for dist are set inside the function, namely loc scale and width

    scale_term is a scaling factor in front of the linear term

    Returns:
    numpy.ndarray: A n x n matrix representing a QUBO problem.
    """
    # Check if 'n' is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("'n' must be a positive integer")

    # Generate a random upper triangular matrix

    if dist == "norm":
        loc, scale = 0, 1  # for "norm" distribution 0, 1
        # loc, scale = 5, 2
        Q = np.random.normal(loc=loc, scale=scale, size=(n, n))
        diagonal = (np.random.normal(loc=loc, scale=scale, size=n)) * scale_term
    elif dist == "unif":
        width = 100  # for "unif" distribution
        Q = np.random.uniform(low=-width, high=width, size=(n, n))
        diagonal = (np.random.uniform(low=-width, high=width, size=n)) * scale_term
    elif dist == "bern":
        Q = 2 * np.random.randint(0, 2, (n, n)) - 1
        # Q = 2 * np.random.randint(0, 2, (n, n)) + 1
        diagonal = (2 * np.random.randint(0, 2, n) - 1) * scale_term
    else:
        raise (Exception(f"\nWrong distribution = {dist}\n"))

    # Since QUBO matrices are symmetric
    # Q = (Q + Q.T) / 2
    Q = np.triu(Q, k=1) + np.triu(Q, k=1).T + np.diag(diagonal)
    # print(Q)

    if PSD:
        # enforce PSD by computing |Q|
        Q = compute_absVal_matrix(Q)

    if not linear_term:
        # Zero-out the diagonal if you don't want any linear terms in your QUBO (optional)
        np.fill_diagonal(Q, 0)

    return Q


def generate_Ising(n, dist="norm", scale_term=1, PSD=False, linear_term=True):
    """
    Generates J and h of an Ising spin glass

    Args:
    n (int): The linear size of the J matrix.

    dist = "norm", "unif", "bern"
    *** Some parameters for dist are set inside the function, namely loc scale and width

    scale_term is a scaling factor in front of the linear field term h

    Returns:
    (J, h)
    """

    # the linear term is now interpreted as the h_i local fields (possibly with a scale_term factor)
    # J must have all-zero diagonal elements!
    J = generate_qubo_matrix(n, dist=dist, scale_term=scale_term, PSD=PSD, linear_term=linear_term)
    h = np.copy(np.diag(J))  # shallow copy it!!
    np.fill_diagonal(J, 0)

    return J, h


def gauge_transformation(Q, xi):
    """
    Applies a gauge transf. to the QUBO matrix Q
    xi: it is a vector of 0 (flip) or 1 (not flip). Different notation from the paper: (xi paper) = 2*(xi code) -1 --> 1-(xi paper) = 2*[1-(xi code)]
    From QUBO to QUBO' (all bit flipped)
    QUBO: E = - x.T Q x
    QUBO': E = - x'.T Q' x' - K
    gamma = 1 and delta = 0 in the paper
    gamma' = 1 and delta' = - K in the paper
    """

    # non-diagonal (interaction terms)
    Q_prime = np.einsum("i, ij, j -> ij", (2 * xi - 1), Q, (2 * xi - 1))

    # diagonal (field term)
    Q_prime_diagonal = np.diag(Q) + np.einsum("i, il, l -> i", 2 * (2 * xi - 1), Q, (1 - xi))
    np.fill_diagonal(Q_prime, Q_prime_diagonal)

    # constant
    K = np.einsum("i, ij, j -> ", (1 - xi), Q, (1 - xi))

    return Q_prime, K


def gauge_transformation_configurations(configs, xi):
    """
    configs.shape = (M, n)
    xi: it is a vector of 0 (flip) or 1 (not flip).
    """

    # Create a copy of configs to avoid modifying the original
    configs_prime = np.copy(configs)

    boolean_mask = xi == 0  # where to flip

    configs_prime[:, boolean_mask] = 1 - configs[:, boolean_mask]

    return configs_prime


def gauge_transformation_single_configuration(config, xi):
    """
    As above, but for a single configuration
    """

    # Create a copy of configs to avoid modifying the original
    config_prime = np.copy(config)

    boolean_mask = xi == 0  # where to flip

    config_prime[boolean_mask] = 1 - config[boolean_mask]

    return config_prime


def make_sparse(Q, p):
    """
    Given an input symmetric matrix Q, set each matrix element to zero with prob p
    Keep it symmetric!
    """

    n = Q.shape[0]
    Q_sparse = np.copy(Q)
    for i in range(n):
        for j in range(i):
            if random.random() <= p:
                Q_sparse[i, j] = Q_sparse[j, i] = 0
            else:
                pass
    return Q_sparse


def get_lower_bounds(Q):
    """
    Get a rough lower bound and a statistical lower bound for the QUBO problem Q: E = -x^T Q x
    """
    # (rough: normalization is an issue) lower bound
    if not is_symmetric(Q):
        print("\nERROR: Q is not symmetric\n")
    n = Q.shape[0]
    eigenvalues = eigvalsh(Q)
    largest_eigenvalue = eigenvalues[-1]
    lower_bound = -largest_eigenvalue * n  # if (1,1,...,1) is the eigenvector of largest_eigenvalue
    stat_lower_bound = (
        lower_bound / 4
    )  # if (x1,x2,...,xN) is the eigenvector of largest_eigenvalue with each component being 0 or 1 with equal probability
    return lower_bound, stat_lower_bound


def statistical_analysis(J, h):
    """
    Statistical Analysis for a typical QUBO problem defined as Q = np.copy(J) + np.diag(h)

    Compute S^+, S^- etc
    Also compute the warm start "approximate" (neglecting the diagonal terms)
    """
    if not is_symmetric(J):
        print("\nERROR: J is not symmetric\n")

    N = J.shape[0]
    A_plus = []
    A_minus = []
    S_plus = 0  # >=0
    S_minus = 0  # >0

    warm_start_approximate = np.zeros(N, dtype=int)

    for j in range(N):
        sum = np.sum(J[j, :])
        # print(sum)
        check = sum >= 0
        A_plus.append(check)  #  positive or zero!
        A_minus.append(not check)  # strictly negative
        if check:
            S_plus += sum
            warm_start_approximate[j] = 1
        else:
            S_minus += -sum
            warm_start_approximate[j] = 0

    A_plus_plus = np.einsum("i,j -> ij", A_plus, A_plus)
    A_plus_minus = np.einsum("i,j -> ij", A_plus, A_minus)
    A_minus_minus = np.einsum("i,j -> ij", A_minus, A_minus)

    J_bar = S_plus - S_minus
    S_plus_plus = np.sum(J[A_plus_plus])
    S_plus_minus = np.sum(J[A_plus_minus])
    S_minus_minus = np.sum(J[A_minus_minus])
    h_plus = np.sum(h[A_plus])
    h_minus = np.sum(h[A_minus])
    h_bar = h_plus + h_minus

    return (
        warm_start_approximate,
        S_plus,
        S_minus,
        J_bar,
        S_plus_plus,
        S_plus_minus,
        S_minus_minus,
        h_plus,
        h_minus,
        h_bar,
    )


def Ising_to_QUBO(J, h=None):
    """
    ISING: H = - 0.5 * sigma.T J sigma - h.sigma
    QUBO: E = - x.T Q_I x
    H = 2*E + C_I
    With the paper's notation: alpha = 1, beta = 0
    Thus: gamma = 2, delta = C_I

    Explicitly: energy_Ising==energy_QUBO

    energy_Ising = - 0.5 * sigma_Ising.T @ J @ sigma_Ising - np.dot(sigma_Ising, h)
    energy_QUBO = - 2 * x_QUBO.T @ Q_I @ x_QUBO + C_I,
        with C_I = - 0.5 * np.sum(J) + np.sum(h)

    """

    assert is_symmetric(J), "Non symmetric input matrix J: stop"
    check_on_diagonal_J = not any(np.diag(J) != 0)
    assert check_on_diagonal_J, "Some diagonal elements of J are non zero: stop"

    # off-diagonal terms
    Q_I = np.copy(J)  # you MUST use np.copy to shallow copy it or you modify J!!
    if h is None:
        h = np.zeros(J.shape[0])

    # diagonal terms
    n = Q_I.shape[0]
    #
    for j in range(n):
        Q_I[j, j] = h[j] - 1 * np.sum(J[:, j])  # no factor 2

    C_I = -0.5 * np.sum(J) + np.sum(h)

    return Q_I, C_I


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

    # off-diagonal terms
    J_Q = np.copy(Q)  # you MUST use np.copy to shallow copy it or you modify Q
    np.fill_diagonal(J_Q, 0)

    # diagonal terms
    n = J_Q.shape[0]
    h_Q = np.zeros(n)
    #
    for j in range(n):
        h_Q[j] = np.sum(Q[:, j])

    C_Q = 0.5 * (np.sum(Q) + np.sum(np.diag(Q)))

    return J_Q/4, h_Q/2, C_Q/2


def get_J_h(Q):
    """
    Extracts the J and h parameters from the Q matrix for a QUBO problem.

    Parameters:
    Q (numpy.ndarray): The Q matrix of the QUBO problem.

    Returns:
    J (numpy.ndarray): The interaction terms of the QUBO problem.
    h (numpy.ndarray): The linear terms of the QUBO problem.
    """
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                J[i, j] = Q[i, j]
            else:
                h[i] = Q[i, i]

    return J, h


"""
Generate samples that automatically assign one and only one starting time
for each operation of JSSP in QUBO formulation
"""


def generate_vector(size):
    # Initialize a zero vector
    vector = np.zeros(size, dtype=int)
    # Randomly choose an index to set to 1
    index = np.random.randint(0, size)
    vector[index] = 1
    return vector


def generate_full_vector(size, N):
    # Initialize an empty list to store the vectors
    appended_vector = []
    # Append the generated vectors M times
    for _ in range(N):
        appended_vector.extend(generate_vector(size))
    return np.array(appended_vector)


def generate_dataset_vectors(size, N, M):
    """
    M is the number of configurations, each with length = size * N
    """
    dataset = np.zeros((M, int(size * N)))
    for m in range(M):
        dataset[m, :] = generate_full_vector(size, N)
    return dataset
