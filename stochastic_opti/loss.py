import numpy as np
import pickle
from scipy.special import gammaln, digamma
from scipy.stats import dirichlet


# Unobserved transitions show up as exact zeros in the empirically fitted
# matrix, which would make log(0) = -inf. Treat them as rare rather than
# impossible: additive smoothing gives them a small finite probability so the
# optimizer sees a graded penalty instead of a flat +inf.
PSEUDOCOUNT = 1e-6

# Defensive floor for probabilities that are still zero (or negative from
# round-off) after smoothing, so we never take log(0).
MIN_PROBABILITY = 1e-300


def smooth_probabilities(probabilities, pseudocount=PSEUDOCOUNT):
    """
    Additive (Laplace) smoothing of a probability vector or row-stochastic
    matrix, renormalised so each row still sums to 1.

    Idempotent on already-smoothed probabilities.

    Parameters:
    probabilities (array-like): 1-D vector or 2-D row-stochastic matrix.
    pseudocount (float): mass added to every entry before renormalising.

    Returns:
    np.ndarray: smoothed probabilities, same shape, with no exact zeros.
    """
    # only smooth if there are any zeros, otherwise return the original array

    if np.all(probabilities > 0):
        return np.asarray(probabilities, dtype=float)

    smoothed = np.asarray(probabilities, dtype=float) + pseudocount
    return smoothed / smoothed.sum(axis=-1, keepdims=True)


def negative_log_likelihood(state_sequence, initial_probabilities, transition_matrix):
    """
    Compute the negative log-likelihood of a state sequence given an HMM
    using the forward algorithm:

    P(z1:T) = pi_z1 * PI(A{z_t-1, z_t}))

    Parameters:
    sequence (array-like): The state sequence.
    hmm (HMM): The hidden Markov model.

    Returns:
    float: The negative log-likelihood of the sequence given the HMM.
    """
    state_sequence = np.asarray(state_sequence, dtype=int)
    if state_sequence.size == 0:
        raise ValueError("state_sequence is empty; cannot compute a likelihood")

    initial_probabilities = np.asarray(initial_probabilities, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)

    likelihood = _safe_log(initial_probabilities[state_sequence[0]])
    for previous_state, state in zip(state_sequence[:-1], state_sequence[1:]):
        likelihood += _safe_log(
            transition_matrix[previous_state, state]
        )  # stay in log space

    return -likelihood


def _safe_log(probability):
    """log() with a floor, so an unsmoothed zero costs a lot but stays finite."""
    return np.log(max(float(probability), MIN_PROBABILITY))


def KL_divergence(A1, A2, PI_1: np.ndarray):
    """
    count-normalized (probabilities) KL divergence between two transition matrices. from https://arxiv.org/pdf/2106.05669 and https://arxiv.org/pdf/2312.04863.
    Not symmetric -- symmetric KL divergence if KL is not infinite: (KL(A1||A2) + KL(A2||A1))/2
    """

    A1 = smooth_probabilities(A1)
    A2 = smooth_probabilities(A2)

    likelihood_ratios = np.sum(
        A1 * np.log(A1 / A2), axis=1
    )  # summing rows of KL divergences for each row to include stationary distributions

    kl_rows = PI_1 * likelihood_ratios  # weighting by stationary distribution of A1

    total_kl = np.sum(kl_rows)  # sum over all rows to get total KL divergence

    return total_kl


def symmetric_KL_divergence(A1, A2, PI_1: np.ndarray, PI_2: np.ndarray):
    """
    Symmetric KL divergence (average of KL(A1||A2) and KL(A2||A1))
    """

    A1 = smooth_probabilities(A1)
    A2 = smooth_probabilities(A2)

    likelihood_ratios_1 = np.sum(
        A1 * np.log(A1 / A2), axis=1
    )  # summing rows of KL divergences for each row to include stationary distributions
    kl_rows_1 = PI_1 * likelihood_ratios_1  # weighting by stationary distribution of A1

    likelihood_ratios_2 = np.sum(
        A2 * np.log(A2 / A1), axis=1
    )  # summing rows of KL divergences for each row to include stationary distributions
    kl_rows_2 = PI_2 * likelihood_ratios_2  # weighting by stationary distribution of A2

    total_symmetric_kl = (
        np.sum(kl_rows_1) + np.sum(kl_rows_2)
    ) / 2  # sum over all rows to get total symmetric KL divergence

    return total_symmetric_kl


def wasserstein_1_distance(A1, A2: np.ndarray):  ## suggested by Sho
    """count-normalized (probabilities)
    Wasserstein-1 distance between two transition matrices
    
    Actually, shouldn't this be a difference b/t CDFs?
    I should do it cumulatively
    
    """

    A1 = smooth_probabilities(A1)
    A2 = smooth_probabilities(A2)

    # Compute the absolute differences between the two matrices
    abs_diff = np.abs(A1 - A2)

    # Sum over all rows to get total Wasserstein-1 distance
    total_wasserstein_distance = np.sum(abs_diff)

    return total_wasserstein_distance


def experiment_weighted_wasserstein_1_distance(A1, A2: np.ndarray, PI_1: np.ndarray):
    """count-normalized (probabilities), stationary distribution weighted
    Wasserstein-1 distance between two transition matrices"""

    A1 = smooth_probabilities(A1)
    A2 = smooth_probabilities(A2)

    # Compute the absolute differences between the two matrices
    abs_diff = np.abs(A1 - A2)

    # Weight by the stationary distribution of A1
    weighted_abs_diff_rows = (
        np.sum(abs_diff, axis=1) * PI_1
    )  # sum over columns for each row, then weight by stationary distribution

    # Sum over all rows to get total weighted Wasserstein-1 distance
    total_weighted_wasserstein_distance = np.sum(weighted_abs_diff_rows)

    return total_weighted_wasserstein_distance


def KL_dirichlet_prior(
    n_states,
    alpha_counts,
    beta_counts,
    stationary
):
    """KL divergence representing the transition matrix prior as a Dirichlet distribution.
    A transition matrix is a square matrix where n_rows = n_cols = n_states

    Based on https://stats.stackexchange.com/questions/611747/statistical-comparison-of-two-markov-chain-transition-matrices

    This also takes into account the number of counts -- transitions with more counts are more trustworthy
    than ones with less, even if they have the same probabilities. 
    """
    prior = np.ones(n_states)

    row_kls = []

    for i in range(n_states):
        alpha = (
            prior + alpha_counts[i]
        )  # Dirichlet prior for row i of expected transition matrix

        beta = (
            prior + beta_counts[i]
        )  # Dirichlet prior for row i of simulated transition matrix

        ## KL divergence of 2 dirichlet distributions
        ## from https://statproofbook.github.io/P/dir-kl.html

        alpha_0 = np.sum(alpha)
        beta_0 = np.sum(beta)

        KL_exp_sim = (
            gammaln(alpha_0)
            - gammaln(beta_0)
            - np.sum(gammaln(alpha) - gammaln(beta))
            + np.sum((alpha - beta) * (digamma(alpha) - digamma(alpha_0)))
        )

        KL_exp_sim_weighted = KL_exp_sim * stationary[i]

        row_kls.append(KL_exp_sim_weighted)

    return np.sum(row_kls) ## sum over all rows to get total KL divergence


def symmetric_KL_dirichlet_prior(
    n_states,
    simulated_counts,
    expected_counts,
    simulated_stationary,
    expected_stationary,
):
    """
    Symmetric KL divergence with a Dirichlet prior
    """

    KL_exp_sim = KL_dirichlet_prior(
        n_states, expected_counts, simulated_counts, expected_stationary
    )

    KL_sim_exp = KL_dirichlet_prior(
        n_states, simulated_counts, expected_counts, simulated_stationary
    )

    return (KL_exp_sim + KL_sim_exp) / 2


def mean_squared_error(prediction, target):

    return np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)
