"""
Simulate FRET intensities from the eb-HMM parameters

All traces share a single covariance matrix estimated from all of the data.
"""

import numpy as np

stateindex = np.array(
    [
        [
            3,
            1,
            2,
            2,
            2,
            1,
        ],  ## each column is a state; numbers are 1-based level indices for intensity
        [3, 3, 1, 2, 3, 3],
        [3, 3, 3, 2, 1, 3],
    ]
)


class GaussianPDF:
    def __init__(self, mean, cov, seed):
        rng = np.random.default_rng(seed)
        self.mean = mean
        self.cov = cov
        self.rng = rng

    def sample(self, n_samples):
        return self.rng.multivariate_normal(self.mean, self.cov, n_samples)


## lifted from the HMM code
def build_select_matrices(stateindex):

    state_num = stateindex.shape[1]
    d = stateindex.shape[0]

    # green
    Mg = np.zeros((d, max(stateindex[0, :]), state_num))
    for i in range(state_num):
        index = stateindex[0, i] - 1
        Mg[0, index, i] = 1

    # blue
    Mb = np.zeros((d, max(stateindex[1, :]), state_num))
    for i in range(state_num):
        index = stateindex[1, i] - 1
        Mb[1, index, i] = 1

    # red
    Mr = np.zeros((d, max(stateindex[2, :]), state_num))
    for i in range(state_num):
        index = stateindex[2, i] - 1
        Mr[2, index, i] = 1

    return Mg, Mb, Mr


def simulate_intensities_with_gaussian_pdf(state_sequence, means, cov, sigmas, seed=42):
    gaussian_pdf = GaussianPDF(means, cov, seed)
    mean_intensities = gaussian_pdf.sample(1)
    Mg, Mb, Mr = build_select_matrices(stateindex)
    Nstates = stateindex.shape[1]
    M_all = [np.hstack([Mg[:, :, i], Mb[:, :, i], Mr[:, :, i]]) for i in range(Nstates)]

    state_intensity_distributions = []
    for state in range(Nstates):
        state_mean = M_all[state] @ mean_intensities.T
        state_cov = sigmas[:, :, state]
        state_distribution = GaussianPDF(state_mean.flatten(), state_cov, seed)
        state_intensity_distributions.append(state_distribution)

    print("state_intensity_distributions size:", len(state_intensity_distributions))

    emissions = []

    for state in state_sequence:
        emission = state_intensity_distributions[state].sample(1).flatten()
        emissions.append(emission)

    return np.array(emissions)
