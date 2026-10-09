"""FRET simulation engine: map a discrete state sequence to noisy 3-color
intensity traces, in the same format as the raw experimental data.

The experimental traces (fret_data/expData_3colorFRET/**/*.dat) have six
columns:

    time_s  I_donor  I_acceptor1  I_acceptor2  E12  E13

where the intensities are background-subtracted (so they can be negative),
E12 ~= I_a1 / (I_d + I_a1) and E13 ~= I_a2 / (I_d + I_a2), and both ratios
are NaN on frames whose total intensity is below ~100 (no bound, excited
donor).  The experimental ratios carry small per-movie crosstalk/gamma
corrections that are not reproduced here.

Each hidden state (see ``stochastic_opti/assign_states.py`` for the six-state
definition of the one_prot NERDSS run) has a mean intensity per channel.  The
observed intensity in every channel and frame is that mean plus zero-mean
noise drawn from either

  * a Normal distribution:  eps ~ N(0, sigma), or
  * a shifted Gamma:        eps ~ Gamma(k, theta) - k*theta, theta = sigma/sqrt(k)

Both have the same mean and standard deviation; the gamma shape k sets the
skewness (2/sqrt(k)), so large k approaches a Gaussian and small k gives a
heavy right tail, as is typical for photon-counting noise.  sigma can differ
per channel and per state.

Primary entry points
--------------------
``IntensityModel``                   per-state channel means + noise config
``intensities_from_efficiencies``    (total, E12, E13) -> channel means
``simulate_intensities(states, ...)``  states -> (F, 3) noisy intensities
``fret_ratios(intensities)``         (F, 3) intensities -> (F, 2) E12, E13
``simulate_trace_from_data_dir``     NERDSS DATA dir -> times, states, intensities

The default per-state (E12, E13) pairs below are placeholders; calibrate them
against the trained HMM emission means or the experimental traces before
using the output quantitatively.  The default total intensity and noise
levels are measured from the experimental data (see constants below).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stochastic_opti"))
from assign_states import assign_states, _read_parms  # noqa: E402

CHANNELS = ("donor", "acceptor1", "acceptor2")
N_CHANNELS = len(CHANNELS)
N_STATES = 6

# Measured across all 183 experimental traces: median total intensity on
# frames with valid ratios, and per-channel std on frames with NaN ratios
# (unbound background).
DEFAULT_BOUND_TOTAL = 350.0
DEFAULT_SIGMA = np.array([31.5, 16.0, 26.3])
DEFAULT_GAMMA_SHAPE = 4.0     # skew = 2/sqrt(k) = 1.0
DEFAULT_MIN_TOTAL = 100.0     # ratios are NaN below this total intensity

# Placeholder (E12, E13) per bound state.  State 0 (P free in solution) has
# no excited donor, so all channels are background (mean 0).
DEFAULT_STATE_EFFICIENCIES = {
    1: (0.20, 0.30),   # P bound to N beyond S2 (+x end)
    2: (0.60, 0.05),   # P bound to S1
    3: (0.50, 0.35),   # P bound to N between S1 and S2
    4: (0.10, 0.45),   # P bound to S2
    5: (0.10, 0.10),   # P bound to N beyond S1 (-x end)
}


def intensities_from_efficiencies(total, e12, e13) -> np.ndarray:
    """Channel intensities (I_d, I_a1, I_a2) with the given total and ratios.

    Inverts E12 = I_a1/(I_d + I_a1), E13 = I_a2/(I_d + I_a2) subject to
    I_d + I_a1 + I_a2 = total.  Inputs broadcast; output has a trailing
    axis of length 3.
    """
    total, e12, e13 = np.broadcast_arrays(*map(np.asarray, (total, e12, e13)))
    if np.any((e12 >= 1) | (e13 >= 1)):
        raise ValueError("efficiencies must be < 1")
    r2 = e12 / (1 - e12)
    r3 = e13 / (1 - e13)
    i_d = total / (1 + r2 + r3)
    return np.stack([i_d, i_d * r2, i_d * r3], axis=-1).astype(float)


def default_state_means(total: float = DEFAULT_BOUND_TOTAL) -> np.ndarray:
    means = np.zeros((N_STATES, N_CHANNELS))
    for s, (e12, e13) in DEFAULT_STATE_EFFICIENCIES.items():
        means[s] = intensities_from_efficiencies(total, e12, e13)
    return means


def _broadcast_sigma(sigma, n_states: int) -> np.ndarray:
    """Scalar, (3,) per-channel, (n_states, 3), or {state: scalar|(3,)}
    -> (n_states, 3) array."""
    if isinstance(sigma, dict):
        out = np.full((n_states, N_CHANNELS), np.nan)
        for s, v in sigma.items():
            out[s] = v
        if np.isnan(out).any():
            missing = np.flatnonzero(np.isnan(out).any(axis=1)).tolist()
            raise ValueError(f"sigma dict is missing states {missing}")
        return out
    arr = np.asarray(sigma, dtype=float)
    try:
        return np.broadcast_to(arr, (n_states, N_CHANNELS)).copy()
    except ValueError:
        raise ValueError(f"sigma must be scalar, shape ({N_CHANNELS},), or "
                         f"({n_states}, {N_CHANNELS}); got {arr.shape}") from None


@dataclass
class IntensityModel:
    """Emission model: per-state mean intensity per channel plus noise.

    Parameters
    ----------
    state_means : (n_states, 3) mean intensity of (donor, acceptor1,
                  acceptor2) in each state.  Build rows from FRET
                  efficiencies with ``intensities_from_efficiencies``.
    sigma       : noise std; scalar, per-channel (3,), per-state-and-channel
                  (n_states, 3), or a dict {state: scalar or (3,)}.
    noise       : "normal" or "gamma".
    gamma_shape : shape k of the gamma noise (ignored for normal noise).
    """
    state_means: np.ndarray = field(default_factory=default_state_means)
    sigma: float | np.ndarray | dict = field(
        default_factory=lambda: DEFAULT_SIGMA.copy())
    noise: str = "normal"
    gamma_shape: float = DEFAULT_GAMMA_SHAPE

    def __post_init__(self):
        self.state_means = np.asarray(self.state_means, dtype=float)
        if self.state_means.ndim != 2 or self.state_means.shape[1] != N_CHANNELS:
            raise ValueError(f"state_means must have shape (n_states, "
                             f"{N_CHANNELS}), got {self.state_means.shape}")
        self.n_states = self.state_means.shape[0]
        self.sigma = _broadcast_sigma(self.sigma, self.n_states)
        if (self.sigma < 0).any():
            raise ValueError("sigma must be non-negative")
        if self.noise not in ("normal", "gamma"):
            raise ValueError(f"noise must be 'normal' or 'gamma', "
                             f"got {self.noise!r}")
        if self.noise == "gamma" and self.gamma_shape <= 0:
            raise ValueError("gamma_shape must be positive")

    def sample_noise(self, states: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
        """(F, 3) zero-mean noise with std sigma[state, channel]."""
        sig = self.sigma[states]
        if self.noise == "normal":
            return rng.normal(0.0, sig)
        # Gamma(k, theta) has mean k*theta and var k*theta^2, so
        # theta = sig/sqrt(k) gives std sig; subtracting k*theta centers it.
        k = self.gamma_shape
        theta = sig / np.sqrt(k)
        return rng.gamma(k, theta) - k * theta


def simulate_intensities(states,
                         model: IntensityModel | None = None,
                         rng: np.random.Generator | int | None = None,
                         ) -> np.ndarray:
    """Generate noisy (donor, acceptor1, acceptor2) intensities per frame.

    Parameters
    ----------
    states : sequence of int state indices (one per frame).
    model  : IntensityModel; defaults to placeholder means and measured noise.
    rng    : np.random.Generator, an int seed, or None for fresh entropy.

    Returns
    -------
    (F, 3) float array of background-subtracted intensities.
    """
    if model is None:
        model = IntensityModel()
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    states = np.asarray(states, dtype=int)
    if states.ndim != 1:
        raise ValueError(f"states must be 1-D, got shape {states.shape}")
    if states.size and (states.min() < 0 or states.max() >= model.n_states):
        raise ValueError(f"state indices must be in [0, {model.n_states}), "
                         f"got range [{states.min()}, {states.max()}]")

    return model.state_means[states] + model.sample_noise(states, rng)


def fret_ratios(intensities: np.ndarray,
                min_total: float = DEFAULT_MIN_TOTAL) -> np.ndarray:
    """(F, 2) array of E12 = I_a1/(I_d+I_a1), E13 = I_a2/(I_d+I_a2), NaN on
    frames whose total intensity is below ``min_total``."""
    i_d, i_a1, i_a2 = np.asarray(intensities, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.column_stack([i_a1 / (i_d + i_a1), i_a2 / (i_d + i_a2)])
    ratios[i_d + i_a1 + i_a2 < min_total] = np.nan
    return ratios


def simulate_trace_from_data_dir(data_dir: str | Path,
                                 model: IntensityModel | None = None,
                                 rng: np.random.Generator | int | None = None,
                                 parms_path: str | Path | None = None,
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full pipeline: NERDSS DATA dir -> states -> noisy intensities.

    Returns
    -------
    times       : (F,) frame times in seconds.
    states      : (F,) int state indices.
    intensities : (F, 3) noisy (donor, acceptor1, acceptor2) intensities.
    """
    data_dir = Path(data_dir)
    if parms_path is None:
        parms_path = data_dir.parent / "parms.inp"

    states = np.asarray(assign_states(data_dir, parms_path), dtype=int)
    intensities = simulate_intensities(states, model=model, rng=rng)

    dt_s, traj_w = _read_parms(Path(parms_path))
    times = np.arange(len(states)) * dt_s * traj_w
    return times, states, intensities


def write_trace(path: str | Path, times, intensities,
                min_total: float = DEFAULT_MIN_TOTAL,
                header: str | None = None) -> None:
    """Write a trace in the experimental 6-column .dat layout
    (time, I_d, I_a1, I_a2, E12, E13); loadable with np.loadtxt."""
    data = np.column_stack([times, intensities,
                            fret_ratios(intensities, min_total)])
    np.savetxt(path, data, fmt="%16.7e", header=header or "",
               comments="# ")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate noisy 3-color FRET intensities from a NERDSS run.")
    parser.add_argument("data_dir", help="NERDSS DATA directory")
    parser.add_argument("--noise", choices=("normal", "gamma"),
                        default="normal")
    parser.add_argument("--sigma", type=float, nargs="+",
                        default=DEFAULT_SIGMA.tolist(),
                        help="noise std: one value for all channels, or three "
                             "(donor acceptor1 acceptor2)")
    parser.add_argument("--total", type=float, default=DEFAULT_BOUND_TOTAL,
                        help="mean total intensity of bound states")
    parser.add_argument("--gamma-shape", type=float,
                        default=DEFAULT_GAMMA_SHAPE,
                        help="gamma shape k (skew = 2/sqrt(k))")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--out", default=None,
                        help="output .dat (default: <data_dir>/../sim_trace.dat); "
                             "ground-truth states go to <out stem>_states.txt")
    args = parser.parse_args()

    if len(args.sigma) not in (1, N_CHANNELS):
        parser.error(f"--sigma takes 1 or {N_CHANNELS} values")

    model = IntensityModel(state_means=default_state_means(args.total),
                           sigma=args.sigma, noise=args.noise,
                           gamma_shape=args.gamma_shape)
    times, states, intensities = simulate_trace_from_data_dir(
        args.data_dir, model=model, rng=args.seed)

    out = (Path(args.out) if args.out
           else Path(args.data_dir).parent / "sim_trace.dat")
    write_trace(out, times, intensities, header=(
        f"noise={args.noise} sigma={args.sigma} total={args.total} "
        f"gamma_shape={args.gamma_shape} seed={args.seed}\n"
        "time_s  I_donor  I_acceptor1  I_acceptor2  E12  E13"))
    states_out = out.with_name(out.stem + "_states.txt")
    np.savetxt(states_out, np.column_stack([times, states]),
               fmt=["%9.3f", "%d"], header="time_s  state")
    print(f"wrote {out} and {states_out} ({len(states)} frames, "
          f"noise={args.noise})")


if __name__ == "__main__":
    main()
