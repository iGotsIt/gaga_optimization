import pickle
import fcntl
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult, differential_evolution

import loss
import argparse

"""
Differential optimization of the parameters of the NERDSS simulation (mainly reaction rates) to fit experimental data.

We use differential evolution to search over large parameter spaces. 

We use the -loglikelihood of seeing a sequence of states, as given by the transition matrix of an HMM fitted to experimental data,
as the objective to optimize over. 
"""

# parameters to optimize -- order must match BOUNDS, so this has to stay a sequence
PARAMS = (
    "log10_ka_N",
    "log10_kb_N",
    "log10_ka_S1",
    "log10_kb_S1",
    "log10_ka_S2",
    "log10_kb_S2",
    "area3dto1d",
)

BOUNDS = [ # ~ 2 magnitude range for everything
    (-2.7, -0.7),  # log10 ka_N
    (-1.2, 1.3),  # log10 kb_N
    (-1.1, 1.0),  # log10 ka_S1
    (-0.83, 1.83),  # log10 kb_S1
    (-1.7, 0.7),  # log10 ka_S2
    (-0.9, 1.9),  # log10 kb_S2
    (1.0, 100.0),  # area3dto1d, linear
]

assert len(PARAMS) == len(BOUNDS)

# NERDSS resolves .mol files and all of its output paths (DATA/, PDB/, RESTARTS/)
# relative to the process cwd, so everything here is anchored to the script's
# directory rather than to whatever cwd happens to be.
BASE_DIR = Path(__file__).resolve().parent
MOL_FILES = ["P.mol", "N.mol", "S1.mol", "S2.mol"]
NERDSS_BIN = os.environ.get("NERDSS_BIN", "nerdss")
RUNS_DIR = BASE_DIR / "runs"

# global variable to hold the template for the params file
with open(BASE_DIR / "parms_template.inp", "r") as f:
    template = ("").join(
        f.readlines()
    )  ## stick the entire template into a single string, so we can fill it in with the parameters later


def build_params_file(params: np.ndarray):
    """
    Build the params.inp file for the NERDSS simulation, given a set of parameters.

    Args:
        params (np.ndarray): Array of parameters to optimize.
    """

    param_dict = {}

    for idx, param in enumerate(PARAMS):
        if param.startswith("log10_"):
            param_name = param[6:]  # remove "log10_" prefix
            param_dict[param_name] = 10 ** params[idx]  # convert from log10 to linear
        else:
            param_dict[param] = params[idx]  # store in the dictionary

    # Fill in the template with the parameter values
    params_file_content = template.format(**param_dict)

    # Write the filled-in template to parms.inp
    return params_file_content


def run_nerdss(params_text):
    RUNS_DIR.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="nerdss_run", dir=RUNS_DIR))

    # NERDSS opens "DATA/...", "PDB/...", "RESTARTS/..." directly and never creates
    # them itself, so they have to exist before it starts or the output is dropped.
    for subdir in ("DATA", "PDB", "RESTARTS"):
        (scratch / subdir).mkdir()

    parms_path = scratch / "parms.inp"
    parms_path.write_text(params_text)
    for mol in MOL_FILES:
        # absolute target: a relative symlink would resolve inside scratch, where
        # the .mol files don't exist
        os.symlink(BASE_DIR / mol, scratch / mol)

    cmd = [NERDSS_BIN, "-f", "parms.inp"]
    log_path = scratch / "nerdss.log"
    try:
        with log_path.open("w") as log_file:
            # cwd=scratch is what keeps concurrent workers from clobbering each
            # other's DATA/ -- NERDSS has no output-directory option
            subprocess.run(
                cmd, check=True, cwd=scratch, stdout=log_file, stderr=subprocess.STDOUT
            )
        return scratch
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"NERDSS failed with exit code {e.returncode}. See log file: {log_path}"
        ) from e


def parse_parms(individual_id):
    with open(f"{individual_id}/parms.inp") as f:
        params = f.readlines()
    for line in params:
        if line.strip().startswith("trajWrite"):
            traj_write = float(line.split()[2])
        if line.strip().startswith("timeStep"):
            time_step = float(line.split()[2])

    return traj_write, time_step


def parse_trajectory(individual_id):

    ## set up number of atoms and COM indices based on how many molecules + atoms per molecule
    p_atoms = 4
    n_atoms = 3
    s1_atoms = 3
    s2_atoms = 3

    num_p = 1
    num_n = 240
    num_s1 = 1
    num_s2 = 1

    num_atoms = (
        num_p * p_atoms + num_n * n_atoms + num_s1 * s1_atoms + num_s2 * s2_atoms
    )

    com_indices = (
        [0]
        + [(p_atoms * num_p) + n_atoms * k for k in range(num_n)]
        + [(p_atoms * num_p) + (n_atoms * num_n)]
        + [(p_atoms * num_p) + (n_atoms * num_n) + (s1_atoms * num_s1)]
    )

    iteration_coms = []

    with open(f"{individual_id}/DATA/trajectory.xyz") as f:
        while True:
            line = next(f, None)
            if line is None:
                break
            if line.strip().startswith(
                "iteration"
            ):  # cycle through every line in the file until we hit the end
                coords = []
                for i in range(
                    num_atoms
                ):  # build a list of the coordinates of the COMs for this iteration
                    line = next(f)
                    if i in com_indices:
                        parts = line.split()
                        coords.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )
                iteration_coms.append(
                    np.array(coords)
                )  # add the COMs for this iteration to the list

        iteration_coms = np.array(
            iteration_coms
        )  # convert the list of COMs to a numpy array for easier manipulation

    return iteration_coms


def parse_binding_events(individual_id):
    with open(f"{individual_id}/DATA/assoc_dissoc_time.dat") as f:
        # OUTPUT form:
        # ITR:<simitr>,BOND/BREAK,<mol1>,<relIface1>,<mol2>,<relIface2>
        events = f.readlines()
        events = [line.split(",") for line in events]
    return events


## measurements on the raw trajectory file are made every 0.05s
def assign_states(individual_id):
    """
    Assign states to the trajectory of a given individual in a population based on the events and parameters.

    Args:
        individual_id (int): The ID of the individual for which to assign states.
    """

    ## read all of the params, trajectories, and binding events

    traj_write, time_step = parse_parms(individual_id)
    iteration_coms = parse_trajectory(individual_id)
    binding_events = parse_binding_events(individual_id)

    ## match binding events to iterations -- if no binding event window within a certain iteration, it is unbound.

    states = []

    event_map = {}

    for event in binding_events:
        ## build event intervals map
        simitr = int(event[0].split(":")[1])
        action = event[1]
        mol1 = event[2]
        mol1_idx = event[3]
        mol2 = event[5]
        mol2_idx = event[6]

        if action == "BOND":
            # create a list of interaction intervals
            if f"{mol1}_{mol1_idx}_{mol2}_{mol2_idx}" not in event_map:
                event_map[f"{mol1}_{mol1_idx}_{mol2}_{mol2_idx}"] = []
            event_map[f"{mol1}_{mol1_idx}_{mol2}_{mol2_idx}"].append([simitr, None])
        elif action == "BREAK":
            # if there is a break event, we need to find the last bond event and set its end time
            event_map[f"{mol1}_{mol1_idx}_{mol2}_{mol2_idx}"][-1][1] = simitr
    # s1 is frame[-2], s2 is frame[-1]

    for frame_idx, frame in enumerate(iteration_coms):
        state = None
        frame_iter = (
            frame_idx * traj_write
        )  # this gets the iteration number for this frame
        possible_states = []
        for mol_pair, intervals in event_map.items():
            for start_iter, end_iter in intervals:
                start_time = start_iter
                if end_iter is None:
                    end_iter = float("inf")
                if start_time <= frame_iter <= end_iter:
                    mol_pair = mol_pair.split("_")
                    mol_pair = f"{mol_pair[0]}_{mol_pair[2]}"
                    if mol_pair in ("N_P", "P_N"):
                        # if p is farther than s1 toward -x it's position 1
                        # if p is farther than s2 toward +x it's position 5
                        p_x = frame[0][0]
                        s1_x = frame[-2][0]
                        s2_x = frame[-1][0]
                        if p_x < s1_x:
                            state = 1
                        elif p_x > s2_x:
                            state = 5
                        else:  # if p is between s1 and s2, it's position 3
                            state = 3
                    elif mol_pair in ("S1_P", "P_S1"):
                        state = 2
                    elif mol_pair in ("S2_P", "P_S2"):
                        state = 4
                    possible_states.append(state)

        # go through possible states and assign one by precedence -- specific binding first
        if 2 in possible_states:
            state = 2
        elif 4 in possible_states:
            state = 4
        elif 1 in possible_states:
            state = 1
        elif 5 in possible_states:
            state = 5
        elif 3 in possible_states:
            state = 3
        else:
            state = 0  # unbound

        states.append(state)

    return states



def extract_subsequences(
    states: list[int],
    bracket_frames: int = 10,
    max_inner_zero_run: int = 2,
    min_nonzero_count: int = 2,
) -> list[list[int]]:
    """Extract all subsequences of the form
       [bracket_frames zeros] + [binding event] + [bracket_frames zeros].

    A "binding event" is a contiguous block of frames whose nonzero states
    are separated by runs of zeros no longer than ``max_inner_zero_run``,
    and which contains at least ``min_nonzero_count`` nonzero frames in total.

    With the default settings and a 0.05 s frame stride this requires:
      - 0.5 s of pure state 0 on each side (10 frames)
      - the binding event may contain runs of up to 0.1 s of state 0 inside
      - the event must contain at least 2 nonzero frames (so single isolated
        nonzero frames flanked entirely by zeros are excluded)

    Returns an empty list if no qualifying subsequences exist.

    Trying to create single binding events here (not in the middle of a binding event) 
    to avoid initial probability affecting likelihood estimates
    """
    s = list(states)
    n = len(s)

    nz = [i for i, v in enumerate(s) if v != 0]
    if not nz:
        return []

    # Cluster nonzero indices into events.  Two consecutive nonzero frames
    # belong to the same event if separated by ≤ max_inner_zero_run zeros.
    events: list[tuple[int, int, int]] = []   # (first_nz, last_nz, count)
    cur_start = nz[0]
    cur_end   = nz[0]
    cur_count = 1
    for i in nz[1:]:
        if i - cur_end - 1 <= max_inner_zero_run:
            cur_end   = i
            cur_count += 1
        else:
            events.append((cur_start, cur_end, cur_count))
            cur_start, cur_end, cur_count = i, i, 1
    events.append((cur_start, cur_end, cur_count))

    out: list[list[int]] = []
    for start, end, count in events:
        if count < min_nonzero_count:
            continue
        if start < bracket_frames or end + bracket_frames >= n:
            continue
        # Bracket zones must be pure zeros.
        if any(v != 0 for v in s[start - bracket_frames:start]):
            continue
        if any(v != 0 for v in s[end + 1:end + 1 + bracket_frames]):
            continue
        out.append(s[start - bracket_frames:end + 1 + bracket_frames])

    return out


def create_transition_counts(state_sequence, num_states=6):

    """
    Create a count matrix of transitions between states.

    Args:
        state_sequence (list): A list of states (integers).
        num_states (int): The number of unique states.
    """

    transition_counts = np.zeros((num_states, num_states), dtype=int)

    for current_state, next_state in itertools.pairwise(state_sequence):
        transition_counts[current_state, next_state] += 1

    return transition_counts


def create_transition_matrix(state_sequence, num_states=6):
    """
    Create a transition matrix from a sequence of states.

    Args:
        state_sequence (list): A list of states (integers).
        num_states (int): The number of unique states.
    """

    transition_counts = create_transition_counts(state_sequence, num_states)

    # Normalize the rows to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts, dtype=float),
        where=row_sums != 0,
    )

    return transition_matrix


def get_stationary_distribution(transition_matrix):
    """
    Find the stationary distribution of a Markov chain. The stationary distribution is the
    normalized left eigenvector corresponding to the eigenvalue 1 of the transition matrix.
    Because we are working with a finite number of states, the stationary distribution exists and is unique.
    """
    eigenvalues, l_eigenvectors = sp.linalg.eig(
        transition_matrix, left=True, right=False
    )
    order = np.argsort(np.abs(eigenvalues - 1))  # eigenvalue closest to 1 first
    eigenvalues = eigenvalues[order]
    l_eigenvectors = l_eigenvectors[:, order]
    v = np.real(
        l_eigenvectors[:, 0]
    )  # get the real part of the left eigenvector corresponding to eigenvalue 1
    pi = v / np.sum(v)  # normalize to get stationary distribution

    return pi, eigenvalues


class EvalLogger:
    """
    Append-only CSV holding one row per objective evaluation: the parameter
    vector that was tried and the value it scored.

    differential_evolution with workers=-1 evaluates the population in child
    processes, so several processes append to this file at once. Each row is
    written under an exclusive flock so lines can't interleave, and only a Path
    is kept as state so the logger survives being pickled out to the workers.
    """

    COLUMNS = (
        "iteration",
        "timestamp",
        "pid",
        *PARAMS,
        "objective",
        "seconds",
        "status",
    )

    def __init__(self, path):
        self.path = Path(path)

    def write_header(self):
        """Called once from the parent, before any worker starts."""
        with self.path.open("w") as f:
            f.write(",".join(self.COLUMNS) + "\n")

    def log(self, params, objective, seconds, status, iteration):
        row = [
            str(iteration),
            datetime.now().isoformat(timespec="seconds"),
            str(os.getpid()),
            *(f"{v:.6g}" for v in params),
            f"{objective:.6g}",
            f"{seconds:.1f}",
            status,
        ]
        line = ",".join(row) + "\n"
        with self.path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)


class Evaluator:
    """
    Class to evaluate the objective function for a given set of parameters.

    Runs a NERDSS simulation with the given params, then returns the calculated loss
    """

    def __init__(
        self,
        initial_probabilities,
        transition_matrix,
        transition_counts,
        callback,
        loss_function: Callable,
        logger=None,
    ):
        self.initial_probabilities = initial_probabilities

        ## we smooth the transition matrix here for consistency with the loss functions
        self.transition_matrix = loss.smooth_probabilities(transition_matrix)
        self.transition_counts = transition_counts
        self.logger = logger
        self.callback = callback


        ## there's a little bit of error from smoothing the stationary 
        self.loss_function = loss_function
        self.stationary_distribution = get_stationary_distribution(transition_matrix)[0]

    def __call__(self, params: np.ndarray):
        params_text = build_params_file(params)
        print("Running NERDSS with parameters:", params)
        started = time.time()
        self.iteration = self.callback.iteration

        try:
            nerdss_result_path = run_nerdss(params_text)
        except RuntimeError as e:
            print(f"Error running NERDSS: {e}")
            ## a failed nerdss run should return a large value to indicate failure
            self._log(params, np.inf, started, "nerdss_failed", self.iteration)
            return np.inf

        status = "ok"

        try:
            state_sequence = assign_states(
                individual_id=nerdss_result_path
            )  # full path to the run directory, not just its basename

            simulation_transition_matrix = create_transition_matrix(state_sequence)
            simulation_transition_matrix = loss.smooth_probabilities(
                simulation_transition_matrix
            )  # smooth the transition matrix to avoid rows of zeros
            simulation_stationary_distribution = get_stationary_distribution(
                simulation_transition_matrix
            )[0]

            simulated_counts = create_transition_counts(state_sequence)

            match self.loss_function:
                case loss.negative_log_likelihood:
                    result = self.loss_function(
                        state_sequence,
                        self.initial_probabilities,
                        self.transition_matrix,
                    )

                case loss.KL_divergence:
                    result = self.loss_function(
                        self.transition_matrix,
                        simulation_transition_matrix,
                        self.stationary_distribution,
                    )

                case loss.symmetric_KL_divergence:
                    result = self.loss_function(
                        self.transition_matrix,
                        simulation_transition_matrix,
                        self.stationary_distribution,
                        simulation_stationary_distribution,
                    )

                case loss.wasserstein_1_distance:
                    result = self.loss_function(
                        self.transition_matrix, simulation_transition_matrix
                    )

                case loss.experiment_weighted_wasserstein_1_distance:
                    result = self.loss_function(
                        self.transition_matrix,
                        simulation_transition_matrix,
                        self.stationary_distribution,
                    )

                case loss.KL_dirichlet_prior:
                    result = self.loss_function(
                        self.transition_matrix.shape[0],
                        self.transition_counts,
                        simulated_counts,
                        self.stationary_distribution,
                    )

                case loss.symmetric_KL_dirichlet_prior:
                    result = self.loss_function(
                        self.transition_matrix.shape[0],
                        simulated_counts,
                        self.transition_counts,
                        simulation_stationary_distribution,
                        self.stationary_distribution,
                    )

                case _:
                    raise ValueError(
                        f"Unknown loss function: {self.loss_function}. Please use one of the defined loss functions."
                    )

        except Exception as e:  # shut up
            print(f"Error processing NERDSS output: {e}")
            result = np.inf
            status = "parse_failed"
        finally:
            shutil.rmtree(nerdss_result_path)
        self._log(params, result, started, status, self.iteration)
        return result

    def _log(self, params, result, started, status, iteration):
        if self.logger is not None:
            self.logger.log(params, result, time.time() - started, status, iteration)


class DiffEvolutionCallback:
    """Per-generation log of the best individual so far. Runs in the parent
    process, so it needs no locking."""

    def __init__(self, log_file="optimization_log.txt"):
        self.iteration = 0
        self.log_file = log_file
        with open(self.log_file, "w") as log:
            # one column per parameter -- the array repr used to land in the file
            # with spaces and brackets, which nothing could read back as CSV
            log.write(",".join(("iteration", *PARAMS, "objective")) + "\n")

    def __call__(self, intermediate_result: OptimizeResult):
        self.iteration += 1
        print(
            f"Iteration {self.iteration}: Solution: {intermediate_result.x}, Objective: {intermediate_result.fun}"
        )
        with open(self.log_file, "a") as log:
            row = (
                str(self.iteration),
                *(f"{v:.6g}" for v in intermediate_result.x),
                f"{intermediate_result.fun:.6g}",
            )
            log.write(",".join(row) + "\n")


def run_loop(initial_params=None, loss_function=loss.symmetric_KL_divergence):
    """Run the optimization loop."""
    """We uniquely identify each individual in the iteration to track"""
    initial_probabilities = np.load("initial_matrix_multi_precision.npy")
    # Smooth once here rather than per evaluation: the empirical matrix contains
    # exact zeros for transitions never seen experimentally, which would make the
    # NLL infinite (and log() complain) the moment a simulation takes one. The
    # initial vector is left alone -- it has no exact zeros, and smoothing it
    # would substantially cheapen starting outside state 0.

    # transition_matrix = loss.smooth_probabilities(
    #     np.load("transition_matrix_multi_precision.npy")
    # )

    with open("multi_hmm_fit_results_multistart.pkl", "rb") as f:
        hmm = pickle.load(f)

    transition_matrix = hmm["transition_mat"]
    transition_counts = hmm["trans_num"]


    popsize = 14 ## 14 * 7 = 98 mutants per population
    maxiter = 30 ## triple # iters
    tag = f"{popsize}_{maxiter}_{datetime.now():%Y%m%d_%H%M%S}_{loss_function.__name__}"

    eval_logger = EvalLogger(BASE_DIR / f"evaluations_{tag}.csv")
    eval_logger.write_header()

    generation_log = BASE_DIR / f"optimization_log_{tag}.csv"
    callback = DiffEvolutionCallback(log_file=generation_log)

    evaluator = Evaluator(
        initial_probabilities,
        transition_matrix,
        transition_counts,
        callback,
        loss_function,
        eval_logger,
    )

    print("Starting optimization...")
    print(f"Logging every evaluation to {eval_logger.path}")
    print(f"Logging per-generation bests to {generation_log}")
    result = differential_evolution(
        evaluator,
        BOUNDS,
        maxiter=maxiter,
        workers=-1,
        callback=callback,
        popsize=popsize,
        x0=initial_params,
        updating="deferred",
        polish=False, # don't polish -- will speed up optimization
    )

    print(result)



def main(initial_params, loss_function):
    run_loop(initial_params, loss_function)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parameter optimization for NERDSS."
    )

    parser.add_argument(
        "--initial_params",
        nargs="*",
        type=float,
        help="Initial parameters for optimization.",
        required=False,
    )

    parser.add_argument(
        "--loss_function",
        type=str,
        default="symmetric_KL_divergence",
        help="Loss function to use for optimization. Options: negative_log_likelihood, KL_divergence, symmetric_KL_divergence, wasserstein_1_distance, experiment_weighted_wasserstein_1_distance.",
    )

    args = parser.parse_args()

    loss_function_map = {
        "negative_log_likelihood": loss.negative_log_likelihood,
        "KL_divergence": loss.KL_divergence,
        "symmetric_KL_divergence": loss.symmetric_KL_divergence,
        "wasserstein_1_distance": loss.wasserstein_1_distance,
        "experiment_weighted_wasserstein_1_distance": loss.experiment_weighted_wasserstein_1_distance,
        "KL_dirichlet_prior": loss.KL_dirichlet_prior,
        "symmetric_KL_dirichlet_prior": loss.symmetric_KL_dirichlet_prior,
    }

    main(
        initial_params=args.initial_params,
        loss_function=loss_function_map.get(
            args.loss_function, loss.symmetric_KL_divergence
        ),
    )
