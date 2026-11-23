import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from assign_states import assign_states
from simulate_fret import simulate_intensities_with_gaussian_pdf
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)

project_dir = "/home/jhu/jzhou59/projects_mjohn218/jeffery/data"

GAF_binding = Path(project_dir) / "GAF_Binding"


def build_dataframe():

    params_set = []

    param_map = [
        "log_ka_N",
        "log_kb_N",
        "log_ka_S1",
        "log_kb_S1",
        "log_ka_S2",
        "log_kb_S2",
        "area3dto1d",
    ]

    for run in glob(
        str(
            GAF_binding
            / "run_*"
        )
    ):
        params = Path(run).name.split("_")
        # expect run_<7 params>_seed<N>; skip anything else (e.g. the "run_123_seed1234" test dir)
        if len(params) != len(param_map) + 2:
            print(f"skipping malformed run dir: {run}")
            continue
        params_set.append(params[1:-1] + [run.split("/")[-1]])

    data_df = pd.DataFrame(data=params_set, columns=param_map + ["dir_name"])
    return data_df


# we can use the ground truth states to get the binding events
def extract_subsequences(
    states,
    max_inner_zero_run,
    min_nonzero_count,
) -> list[tuple[int, int]]:
    """Return half-open ``(start, end)`` frame indices of all subsequences of the form
       [all preceding zeros] + [binding event] + [all following zeros].

    Each event is bracketed by every zero between it and the neighbouring
    event (or the trajectory edge), so adjacent subsequences share the zeros
    between them.

    A "binding event" is a contiguous block of frames whose nonzero states
    are separated by runs of zeros no longer than ``max_inner_zero_run``,
    and which contains at least ``min_nonzero_count`` nonzero frames in total.

    An event that is already bound at the first frame, or still bound at the
    last frame, is kept with whatever bracket fits (possibly none), so a
    trajectory that is bound throughout comes back as one subsequence.

    Slice with ``trajectory[:, start:end]``. Returns an empty list if no
    qualifying subsequences exist.

    Trying to create single binding events here (not in the middle of a binding event)
    to avoid initial probability of zeros affecting estimates
    """

    # donor_channel = np.asarray(trajectory[0, :])
    # n = donor_channel.shape[0]

    # s = np.where(donor_channel > threshold, 1, 0)
    # if not s.any():
    #     return []

    states = np.asarray(states)  # assign_states returns a list; list != 0 is a single bool
    s = np.where(
        states != 0, 1, 0
    )  ## Convert states to binary mask to make tracking easier

    n = len(s)  # number of frames in the trajectory
    if n == 0:
        return []
    if not s.any():  # never bound -> whole trajectory is one all-zero subsequence
        return [(0, n)]

    # Runs of 1s as half-open [start, end) intervals.
    edges = np.diff(np.concatenate(([0], s, [0])))

    ## diff gets the starts and ends (starts are 1, ends are -1)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    # Merge runs of 1s separated by short zero gaps into single binding events.
    events = []  # [start, end) of each merged event
    cur_start, cur_end = starts[0], ends[0]
    for st, en in zip(starts[1:], ends[1:]):
        if st - cur_end <= max_inner_zero_run:
            cur_end = en
        else:
            events.append((cur_start, cur_end))
            cur_start, cur_end = st, en
    events.append((cur_start, cur_end))

    # Bracket each event with all zeros out to the neighbouring event (kept or
    # not) or the trajectory edge. Brackets only ever contain zeros, and adjacent
    # subsequences share the zeros between them.
    out = []
    for i, (ev_start, ev_end) in enumerate(events):
        if s[ev_start:ev_end].sum() < min_nonzero_count:
            continue
        prev_end = events[i - 1][1] if i > 0 else 0
        next_start = events[i + 1][0] if i + 1 < len(events) else n
        out.append((int(prev_end), int(next_start)))

    return out


def build_dataset():
    df = build_dataframe()
    hmm_stats = pickle.load(
        open(
            "/home/jhu/jzhou59/scratch_mjohn218/jeffery/hmm-nerdss/HMMS/hb_hmm_fit_results.pkl",
            "rb",
        )
    )

    sigmas = hmm_stats["sigma"]  # covariance matrix for the variation within a trace
    S = hmm_stats["S"]  # population sigma
    m = hmm_stats["m"]  # population mean

    for row in df.itertuples():

        print(f"Building row: {row.dir_name}")
        run_dir = GAF_binding / row.dir_name

        states = assign_states(run_dir)

        subsequences = extract_subsequences(
            states=states,
            max_inner_zero_run=0,
            min_nonzero_count=0,
        )

        for i, sequence in enumerate(subsequences):
            start, end = sequence
            subsequence_states = states[start:end]
            intensities = simulate_intensities_with_gaussian_pdf(
                subsequence_states, m, S, sigmas
            )

            np.save(
                run_dir / f"subsequence_{i}_intensities.npy",
                intensities,
            )


if __name__ == "__main__":
    build_dataset()
