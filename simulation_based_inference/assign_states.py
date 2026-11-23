from pathlib import Path

import numpy as np


def parse_parms(path):
    with open(f"{path}/parms.inp") as f:
        params = f.readlines()
    for line in params:
        if line.strip().startswith("trajWrite"):
            traj_write = float(line.split()[2])
        if line.strip().startswith("timeStep"):
            time_step = float(line.split()[2])

    return traj_write, time_step


def parse_trajectory(path):

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

    if not Path(f"{path}/DATA/trajectory.xyz").exists():
        file_path = Path(f"{path}/trajectory.xyz")
    else:
        file_path = Path(f"{path}/DATA/trajectory.xyz")

    with open(f"{file_path}") as f:
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


def parse_binding_events(path):
    if not Path(f"{path}/DATA/assoc_dissoc_time.dat").exists():
        file_path = Path(f"{path}/assoc_dissoc_time.dat")
    else:
        file_path = Path(f"{path}/DATA/assoc_dissoc_time.dat")

    with open(f"{file_path}") as f:
        # OUTPUT form:
        # ITR:<simitr>,BOND/BREAK,<mol1>,<relIface1>,<mol2>,<relIface2>
        events = f.readlines()
        events = [line.split(",") for line in events]
    return events


## measurements on the raw trajectory file are made every 0.05s
def assign_states(path):
    """
    Assign states to the trajectory of a given individual in a population based on the events and parameters.

    Args:
        path (str): The path to the individual's data.
    """

    ## read all of the params, trajectories, and binding events

    traj_write, time_step = parse_parms(path)
    iteration_coms = parse_trajectory(path)
    binding_events = parse_binding_events(path)

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
