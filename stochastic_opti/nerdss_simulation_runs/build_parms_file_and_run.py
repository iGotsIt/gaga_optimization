import concurrent.futures
import os
import random
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np

PARAMS = (
    "log10_ka_N",
    "log10_kb_N",
    "log10_ka_S1",
    "log10_kb_S1",
    "log10_ka_S2",
    "log10_kb_S2",
    "area3dto1d",
)

MOL_FILES = ["P.mol", "N.mol", "S1.mol", "S2.mol"]
BASE_DIR = Path(__file__).resolve().parent

NERDSS_BIN = os.environ.get("NERDSS_BIN", "nerdss")


def build_parms_file(params, parms_file_path, parms_template_path):

    # Read the template file
    with open(parms_template_path, "r") as template_file:
        template = template_file.read()

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
    with open(parms_file_path, "w") as parms_file:
        parms_file.write(params_file_content)

    return params_file_content


def main():
    # sample a few parameter sets to run
    numbers = random.sample(
        range(1, 7700), 200
    )  # select 200 non-repeating numbers from 1 to 7700

    with open(
        "/home/jzhou59/data_mjohn218/jzhou59/gaga_optimization/stochastic_opti/evaluations_100_10_20260824_160356.csv",
        "r",
    ) as f:
        lines = f.readlines()

    lines = np.array(lines)
    sampled_params = lines[numbers]

    # sample param configurations across a bunch of runs

    rand_ids = []  # list of rand_ids that identify each run

    DATA_DIR = Path(BASE_DIR) / "nerdss_data"
    os.makedirs(DATA_DIR, exist_ok=True)

    for line in sampled_params:
        params_cols = range(3, 10)  # parameters are columns 3, 4, 5, 6, 7, 8, 9
        params_array = np.array(line.split(","))
        params = np.array([float(x) for x in params_array[params_cols]])

        # get a random identifier
        rand_id = secrets.token_bytes(32)

        # make the directory with the random_id identifier and symlink the mol files into it
        scratch = Path(BASE_DIR) / "nerdss_data" / str(rand_id)
        os.makedirs(scratch, exist_ok=True)

        for mol in MOL_FILES:
            # absolute target: a relative symlink would resolve inside scratch, where
            # the .mol files don't exist
            os.symlink(BASE_DIR / mol, scratch / mol)

        parms_file_path = Path(scratch) / "parms.inp"
        parms_template_path = "parms_template.inp"

        with open(scratch / "params.log", "w") as f:
            f.write(f"Run: {params}\n")

        build_parms_file(params, parms_file_path, parms_template_path)

        rand_ids.append(rand_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        cmd = [NERDSS_BIN, "-f", "parms.inp"]

        DATA_DIR = Path(BASE_DIR) / "nerdss_data"

        futures = []

        for rand_id in rand_ids:
            run_dir = DATA_DIR / str(rand_id)
            futures.append(
                executor.submit(subprocess.run, cmd, cwd=run_dir, check=True)
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(f"Run result: {result}")
            except Exception as e:
                print(f"Error occurred: {e}")


def run_single_simulation():
    params = np.array(
        [-2.61249, 0.759774, -1.98483, -1.20701, -1.65477, -1.4673, 51.7211]
    )

    os.makedirs(Path(BASE_DIR) / "nerdss_data" / "single_run", exist_ok=True)

    build_parms_file(
        params,
        Path(BASE_DIR) / "nerdss_data" / "single_run" / "parms.inp",
        "parms_template.inp",
    )

    for mol in MOL_FILES:
        os.symlink(BASE_DIR / mol, Path(BASE_DIR) / "nerdss_data" / "single_run" / mol)

    cmd = [NERDSS_BIN, "-f", "parms.inp"]
    subprocess.run(cmd, cwd=Path(BASE_DIR) / "nerdss_data" / "single_run", check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        run_single_simulation()
    else:
        main()
