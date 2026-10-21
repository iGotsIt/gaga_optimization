import numpy as np
import argparse


def convert_to_internal_format(box_x, box_y, box_z, pdb_file, output_file):
    """
    Convert a nerdss pdb file coordinates from
    [0, size of box], to [-box/2, box/2] format.
    """

    with open(pdb_file, "r") as f:
        lines = f.readlines()

    updated_lines = []

    for line in lines:
        if line.startswith("ATOM"):
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            # Convert coordinates to internal format
            x_internal = x - box_x / 2
            y_internal = y - box_y / 2
            z_internal = z - box_z / 2

            # Update the line with new coordinates
            line = (
                line[:30]
                + f"{x_internal:8.3f}{y_internal:8.3f}{z_internal:8.3f}"
                + line[54:]
            )
        updated_lines.append(line)

    with open(output_file, "w") as f:
        f.writelines(updated_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert nerdss pdb file coordinates to internal format."
    )
    parser.add_argument("--pdb", type=str, help="Path to the input pdb file")
    parser.add_argument("--box_x", type=float, help="Box dimension in x direction")
    parser.add_argument("--box_y", type=float, help="Box dimension in y direction")
    parser.add_argument("--box_z", type=float, help="Box dimension in z direction")
    parser.add_argument(
        "--output", type=str, help="Path to the output pdb file with internal coordinates"
    )

    args = parser.parse_args()

    convert_to_internal_format(
        args.box_x, args.box_y, args.box_z, args.pdb, args.output
    )
