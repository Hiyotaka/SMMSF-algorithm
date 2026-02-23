from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smmsf import smmsf_clustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SMMSF clustering on a CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV path (features only).")
    parser.add_argument("--k", required=True, type=int, help="Target number of clusters.")
    parser.add_argument("--output", default="labels.csv", help="Output CSV path for labels.")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',').")
    parser.add_argument("--alpha", type=int, default=3, help="Number of MST rounds (default: 3).")
    parser.add_argument(
        "--stage1-threshold",
        type=float,
        default=1.5,
        help="Merge stage-1 threshold (default: 1.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data = np.loadtxt(input_path, delimiter=args.delimiter)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    labels = smmsf_clustering(
        data,
        args.k,
        alpha=args.alpha,
        stage1_threshold=args.stage1_threshold,
    )

    np.savetxt(output_path, labels.astype(int), fmt="%d", delimiter=args.delimiter)
    print(f"Saved labels: {output_path.resolve()}")
    print(f"Samples: {data.shape[0]}, Features: {data.shape[1]}, K: {args.k}")


if __name__ == "__main__":
    main()
