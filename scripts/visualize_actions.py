#!/usr/bin/env python
"""Visualize OpenVLA-style action sequences from CSV or a fake example."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED_FIELDS = ("step", "dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot action XYZ, rotation, and gripper curves from a CSV file."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV file with fields: step,dx,dy,dz,droll,dpitch,dyaw,gripper. If omitted, a fake example is used.",
    )
    return parser.parse_args()


def load_actions_from_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file is missing required fields: {', '.join(missing)}")

        data = {field: [] for field in REQUIRED_FIELDS}
        for row in reader:
            for field in REQUIRED_FIELDS:
                data[field].append(float(row[field]))

    if not data["step"]:
        raise ValueError("CSV file contains no action rows.")
    return data


def make_fake_actions(num_steps: int = 40) -> dict[str, list[float]]:
    data = {field: [] for field in REQUIRED_FIELDS}
    for step in range(num_steps):
        phase = step / max(num_steps - 1, 1)
        data["step"].append(float(step))
        data["dx"].append(0.03 * phase)
        data["dy"].append(0.02 * (1.0 - phase))
        data["dz"].append(0.015 if 10 <= step <= 24 else -0.005)
        data["droll"].append(0.01 * (step % 6 - 3))
        data["dpitch"].append(0.008 * (3 - step % 6))
        data["dyaw"].append(0.02 * (phase - 0.5))
        data["gripper"].append(1.0 if step < 22 else 0.0)
    return data


def plot_curves(data: dict[str, list[float]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = data["step"]
    outputs: list[Path] = []

    xyz_path = output_dir / "action_xyz_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, data["dx"], label="dx")
    plt.plot(steps, data["dy"], label="dy")
    plt.plot(steps, data["dz"], label="dz")
    plt.title("Action Translation Delta Over Time")
    plt.xlabel("Step")
    plt.ylabel("Translation Delta")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(xyz_path, dpi=180)
    plt.close()
    outputs.append(xyz_path)

    rotation_path = output_dir / "action_rotation_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, data["droll"], label="droll")
    plt.plot(steps, data["dpitch"], label="dpitch")
    plt.plot(steps, data["dyaw"], label="dyaw")
    plt.title("Action Rotation Delta Over Time")
    plt.xlabel("Step")
    plt.ylabel("Rotation Delta")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(rotation_path, dpi=180)
    plt.close()
    outputs.append(rotation_path)

    gripper_path = output_dir / "gripper_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.step(steps, data["gripper"], where="post", label="gripper")
    plt.title("Gripper Command Over Time")
    plt.xlabel("Step")
    plt.ylabel("Gripper Command")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(gripper_path, dpi=180)
    plt.close()
    outputs.append(gripper_path)

    return outputs


def main() -> int:
    args = parse_args()
    if args.input:
        input_path = Path(args.input)
        print(f"Loading action CSV: {input_path}")
        data = load_actions_from_csv(input_path)
    else:
        print("No input CSV provided; generating a fake action sequence example.")
        data = make_fake_actions()

    outputs = plot_curves(data, Path("results/figures"))
    print("Generated action visualization figures:")
    for path in outputs:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
