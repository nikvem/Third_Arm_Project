"""
The fitted model is:

    torque_input = J * angular_acceleration + b * omega + coulomb_friction_torque * sign(omega)

This assumes there is no external load torque and the torque in the CSV is
motor shaft torque. The fit combines all matching experiment CSVs in result/.
Using linear regression to calculate the J,b and F_c, like using input torque to calculate these value, then using least square produt to reduce the difference between the realist and expected. 
"""

import csv
import math
import os
import re
import statistics

import numpy as np


RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
OUTPUT_FILE = "motor_parameter_estimate.csv"
SMOOTHING_WINDOW = 5  # odd number of samples used to reduce speed noise
MIN_SAMPLES_PER_CYCLE = SMOOTHING_WINDOW + 2


def read_experiment_csv(path):
    """Read and validate time, speed, and commanded torque from one CSV."""
    samples = []
    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                time_s = float(row["time_s"])
                omega = float(row["omega_rad_s"])
                torque = float(row["torque_input"])
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in (time_s, omega, torque)):
                samples.append((time_s, omega, torque))

    # Ensure ordered samples and discard duplicate/backward timestamps.
    samples.sort(key=lambda sample: sample[0])
    ordered = []
    for sample in samples:
        if not ordered or sample[0] > ordered[-1][0]:
            ordered.append(sample)
    return ordered

# performing the dynamic equation part
def build_regression_rows(samples, window=SMOOTHING_WINDOW):
    if len(samples) < max(MIN_SAMPLES_PER_CYCLE, window + 2):
        return [], []
    if window < 1 or window % 2 == 0:
        raise ValueError("SMOOTHING_WINDOW must be a positive odd integer")
    #loaded the time, speed and torque for calculating the data. 
    times = [sample[0] for sample in samples]
    speeds = [sample[1] for sample in samples]
    torques = [sample[2] for sample in samples]
    half_window = window // 2

    # Centered moving average reduces encoder noise before differentiating.
    smoothed_speeds = [
        sum(speeds[index - half_window:index + half_window + 1]) / window
        for index in range(half_window, len(samples) - half_window)
    ]
    rows = []
    targets = []

    # Central difference of the smoothed speed; omit edges where the filter does not have a full window and skip zero-speed sign ambiguity.

    for local_index in range(1, len(smoothed_speeds) -1):
        source_index=half_window + local_index
        left_index = source_index -1
        right_index = source_index +1
        delta_t = times[right_index] - times[left_index]
        omega = smoothed_speeds[local_index]
        if delta_t <= 0 or abs(omega) < 1e-9:
            continue
        alpha = (smoothed_speeds[local_index + 1] - smoothed_speeds[local_index - 1]) / delta_t
        rows.append([alpha, omega, math.cpysign(1.0,omega)]) # appedning the omega_dot, omega and sign_omega into a same array to build the matrix
        targets.append(torques[source_index])
    return rows, targets

#linear regression part, using least square method to get close to correct value
def estimate_parameters(cycle_data):
    """Fit inertia, viscous friction, and Coulomb friction by least squares."""
    design_rows = []
    target_values = []
    for samples in cycle_data.values():
        rows, targets = build_regression_rows(samples)
        design_rows.extend(rows)
        target_values.extend(targets)

    if not design_rows:
        raise ValueError("No usable samples; collect longer excitation cycles first")

    design = np.asarray(design_rows, dtype=float)
    target = np.asarray(target_values, dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError(
            "The data cannot uniquely identify J, b, and Coulomb friction F. "
            "Collect cycles at different torque levels; use both torque directions if possible."
        )

    predicted = design @ coefficients
    residuals = target - predicted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "inertia_J_kg_m2": float(coefficients[0]),
        "viscous_friction_b_Nm_per_rad_s": float(coefficients[1]),
        "coulomb_friction_torque_Nm": float(coefficients[2]),
        "rmse_torque_Nm": rmse,
        "samples_used": int(len(target)),
        "design_rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
    }


def main():
    if not os.path.isdir(RESULT_DIR):
        raise FileNotFoundError(f"Result folder not found: {RESULT_DIR}")

    input_pattern = re.compile(r"excitation_cycle_\d+_torque_.*\.csv$", re.IGNORECASE)
    input_paths = sorted(
        os.path.join(RESULT_DIR, filename)
        for filename in os.listdir(RESULT_DIR)
        if input_pattern.match(filename)
    )
    if not input_paths:
        raise FileNotFoundError(f"No excitation CSV files found in {RESULT_DIR}")

    cycle_data = {}
    used_files = []
    for path in input_paths:
        samples = read_experiment_csv(path)
        if len(samples) >= MIN_SAMPLES_PER_CYCLE:
            cycle_data[os.path.basename(path)] = samples
            used_files.append(os.path.basename(path))
        else:
            print(f"Skipping {os.path.basename(path)}: too few valid samples")

    estimate = estimate_parameters(cycle_data)
    output_path = os.path.join(RESULT_DIR, OUTPUT_FILE)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "inertia_J_kg_m2",
                "viscous_friction_b_Nm_per_rad_s",
                "coulomb_friction_torque_Nm",
                "rmse_torque_Nm",
                "samples_used",
                "cycles_used",
                "design_rank",
                "condition_number",
                "input_files",
            ],
        )
        writer.writeheader()
        writer.writerow({
            **estimate,
            "cycles_used": len(cycle_data),
            "input_files": ";".join(used_files),
        })

    print("Estimated motor parameters from", len(cycle_data), "cycles:")
    print(f"  Inertia J: {estimate['inertia_J_kg_m2']:.8g} kg m^2")
    print(f"  Viscous friction b: {estimate['viscous_friction_b_Nm_per_rad_s']:.8g} Nm/(rad/s)")
    print(f"  Coulomb friction torque: {estimate['coulomb_friction_torque_Nm']:.8g} Nm")
    print(f"  Torque fit RMSE: {estimate['rmse_torque_Nm']:.8g} Nm")
    print(f"Wrote parameter estimate: {output_path}")
    print("Check that runs use multiple torque levels; positive and negative torque runs improve friction identification.")


if __name__ == "__main__":
    main()
