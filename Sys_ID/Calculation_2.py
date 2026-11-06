"""Estimate b and Coulomb friction from steady speeds, then J from transients.

    tau_in = J * alpha + b * omega + F_c * sign(omega)
    omega(t) = omega_inf * (1 - exp(-b * t / J))

Exponential regression fits log(1 - omega / omega_inf) = slope*t,
where omega_inf = (tau_in - F_s) / b and J = -b / slope.
Fit each rotation direction separately, then use the equal-weight mean of
the two b values and the two absolute friction intercepts for all transients.
The final J is the arithmetic mean of the valid per-cycle estimates.
"""


import os
import numpy as np 
import math
import csv
import re

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
MIN_SAMPLES_PER_CYCLE = 5
REGRESSION_START_TIME_S = 1.0
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




def bF_calculation(omega_static: list, tau_in: list):
    """Average directional slopes and absolute intercepts, weighting sides equally."""
    omega_plot = np.asarray(omega_static, dtype=float)
    torque = np.asarray(tau_in, dtype=float)
    if omega_plot.ndim != 1 or torque.shape != omega_plot.shape:
        raise ValueError("Provide matching one-dimensional speed and torque lists")
    if not np.all(np.isfinite(omega_plot)) or not np.all(np.isfinite(torque)):
        raise ValueError("Steady speeds and torques must be finite")

    slopes = []
    friction_magnitudes = []
    for direction, label in ((1, "positive"), (-1, "negative")):
        selected = omega_plot * direction > 0
        speeds = omega_plot[selected]
        if len(speeds) < 2:
            raise ValueError(f"Need at least two steady-speed measurements for {label} rotation")
        # Fit tau = b_direction * omega + signed friction intercept.
        design = np.column_stack((speeds, np.ones_like(speeds)))
        coefficients, _, rank, _ = np.linalg.lstsq(design, torque[selected], rcond=None)
        if rank < 2:
            raise ValueError(f"Need distinct steady speeds for {label} rotation")
        b_direction, friction_intercept = map(float, coefficients)
        if not np.isfinite(b_direction) or b_direction <= 0:
            raise ValueError(f"The {label} rotation fit must produce a positive finite b")
        if not np.isfinite(friction_intercept):
            raise ValueError(f"The {label} rotation fit must produce finite friction")
        slopes.append(b_direction)
        friction_magnitudes.append(abs(friction_intercept))

    return float(np.mean(slopes)), float(np.mean(friction_magnitudes))

def exponential_regression(samples, b, F_s, start_time_s=REGRESSION_START_TIME_S):

    data = np.asarray(samples, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 3:
        raise ValueError("Provide at least three (time, omega, torque) samples")
    if not np.all(np.isfinite(data)):
        raise ValueError("Samples must contain only finite values")
    if not np.isfinite(b) or b <= 0 or not np.isfinite(F_s):
        raise ValueError("b must be positive and finite; F_s must be finite")

    if not np.isfinite(start_time_s) or start_time_s < 0:
        raise ValueError("Regression start time must be nonnegative and finite")

    time_s = data[:, 0]
    omega = data[:, 1]
    tau_in = data[0, 2]
    if np.any(time_s < 0) or np.any(np.diff(time_s) <= 0):
        raise ValueError("Sample times must be nonnegative and strictly increasing")
    if not np.allclose(data[:, 2], tau_in, rtol=1e-6, atol=1e-9):
        raise ValueError("The exponential model requires constant input torque")

    omega_inf = (tau_in - F_s) / b
    if abs(omega_inf) <= 1e-12 or np.ptp(omega) <= 1e-12:
        raise ValueError("A changing-speed transient is needed to estimate J")
    if omega[-1] * omega_inf <= 0 or np.any(omega * omega_inf < 0):
        raise ValueError("Recorded rotation disagrees with the predicted steady speed")

    # Apply the requested cutoff without shifting the time origin.
    selected = time_s >= start_time_s
    fit_time = time_s[selected]
    fit_omega = omega[selected]
    remaining_fraction = 1.0 - fit_omega / omega_inf
    tolerance = np.finfo(float).eps
    valid = remaining_fraction > tolerance
    if np.count_nonzero(valid) < 3:
        raise ValueError(
            "Need at least three samples at or after the start time below steady speed"
        )

    regression_time = np.concatenate(([0.0], fit_time[valid]))
    log_remaining = np.concatenate(([0.0], np.log(remaining_fraction[valid])))
    design = regression_time[:, np.newaxis]
    coefficients, _, rank, _ = np.linalg.lstsq(design, log_remaining, rcond=None)
    slope = float(coefficients[0])
    if rank < 1 or not np.isfinite(slope) or slope >= 0:
        raise ValueError("The regression must show exponential decay toward steady speed")
    J = -b / slope
    if not np.isfinite(J) or J <= 0:
        raise ValueError("The regression did not produce a finite positive inertia")
    return J



def main():
    if not os.path.isdir(RESULT_DIR):
        raise FileNotFoundError(f"result folder not found:{RESULT_DIR}")
    input_pattern = re.compile(r"excitation_cycle_\d+_torque_.*\.csv$", re.IGNORECASE)
    input_paths = sorted(
        os.path.join(RESULT_DIR, filename)
        for filename in os.listdir(RESULT_DIR)
        if input_pattern.match(filename)
    )
    if not input_paths:
        raise FileNotFoundError(f"No excitation CSV files found in {RESULT_DIR}")
    
    cycle_data = {}
    for path in input_paths:
        samples = read_experiment_csv(path)
        if len(samples) >= MIN_SAMPLES_PER_CYCLE:
            cycle_data[os.path.basename(path)] = samples
        else:
            print(f"Skipping {os.path.basename(path)}: too few valid samples")
    if not cycle_data:
        raise ValueError("No cycles contain enough valid samples")

    # Estimate steady speed from the last 10 readings (or all if fewer exist).
    omega = [
        float(np.mean([sample[1] for sample in samples[-10:]]))
        for samples in cycle_data.values()
    ]
    tau_in = [samples[-1][2] for samples in cycle_data.values()]
    b, Fc = bF_calculation(omega, tau_in)
    if not np.isfinite(b) or b <= 0:
        raise ValueError("Steady-state regression must produce a positive finite b")

    J_inertia = []
    for (filename, samples), steady_speed in zip(cycle_data.items(), omega):
        Fs = Fc * np.sign(steady_speed)
        try:
            J = exponential_regression(samples, b, Fs)
        except ValueError as error:
            print(f"Skipping {filename}: {error}")
            continue
        J_inertia.append(J)
        print(f"{filename}: J = {J:.8g} kg m^2")
    if not J_inertia:
        raise ValueError("No valid exponential regressions; cannot estimate final J")

    average_J = float(np.mean(J_inertia))
    print(f"Inertia J (mean of {len(J_inertia)} cycles): {average_J:.8g} kg m^2")
    print(f"Mean directional viscous friction b: {b:.8g} Nm/(rad/s)")
    print(f"Mean absolute Coulomb friction torque: {Fc:.8g} Nm")


if __name__ == "__main__":
    main()

