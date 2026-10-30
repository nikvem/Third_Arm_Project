"""
The new calculation2.py, using the System ID instead of linear regression to find J, b, F

tau_in = J*alpha + b*omega+F_s * omega_sign
finding the data point where alpha = 0, and set the figure out b and F_s using linear regression
collecting many omega value under different tau_in, need at least 20 for the positive and 20 for the negative. 
using linear regression to calculate gradient which is b, and y=intercept which is Fs
Then using exponential interpolation to find inertia, average the inertia for each expeirment cycle.
exponential interpolation w(t)= \frac{tau_in-Fs}{b} \cdot (1-e^{\frac{-bt}{J}})
"""


import os
import robstride
import numpy as np 
import math
import csv
import statistics
import re

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
OUTPUT_FILE = "motor_parameter_estimate.csv"
MIN_SAMPLES_PER_CYCLE = 10
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




#finding the location where alpha close to 0 and recording that omega at that position. Doing the linear regression
def bF_calculation(omega_static: list, tau_in: list):
    omega_plot = np.array(omega_static)
    tau_plot = np.array(tau_in)
    design = np.column_stack((omega_plot, np.sign(omega_plot)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, tau_in, rcond=None)
    if rank < 2:
        raise ValueError("Need the steady state measurement at different speed magnitude")
    b =float(coefficients[0])
    F_s = float(coefficients[1])
    return b,F_s

def exponential_interpolation(samples, b, F_s):
    """Fit one constant-torque cycle and return inertia J in kg*m^2.

    samples contains (time_s, omega_rad_s, torque_input) tuples.
    b is the positive viscous coefficient. F_s is the signed intercept
    from tau = b * omega + F_s, fitted for this rotation direction.

    Fit J directly in omega(t) = ((tau_in - F_s) / b) * (1 - exp(-b*t/J)).
    This assumes zero speed at torque application. Sample times must be
    elapsed seconds since torque application, not since the first reading.
    """
    #this is the fitting function we used
    from scipy.optimize import curve_fit
    #some checking towards the data
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 3:
        raise ValueError("Provide at least three (time, omega, torque) samples")
    if not np.isfinite(b) or b <= 0 or not np.isfinite(F_s):
        raise ValueError("b must be positive and finite; F_s must be finite")
    #extracted the data in the column of time, omega and tau_in
    time_s = data[:, 0]
    omega = data[:, 1]
    tau_in = data[0, 2]
    if np.any(time_s < 0) or np.any(np.diff(time_s) <= 0):
        raise ValueError("Sample times must be nonnegative and strictly increasing")
    if not np.allclose(data[:, 2], tau_in, rtol=1e-6, atol=1e-9):
        raise ValueError("The exponential model requires constant input torque")
    
    #infinite omega value
    omega_inf = (tau_in - F_s) / b
    if abs(omega_inf) <= 1e-12 or np.ptp(omega) <= 1e-12:
        raise ValueError("A changing-speed transient is needed to estimate J")
    if omega[-1] * omega_inf <= 0:
        raise ValueError("Recorded rotation disagrees with the predicted steady speed")
    #This is our equation
    def exponential_response(t, J):
        return ((tau_in - F_s) / b) * (1 - np.exp(-b * t / J))

    # One time constant reaches 1 - exp(-1), about 63.2% of the speed change.
    progress = omega / omega_inf
    #finding whether there is taht value
    reached = np.flatnonzero((time_s > 0) & (progress >= 1 - np.exp(-1)))
    #if that value exist, then program using that 63.2% time to take the first guess, if that doesn't exit, then using the last recorded time in the sample.
    if reached.size:
        initial_time_constant = time_s[reached[0]]
    else:
        initial_time_constant = time_s[-1]
    initial_J = b * initial_time_constant

    #fitted the curve to find the actual J. 
    fitted, _ = curve_fit(
        exponential_response,
        time_s,
        omega,
        p0=[initial_J],
        bounds=(0.0, np.inf),
        maxfev=10000,
    )
    return float(fitted[0])



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
    
    omega = []
    tau_in = []
    used_files=[]
    for path in input_paths:
        samples = read_experiment_csv(path)
        if len(samples) >= MIN_SAMPLES_PER_CYCLE:
            omega.append(samples[-1][1])
            tau_in.append(samples[-1][2])
            used_files.append(os.path.basename(path))
        else:
            print(f"Skipping{os.path.basename(path)}: too few valid samples")

    b,Fc = bF_calculation(omega,tau_in)
    direction = np.sign(samples[-1][1])
    Fs = Fc * direction
    J_inertia =[]
    for path in input_paths:
        samples = read_experiment_csv(path)
        if len(samples)>= MIN_SAMPLES_PER_CYCLE:
            J_inertia.append(exponential_interpolation(samples,b,Fs))

        else:
            print(f"Skipping{os.path.basename(path)}:too few valid samples")
    try:
        average_J = sum(J_inertia)/len(J_inertia)
    except ZeroDivisionError:
        print(f"there is no valid data generated by J_inertia:{J_inertia}")
    
    print(f"Inertia J is:{average_J}")
    print(f"Viscous frictionb:{b}")
    print(f"Coulomb friction torque:{Fs}")


if __name__ == "__main__":
    main()

