import can
import robstride
import time
import os
import re
import csv
import matplotlib
matplotlib.use('Agg')  # headless-safe; switch backends if running with a display
import matplotlib.pyplot as plt

####################################
####################################


############ INPUTS ############

motorID = 1             # CAN ID of motor
motorKT = 1.22          # Torque constant, Nm/Arms (1.22 for RS01/RS02, 2.36 for RS03)
timestep = 0.02          # Loop period, s

torque_cmd = 0.3          # <-- EDIT THIS between runs. Constant commanded torque, Nm (can be negative)

start_threshold = 0.01     # |omega| above which logging begins
stop_threshold = 0.01      # Filtered |omega_dot| below which motion is considered saturated
consistency_count = 15     # # consecutive saturated samples required before stopping

max_runtime = 15.0         # Safety timeout (s), measured from when motion starts
no_motion_timeout = 10.0   # Safety timeout (s) if the motor never starts moving at all

c_LPF = 0.3                # LPF filtering coefficient for estimating wdot
#################################


def next_cycle_number(result_dir):
    """Return the next cycle number based on plots already in result_dir."""
    pattern = re.compile(r"excitation_cycle_(\d+)_torque_.*\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    existing_cycles = []
    if os.path.isdir(result_dir):
        for filename in os.listdir(result_dir):
            match = pattern.match(filename)
            if match:
                existing_cycles.append(int(match.group(1)))
    return max(existing_cycles, default=0) + 1


def main():
    cur_cmd = torque_cmd / motorKT
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(result_dir, exist_ok=True)
    cycle_number = next_cycle_number(result_dir)

    bus_cm = can.Bus()

    with bus_cm as bus:
        rs_client = None

        # Logging arrays
        t_log = [] # time logging
        w_log = [] # omega logging

        rs_client = robstride.Client(bus)

        # Enable and zero motor
        rs_client.disable(motorID)
        rs_client.zero_pos(motorID)
        rs_client.write_param(motorID, 'run_mode', robstride.RunMode.Current)
        rs_client.enable(motorID)
        time.sleep(1)

        # Commanding torque to input motor
        print("----------------- Test Starting -----------------")
        print(f"Commanding constant torque = {torque_cmd} Nm (Current = {cur_cmd:.4f} A)")
        rs_client.write_param(motorID, 'iq_ref', cur_cmd)

        # Start logging when omega nonzero (threshold for filtering noise)
        moved = False
        wait_start = time.perf_counter()
        while True:
            w = rs_client.read_param(motorID, 'mechvel')
            if abs(w) > start_threshold:  
                moved = True
                break
            if time.perf_counter() - wait_start > no_motion_timeout:
                print("No motion detected")
                break
            time.sleep(timestep)

        # Log when start moving to reduce delay
        if moved:
            start_time = time.perf_counter()  
            t_log.append(0.0)
            w_log.append(w)

            w_prev = w
            wdot_prev = 0
            t_prev = 0.0
            saturated_samples = 0
            while True:
                time.sleep(timestep)
                t = time.perf_counter() - start_time    # timestamp
                w = rs_client.read_param(motorID, 'mechvel') # omega
                dt = t-t_prev
                raw_wdot = (w - w_prev) / dt
                wdot = c_LPF * wdot_prev + (1 - c_LPF) * raw_wdot #filtered wdot for calculating the rate change, to be below the threshold to consider as saturated. 
                # Logging
                t_log.append(t)
                w_log.append(w)
                if (t > max_runtime):
                    print("System did not converge")
                    print("running exceed the threshold time")
                    break
                # Require consecutive samples of near-zero angular acceleration.
                if abs(wdot) < stop_threshold:
                    saturated_samples += 1
                else:
                    saturated_samples = 0

                if saturated_samples >= consistency_count:
                    break

                w_prev = w
                wdot_prev = wdot
                t_prev = t

        # Stop torque and deactivate the motor
        rs_client.write_param(motorID, 'iq_ref', 0.0)
        rs_client.disable(motorID)

        # Save one matching plot and CSV for this experiment cycle.
        torque_label = (
            f"{torque_cmd:+.3f}"
            .replace("+", "plus")
            .replace("-", "minus")
            .replace(".", "p")
        )
        file_stem = f"excitation_cycle_{cycle_number:03d}_torque_{torque_label}Nm"
        plot_path = os.path.join(result_dir, f"{file_stem}.jpg")
        csv_path = os.path.join(result_dir, f"{file_stem}.csv")

        fig, ax = plt.subplots()
        ax.plot(t_log, w_log)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Omega (rad/s)')
        ax.set_title(f'Excitation cycle {cycle_number}, torque = {torque_cmd:g} Nm')
        ax.grid(True)
        fig.savefig(plot_path, format="jpg", dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        # csv file store accompany with jpg plot exist 

        with open(csv_path, "w", newline = "", encoding = "UTF-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["cycle", "time_s", "omega_rad_s", "torque_input"],
            )
            writer.writeheader()
            for t,omega in zip(t_log, w_log):
                writer.writerow({
                    "cycle": cycle_number,
                    "time_s": t,
                    "omega_rad_s": omega,
                    "torque_input": torque_cmd,
                })
        print(f"saved plot: {plot_path}")
        print(f"Saved data path:{csv_path}")
        if not t_log:
            print("No motion for csv data storing")
if __name__ == "__main__":
    main()

        
