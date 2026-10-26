import can
import robstride
import time
import os
import contextlib
import numpy as np
import pandas as pd
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

def main(): 
    cur_cmd = torque_cmd / motorKT
 
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
            while True:
                time.sleep(timestep)
                t = time.perf_counter() - start_time    # timestamp
                w = rs_client.read_param(motorID, 'mechvel') # omega
                dt = t - t_prev
                wdot = wdot_prev * c_LPF + (w - w_prev) / dt # alpha

                # Logging
                t_log.append(t)
                w_log.append(w)

                # Stopping Conditions
                if wdot < stop_threshold:
                    break
                
                if (t > max_runtime):
                    print("System did not converge")
                    break

                wdot_prev = wdot # Updating previous wdot for next wdot calculation

        # Stop torque and deactivate the motor
        rs_client.write_param(motorID, 'iq_ref', 0.0)
        rs_client.disable(motorID)

        # Plot omega vs time (runs after the motor has been disabled)
        # FTX FROM HERE ON !
        if len(t_log) > 0:
            plt.figure()
            plt.plot(t_log, w_log)
            plt.xlabel('Time (s)')
            plt.ylabel('Omega (rad/s)')
            plt.title(f'Step response, tau = {torque_cmd} Nm')
            plt.grid(True)
            plt.show()

        