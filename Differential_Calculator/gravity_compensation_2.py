import os
import sys
import numpy as np
import math
import calculator as cal
import time
import can
import robstride
from datetime import datetime
from pathlib import Path

####################################
 
# If the CAN module is disconnected, please follow the following steps to reconnect:
# 1. Plug in the CAN module.
# 2. Run the can_test.py script to check the motor IDs.
# 3. sudo ip link set can0 down
# 4. sudo ip link set can0 up type can bitrate 1000000

####################################

"""
angle reading directly from j5 motor and j6 motor used
PD controller for the angle holding
"""
INTERFACE = "socketcan"
CHANNEL = "can0"
BITRATE = 1_000_000
HOST_CAN_ID = 0xAA
J5_ID = 5
J6_ID = 6
MOTOR_KT = 0.94 #motor current constant
CURRENT_LIMIT = cal.MOTOR_TORQUE_LIMIT / MOTOR_KT  # A; existing torque cap converted to current.
# Set these to encoder readings at physical roll=pitch=0 (hand straight).
# Zero defaults assume that the motors were already calibrated to that pose.
J5_ZERO_RAD = 0.0
J6_ZERO_RAD = 0.0
# Fixed forearm elevation above horizontal; positive pitch raises the hand.
# This simple gravity model assumes the pitch motion is in a vertical plane
# and the load COM is on the hand axis, distance `length` from the pitch joint.
FOREARM_ELEVATION_RAD = 0.0
READ_INTERVAL = 0.1  # Seconds between pairs of reads.
PLOT_DIRECTORY = Path(__file__).resolve().parent / "pd_tuning_results"



mass = 0.8 # kg
d_roll = 0 # desired roll in radian
d_pitch = np.pi/4 # desired pitch in radian
length = 6.25 * 0.0254 # m
Kp5 = 0.5
Kd5 = 0.1
Kp6 = 0.1
Kd6 = 0.1



def read_motor_angles(rs_client, j5_id=J5_ID, j6_id=J6_ID):
    # reading j5 and j6 angle from the motor and return the angle in rad.
    j5_angle = float(rs_client.read_param(j5_id, "mechpos"))
    j6_angle = float(rs_client.read_param(j6_id, "mechpos"))
    if not all(math.isfinite(x) for x in (j5_angle, j6_angle)):
        raise ValueError("Motor returned a non-finite angle")
    return j5_angle - J5_ZERO_RAD, j6_angle - J6_ZERO_RAD

def motor_angle_to_pitch_and_roll(j5_angle, j6_angle):
    #convert motor angles to angle of pitch and roll
    roll_angle = (j5_angle +j6_angle)/(2*cal.PULLEY_RATIO)
    pitch_angle = (j5_angle - j6_angle)/(2*cal.PULLEY_RATIO)
    return roll_angle, pitch_angle



def plot_tuning_run(samples):
    """Save and display the plot after the CAN bus has closed.

    Torque is the commanded current multiplied by MOTOR_KT, not measured
    torque. A NaN marks a command that did not complete in that iteration.
    """
    if not samples:
        print("No PD samples recorded; nothing to plot.")
        return

    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    stem = "pd_tuning_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Import only after motor shutdown.
    import matplotlib.pyplot as plt

    times = [sample["time_s"] for sample in samples]
    fig, (torque_ax, angle_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7), constrained_layout=True
    )
    try:
        for motor, color in (("j5", "tab:blue"), ("j6", "tab:orange")):
            torque_ax.plot(times, [s[f"{motor}_commanded_torque_nm"] for s in samples],
                           label=f"{motor.upper()} commanded", color=color,
                           marker=".", markersize=3)
        torque_ax.set_ylabel("Torque (N·m)")
        torque_ax.set_title("PD tuning — recorded commands and angles")
        torque_ax.legend()
        torque_ax.grid(True, alpha=0.3)

        for joint, color in (("pitch", "tab:blue"), ("roll", "tab:orange")):
            angle_ax.plot(times, [math.degrees(s[f"{joint}_rad"]) for s in samples],
                          label=f"{joint.capitalize()} measured", color=color,
                          marker=".", markersize=3)
            angle_ax.plot(times, [math.degrees(s[f"desired_{joint}_rad"]) for s in samples],
                          label=f"{joint.capitalize()} desired", color=color, linestyle="--")
        angle_ax.set_ylabel("Angle (deg)")
        angle_ax.set_xlabel("Time since first sample (s)")
        angle_ax.legend()
        angle_ax.grid(True, alpha=0.3)
        png_path = PLOT_DIRECTORY / f"{stem}.png"
        fig.savefig(png_path, dpi=160)
        print(f"Saved PD plot: {png_path}")
        plt.show()
    finally:
        plt.close(fig)


def run_controller(samples):
    if not math.isfinite(MOTOR_KT) or MOTOR_KT <= 0:
        raise ValueError("MOTOR_KT must be finite and positive")
    if not math.isfinite(CURRENT_LIMIT) or CURRENT_LIMIT <= 0:
        raise ValueError("CURRENT_LIMIT must be finite and positive")
    with can.Bus(
                interface=INTERFACE, channel=CHANNEL, bitrate=BITRATE
            ) as bus:
        rs_client = robstride.Client(bus, host_can_id=HOST_CAN_ID)
        previous_pitch = None
        previous_time = None
        previous_roll = None
        #conditioned to break from while loop when the angle is within 1 degree and the speed is slower than 2 degree/s
        angle_tolerance = math.radians(1)   # Within 1 degree
        speed_tolerance = math.radians(2)   # Slower than 2 degree/second
        settle_since = None
        first_sample_time = None
        
        # Enable and zero motors
        rs_client.write_param(J5_ID, 'run_mode', 3)
        rs_client.bus.send(rs_client._rs_msg(robstride.client.MotorMsg.ZeroPos, rs_client.host_can_id, J5_ID, [1, 0, 0, 0, 0, 0, 0, 0]))
        rs_client.bus.recv(rs_client.recv_timeout) # consume the motor's reply so it doesn't confuse the next read/write
 
        rs_client.write_param(J6_ID, 'run_mode', 3)
        rs_client.bus.send(rs_client._rs_msg(robstride.client.MotorMsg.ZeroPos, rs_client.host_can_id, J6_ID, [1, 0, 0, 0, 0, 0, 0, 0]))
        rs_client.bus.recv(rs_client.recv_timeout) # consume the motor's reply so it doesn't confuse the next read/write



        try:
            # Configure both motors while disabled, without resetting encoder zeros.
            for motor_id in (J5_ID, J6_ID):
                rs_client.disable(motor_id)
            for motor_id in (J5_ID, J6_ID):
                rs_client.write_param(motor_id, 'run_mode', 3)
                rs_client.write_param(motor_id, 'iq_ref', 0.0)
            read_motor_angles(rs_client)  # Check feedback before enabling.
            for motor_id in (J5_ID, J6_ID):
                rs_client.enable(motor_id)

            J5_angle_d, J6_angle_d,_,_ = cal.calculate(d_roll, d_pitch, 0, 0)
                
            #core iteration here
            while True:
                j5_angle, j6_angle = read_motor_angles(rs_client)
                now = time.monotonic()
                roll_angle, pitch_angle = motor_angle_to_pitch_and_roll(j5_angle,j6_angle)
                if first_sample_time is None:
                    first_sample_time = now
                sample = {
                    "time_s": now - first_sample_time,
                    "j5_commanded_torque_nm": math.nan,
                    "j6_commanded_torque_nm": math.nan,
                    "pitch_rad": pitch_angle,
                    "roll_rad": roll_angle,
                    "desired_pitch_rad": d_pitch,
                    "desired_roll_rad": d_roll,
                }

                J5_vel = rs_client.read_param(J5_ID, 'mechvel')
                J6_vel = rs_client.read_param(J6_ID, 'mechvel')
                _, _, j5_torque, j6_torque = cal.calculate(roll_angle, pitch_angle, tau_roll, tau_pitch)
                J5_cur_pos = rs_client.read_param(J5_ID, 'mechpos')
                J6_cur_pos = rs_client.read_param(J5_ID, 'mechpos')

                # PD Controller on motors themselves
                j5_torque = j5_torque + Kp5*(J5_angle_d - J5_cur_pos) + Kd5*J5_vel
                j6_torque = j6_torque + Kp6*(J6_angle_d - J6_cur_pos) + Kd6*J6_vel
                
                current5 = j5_torque / MOTOR_KT
                current6 = j6_torque / MOTOR_KT
                print(f"current6:{current6}")
                if not all(math.isfinite(x) for x in (current5, current6)):
                    raise ValueError("Non-finite motor current")
                if max(abs(current5), abs(current6)) > CURRENT_LIMIT:
                    raise ValueError("Exceeded motor current threshold")
                rs_client.write_param(J5_ID, 'iq_ref', current5)
                sample["j5_commanded_torque_nm"] = current5 * MOTOR_KT
                rs_client.write_param(J6_ID, 'iq_ref', current6)
                sample["j6_commanded_torque_nm"] = current6 * MOTOR_KT


                # Check if the system is at the target angles and velocities

                at_target = (
                    abs(d_pitch - pitch_angle) < angle_tolerance
                    and abs(d_roll - roll_angle) < angle_tolerance
                )
                if previous_time is not None and at_target:
                    if settle_since is None:
                        settle_since = now
                    elif now-settle_since >= 0.5:
                        print("Target reached and settled for 0.5 seconds. Exiting.")
                        break
                else:
                    settle_since = None

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping controller.")
        finally:
            # Always attempt both motors, even if one CAN operation fails.
            # Disabling removes supporting torque: support the load on shutdown.
            for motor_id in (J5_ID, J6_ID):
                try:
                    rs_client.write_param(motor_id, 'iq_ref', 0.0)
                except Exception as error:
                    print(f"Motor {motor_id}: zero-current command failed: {error}", file=sys.stderr)
            for motor_id in (J5_ID, J6_ID):
                try:
                    rs_client.disable(motor_id)
                except Exception as error:
                    print(f"Motor {motor_id}: disable failed: {error}", file=sys.stderr)
        
def main():
    samples = []
    try:
        run_controller(samples)
    finally:
        # Runs on settled exit, Ctrl+C, and errors, after motor cleanup/bus close.
        # A plotting failure must not hide the original controller exception.
        try:
            plot_tuning_run(samples)
        except Exception as error:
            print(f"Could not finish PD plotting: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
