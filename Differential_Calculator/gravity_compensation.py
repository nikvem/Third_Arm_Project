import os
import sys
import numpy as np
import math
import calculator as cal
import time
import can
import robstride

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



mass = 0.5 #kg
d_roll = 0.1 #desired roll in radian
d_pitch = 0.1#desired pitch in radian
length = 0.1
kpp = 0.1
kdp = 0.1
kpr = 0.1
kdr = 0.1



def read_motor_angles(rs_client, j5_id=J5_ID, j6_id=J6_ID):
    # reading j5 and j6 angle from the motor and return the angle in rad.
    j5_angle = float(rs_client.read_param(j5_id, "mechpos"))
    j6_angle = float(rs_client.read_param(j6_id, "mechpos"))
    if not all(math.isfinite(x) for x in (j5_angle, j6_angle)):
        raise ValueError("Motor returned a non-finite angle")
    return j5_angle - J5_ZERO_RAD, j6_angle - J6_ZERO_RAD

def motor_angle_to_pitch_and_roll(j5_angle, j6_angle):
    #convert motor andgles to angle of pitch and roll
    roll_angle = (j5_angle +j6_angle)/(2*cal.PULLEY_RATIO)
    pitch_angle = (j5_angle - j6_angle)/(2*cal.PULLEY_RATIO)
    return roll_angle, pitch_angle



def main():
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
        angle_tolerance = math.radians(1.0)   # Within 1 degree
        speed_tolerance = math.radians(2.0)   # Slower than 2 degrees/second
        settle_since = None
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

                
            #core iteration here
            while True:
                j5_angle, j6_angle = read_motor_angles(rs_client)
                now = time.monotonic()
                roll_angle, pitch_angle = motor_angle_to_pitch_and_roll(j5_angle,j6_angle)
                if previous_time is None:
                    pitch_velocity = 0.0
                    roll_velocity = 0.0
                else:
                    dt = now - previous_time
                    pitch_velocity = (pitch_angle - previous_pitch) / dt
                    roll_velocity = (roll_angle - previous_roll) / dt
                tau_pitch = mass* 9.81 * length *np.cos(pitch_angle + FOREARM_ELEVATION_RAD) + kpp*(d_pitch - pitch_angle) - kdp*pitch_velocity
                tau_roll = kpr * (d_roll - roll_angle) - kdr * roll_velocity
                previous_pitch = pitch_angle
                previous_time = now
                previous_roll = roll_angle

                #converting back to j5_torque and j6_torque
                _, _, j5_torque, j6_torque = cal.calculate(roll_angle, pitch_angle, tau_roll, tau_pitch)
                current5 = j5_torque / MOTOR_KT
                current6 = j6_torque / MOTOR_KT
                if not all(math.isfinite(x) for x in (current5, current6)):
                    raise ValueError("Non-finite motor current")
                if max(abs(current5), abs(current6)) > CURRENT_LIMIT:
                    raise ValueError("Exceeded motor current threshold")
                rs_client.write_param(J5_ID, 'iq_ref', current5)
                rs_client.write_param(J6_ID, 'iq_ref', current6)

                at_target = (
                    abs(d_pitch - pitch_angle) < angle_tolerance
                    and abs(d_roll - roll_angle) < angle_tolerance
                    and abs(pitch_velocity) < speed_tolerance
                    and abs(roll_velocity) < speed_tolerance
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
        
if __name__ == "__main__":
    main()
