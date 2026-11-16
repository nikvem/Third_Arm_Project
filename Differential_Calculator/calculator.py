# Initial differential calculator for J5 & J6
# Reference: 
# Zero is the shaft straight with the forearm
# Positive pitch is up 
# Positive roll is CCW (looking at wrist head-on)

# NOTE: Positive torque on motor makes them move CW

import math
import numpy as np

PULLEY_RATIO = 2 


def calculate(roll, pitch, roll_torque, pitch_torque):
    # Angles are in degrees; torques are in N·m.
    if not all(math.isfinite(x) for x in (roll, pitch, roll_torque, pitch_torque)):
        raise ValueError("Enter finite numbers only.")

    if not -90 <= roll <= 90 or not -90 <= pitch <= 90:
        raise ValueError("Roll and pitch must each be between -90 and 90 degrees.")

    # Convert the wrist angles into motor angles, including the pulley ratio
    j5_angle = PULLEY_RATIO * (roll + pitch)
    j6_angle = PULLEY_RATIO * (roll - pitch)

    # Solving for torques
    A = np.array([[1, 0, -1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 1, -1], 
                  [0, 1, 0, -1]])
    b = np.array([pitch_torque, roll_torque, 0, 0])
    torques = np.linalg.solve(A, b) 
    print(torques)
    j5_torque = torques[0] + torques[1]
    j6_torque = -torques[2] + torques[3]

    if (j5_torque > 5.5 or j6_torque > 5.5):
        raise ValueError("Exceeded motor torque threshold")
    
    return j5_angle, j6_angle, j5_torque, j6_torque


if __name__ == "__main__":
    print("Wrist differential calculator")

    try:
        roll = float(input("Input roll angle (degrees): "))
        pitch = float(input("Input pitch angle (degrees): "))
        roll_torque = float(input("Input roll torque (N·m): "))
        pitch_torque = float(input("Input pitch torque (N·m): "))

        a5, a6, t5, t6 = calculate(roll, pitch, roll_torque, pitch_torque)

        print(f"\nJ5: angle = {a5:.2f} degrees, torque = {t5:.2f} N·m")
        print(f"J6: angle = {a6:.2f} degrees, torque = {t6:.2f} N·m")

    except (ValueError, EOFError) as error:
        print(f"Input error: {error}")
        raise SystemExit(1)