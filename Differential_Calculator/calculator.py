#Initial differential calculator for J5 & J6
import math

J5_SIGN = 1
J6_SIGN = 1

J5_ZERO = 0
J6_ZERO = 0

def calculate(roll, pitch, roll_torque, pitch_torque):
    # Note: All angles are in degrees and torques are in N·m.
    if not all(math.isfinite(x) for x in (roll, pitch, roll_torque, pitch_torque)):
        raise ValueError("Enter finite numbers only.")

    if not -90 <= roll <= 90 or not -90 <= pitch <= 90:
        raise ValueError("Roll and pitch must each be between -90 and 90 degrees.")

    if J5_SIGN not in (-1, 1) or J6_SIGN not in (-1, 1):
        raise ValueError("Motor signs must be 1 or -1.")

    # Convert the wrist angles into motor angles.
    j5_angle = J5_ZERO + J5_SIGN * (roll + pitch)
    j6_angle = J6_ZERO + J6_SIGN * (roll - pitch)

    # Split the wrist torques between the two motors.
    j5_torque = J5_SIGN * (roll_torque + pitch_torque) / 2
    j6_torque = J6_SIGN * (roll_torque - pitch_torque) / 2

    return j5_angle, j6_angle, j5_torque, j6_torque


if __name__ == "__main__":
    print("Wrist differential calculator")

    try:
        roll = float(input("Input Roll angle (degrees): "))
        pitch = float(input("Input Pitch angle (degrees): "))
        roll_torque = float(input("Input Roll torque (N·m): "))
        pitch_torque = float(input("Input Pitch torque (N·m): "))

        a5, a6, t5, t6 = calculate(roll, pitch, roll_torque, pitch_torque)

        print(f"\nJ5: angle = {a5:.2f} degrees, torque = {t5:.2f} N·m")
        print(f"J6: angle = {a6:.2f} degrees, torque = {t6:.2f} N·m")

    except (ValueError, EOFError) as error:
        print(f"Input error: {error}")
        raise SystemExit(1)