
import can
import math
import robstride
import time
import matplotlib.pyplot as plt


####################################
 
# If the CAN module is disconnected, please follow the following steps to reconnect:
# 1. Plug in the CAN module.
# 2. Run the can_test.py script to check the motor IDs.
# 3. sudo ip link set can0 down
# 4. sudo ip link set can0 up type can bitrate 1000000
 
####################################
 
 
############ INPUTS ############
 
J5_ID = 5                   # CAN ID of J5
J6_ID = 6                   # CAN ID of J6
ROLL_DESIRED = 20           # <-- EDIT THIS between runs. Desired roll angle, degrees
PITCH_DESIRED = 10          # <-- EDIT THIS between runs. Desired pitch angle, degrees
WEIGHT = 0.5                  # <-- EDIT THIS between runs. Weight of the robot, kg
 
#################################
 
def main():
    cur_5_amp = J5_TORQUE_AMP / motorKT
    cur_6_amp = J6_TORQUE_AMP / motorKT
 
    bus_cm = can.Bus(
        interface="socketcan",
        channel="can0",
        bitrate = 1000000
    )
    with bus_cm as bus:
        rs_client = robstride.Client(
            bus,
            host_can_id=0xAA
        )
        # Enable motors and zero motors
        rs_client.write_param(J5_ID, 'run_mode', 3)
        rs_client.bus.send(rs_client._rs_msg(robstride.client.MotorMsg.ZeroPos, rs_client.host_can_id, J5_ID, [1, 0, 0, 0, 0, 0, 0, 0]))
        rs_client.bus.recv(rs_client.recv_timeout) # consume the motor's reply so it doesn't confuse the next read/write
 
        rs_client.write_param(J6_ID, 'run_mode', 3)
        rs_client.bus.send(rs_client._rs_msg(robstride.client.MotorMsg.ZeroPos, rs_client.host_can_id, J6_ID, [1, 0, 0, 0, 0, 0, 0, 0]))
        rs_client.bus.recv(rs_client.recv_timeout) # consume the motor's reply so it doesn't confuse the next read/write
 
 
        #################################################
        ############# MAIN RUNNING SEQUENCE #############
        #################################################
 
        try: 
            rs_client.enable(J5_ID)
            rs_client.enable(J6_ID)
            time.sleep(1)
            # Commanding sine torque to both motors
            print("----------------- Test Starting -----------------")
            print(f"Commanding sine torque: amplitude = {J5_TORQUE_AMP} Nm (Current = {cur_5_amp:.4f} A), {FREQ_HZ} Hz  [J5]")
            print(f"Commanding sine torque: amplitude = {J6_TORQUE_AMP} Nm (Current = {cur_6_amp:.4f} A), {FREQ_HZ} Hz  [J6]")
 
            t0 = time.monotonic()
            next_print = 0.0
            pos_5_log = []
            pos_6_log = []
            times = []
            while True:
                t = time.monotonic() - t0
                s = math.sin(2 * math.pi * FREQ_HZ * t)
                rs_client.write_param(J5_ID, 'iq_ref', cur_5_amp * s)
                rs_client.write_param(J6_ID, 'iq_ref', cur_6_amp * s)
 
                if t >= next_print:
                    pos_5 = rs_client.read_param(J5_ID, 'mechpos')
                    pos_6 = rs_client.read_param(J6_ID, 'mechpos')
                    print(f"J5: {pos_5:.4f} rad | J6: {pos_6:.4f} rad")
                    next_print = t + PRINT_DT
                    
                    pos_5_log.append(pos_5)
                    pos_6_log.append(pos_6)
                    times.append(t)

                time.sleep(UPDATE_DT)
 
        # Keyboard interrupt from user
        except KeyboardInterrupt:
            print("Keyboard Interrupt: Stopping Motor Torque")
 
        # Disable motor
        finally:
            rs_client.write_param(J5_ID, 'iq_ref', 0.0)
            rs_client.disable(J5_ID)
            rs_client.write_param(J6_ID, 'iq_ref', 0.0)
            rs_client.disable(J6_ID)
    
    
    ############# PLOTTING #############
 
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(times, pos_5_log, label="J5")
    ax1.plot(times, pos_6_log, label="J6")
    ax1.set_ylabel("Position (rad)")
    ax1.legend()
    ax2.plot(times, [J5_TORQUE_AMP * math.sin(2 * math.pi * FREQ_HZ * x) for x in times], label="J5")
    ax2.plot(times, [J6_TORQUE_AMP * math.sin(2 * math.pi * FREQ_HZ * x) for x in times], label="J6")
    ax2.set_ylabel("Commanded torque (Nm)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    plt.show()
 
 
if __name__ == "__main__":
    main()
