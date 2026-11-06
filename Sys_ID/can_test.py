import can
import time
import struct

def mac_robstride_scan():
    print("Opening Jhoinrch adapter on macOS...")
    try:
        # Swap 'socketcan' from the script with 'gs_usb' for your Mac hardware
        bus = can.interface.Bus(interface='socketcan', channel="can0", bitrate=1000000)
    except Exception as e:
        print(f"Failed to open adapter: {e}")
        return

    # From the script: CommunicationType.READ_PARAMETER or similar probe (usually type 0x00 or 0x11)
    # Let's test the official 0x00 GET_DEVICE_ID with the correct 0xFF Host ID mapping
    COMM_GET_DEVICE_ID = 0x00
    HOST_ID = 0xFF             # CRITICAL: Must be 0xFF as per script constraint

    print("Scanning IDs 1 to 150 using the script's exact bit-packing logic...")
    print("Press Ctrl+C to stop.")

    for device_id in range(1, 151):
        # The exact formula from your script's transmit function:
        ext_id = (COMM_GET_DEVICE_ID << 24) | (HOST_ID << 8) | device_id

        # Standard empty 8-byte payload
        frame = can.Message(
            arbitration_id=ext_id,
            is_extended_id=True,
            dlc=8,
            data=[0x00] * 8
        )

        try:
            bus.send(frame)
            time.sleep(0.01) # Rapid scan pace matching their tqdm loop

            # Catch responses
            rx_msg = bus.recv(timeout=0.02)
            if rx_msg is not None:
                rx_id = rx_msg.arbitration_id
                # Deconstruct using the script's layout rules
                resp_type = (rx_id >> 24) & 0x1F
                resp_device = rx_id & 0xFF

                print(f"\nFOUND MOTOR AT ID: {device_id}!")
                print(f"Raw Response Extended ID: {hex(rx_id)}")
                print(f"Responding Device ID field: {resp_device}")
                print(f"Payload Data (MCU UID): {rx_msg.data.hex()}")
                bus.shutdown()
                return

        except Exception as e:
            print(f"\nError scanning ID {device_id}: {e}")
            bus.shutdown()
            return

    print("\nScan completed. No devices acknowledged.")
    bus.shutdown()

if __name__ == "__main__":
    mac_robstride_scan()
