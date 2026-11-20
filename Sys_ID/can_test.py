
import can
import sys
import time
import struct
 
# ---- Config ---------------------------------------------------------------
INTERFACE = 'socketcan'   # on macOS with the Jhoinrch adapter, use 'gs_usb' instead
CHANNEL = 'can0'
BITRATE = 1000000
 
HOST_ID = 0xFF            # Must be 0xFF per the RobStride script constraint
SCAN_IDS = range(0, 254)  # 0..253 (0xFE / 0xFF are broadcast / host)
REPLY_WINDOW = 0.03       # seconds to listen for replies after each probe
SHOW_ALL_FRAMES = '--raw' in sys.argv   # run with --raw for diagnostics
 
COMM_GET_DEVICE_ID = 0x00
COMM_READ_PARAM = 0x11    # type 17
PARAM_MECH_POS = 0x7019   # mechPos (float, rad) -- verify index in your model's manual
# ---------------------------------------------------------------------------
 
 
def make_ext_id(comm_type, host_id, device_id):
    return (comm_type << 24) | (host_id << 8) | device_id
 
 
def describe(rx):
    kind = "ext" if rx.is_extended_id else "std"
    return f"{kind} id={hex(rx.arbitration_id)} data={bytes(rx.data).hex()}"
 
 
def sniff(bus, seconds=3.0):
    """Listen without transmitting, to see whether anything is talking at all."""
    print(f"Listening passively for {seconds:.0f}s (nothing is transmitted)...")
    deadline = time.monotonic() + seconds
    count = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        rx = bus.recv(timeout=remaining)
        if rx is not None:
            count += 1
            print(f"  heard: {describe(rx)}")
    if count == 0:
        print("  No traffic heard.")
    print()
 
 
def scan(bus):
    """Probe every ID and collect ALL responders: {motor_id: [uid_hex, ...]}."""
    found = {}
 
    for device_id in SCAN_IDS:
        ext_id = make_ext_id(COMM_GET_DEVICE_ID, HOST_ID, device_id)
        bus.send(can.Message(
            arbitration_id=ext_id,
            is_extended_id=True,
            dlc=8,
            data=[0x00] * 8,
        ))
 
        # Drain everything that arrives during the reply window.
        deadline = time.monotonic() + REPLY_WINDOW
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            rx = bus.recv(timeout=remaining)
            if rx is None:
                break
            if rx.is_extended_id and rx.arbitration_id == ext_id:
                continue  # our own echoed probe
 
            is_reply = (rx.is_extended_id and
                        ((rx.arbitration_id >> 24) & 0x1F) == COMM_GET_DEVICE_ID)
            if not is_reply:
                if SHOW_ALL_FRAMES:
                    print(f"  [other frame while probing {device_id}] {describe(rx)}")
                continue
 
            # Reply layout: bits 15..8 = motor CAN ID, low byte = reply target
            # (0xFE on your hardware), data = 8-byte MCU UID.
            motor_id = (rx.arbitration_id >> 8) & 0xFF
            uid = bytes(rx.data).hex()
 
            uids = found.setdefault(motor_id, [])
            if uid not in uids:
                if uids:
                    print(f"WARNING: id={motor_id} answered with a DIFFERENT UID "
                          f"({uid}) -- two motors share this CAN ID!")
                uids.append(uid)
                print(f"FOUND MOTOR  id={motor_id:<3} (0x{motor_id:02X})  "
                      f"UID={uid}  raw ext id={hex(rx.arbitration_id)}  "
                      f"[probed {device_id}]")
 
    return found
 
 
def read_position(bus, motor_id, timeout=0.05):
    """Read mechPos (rad) from one motor without enabling it. None on failure."""
    ext_id = make_ext_id(COMM_READ_PARAM, HOST_ID, motor_id)
    data = list(struct.pack('<H', PARAM_MECH_POS)) + [0x00] * 6
    bus.send(can.Message(arbitration_id=ext_id, is_extended_id=True, dlc=8, data=data))
 
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        rx = bus.recv(timeout=remaining)
        if rx is None:
            return None
        if not rx.is_extended_id or rx.arbitration_id == ext_id:
            continue
        rx_type = (rx.arbitration_id >> 24) & 0x1F
        rx_motor = (rx.arbitration_id >> 8) & 0xFF
        if rx_type == COMM_READ_PARAM and rx_motor == motor_id and len(rx.data) == 8:
            index = struct.unpack('<H', bytes(rx.data[0:2]))[0]
            if index == PARAM_MECH_POS:
                return struct.unpack('<f', bytes(rx.data[4:8]))[0]
 
 
def identify_by_hand(bus, motor_ids):
    """Live position readout. Turn one motor by hand; the ID that changes is it."""
    print("\nIdentify mode: rotate ONE motor shaft by hand and watch which ID changes.")
    print("Press Ctrl+C to stop.\n")
 
    try:
        while True:
            parts = []
            for mid in motor_ids:
                pos = read_position(bus, mid)
                if pos is None:
                    parts.append(f"ID {mid}: no reply")
                    continue
                parts.append(f"ID {mid}: {pos:8.3f} rad")
            print(" | ".join(parts))
    except KeyboardInterrupt:
        print("\nStopped.")
 
 
def main():
    print("Opening adapter...")
    try:
        bus = can.interface.Bus(interface=INTERFACE, channel=CHANNEL, bitrate=BITRATE)
    except Exception as e:
        print(f"Failed to open adapter: {e}")
        return
 
    try:
        if SHOW_ALL_FRAMES:
            sniff(bus)
 
        print(f"Scanning IDs {SCAN_IDS.start} to {SCAN_IDS.stop - 1}...")
        found = scan(bus)
 
        print("\n--- Scan complete ---")
        if not found:
            print("No devices acknowledged.")
            return
 
        print(f"{len(found)} distinct CAN ID(s) found:")
        for motor_id, uids in sorted(found.items()):
            for uid in uids:
                print(f"  CAN ID {motor_id:<3} (0x{motor_id:02X})   UID {uid}")
 
        answer = input("\nRun hand-rotate identification? [y/N] ").strip().lower()
        if answer == 'y':
            identify_by_hand(bus, sorted(found))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        bus.shutdown()
 
 
if __name__ == "__main__":
    main()
 