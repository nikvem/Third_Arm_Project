"""
bringup.py - first-run helper for the forearm controller with the official RobStride SDK.

    python bringup.py scan                   # list motor CAN IDs on can0 (replaces can_test.py)
    python bringup.py zero                   # wrist sitting in the zeroing jig -> writes calibration.json
    python bringup.py hold ROLL_DEG PITCH_DEG   # gravity-comp PD hold (Ctrl+C = go limp)

Bring up CAN first:  sudo ip link set can0 up type can bitrate 1000000
Edit CHANNEL / J5_ID / J6_ID below. Test with two people, a hand on the power switch.
"""
import json, math, sys, time
from forearm_gravity_comp import (DifferentialCalculatorAdapter, ControllerConfig,
                                  ForearmGravityCompController, HandGravityParams,
                                  RobstrideSDKBus, WristGravityModel)

CHANNEL, J5_ID, J6_ID, MODEL = "can0", 1, 2, "rs-05"      # TODO: set from `scan`
CAL_FILE = "calibration.json"
MODE = "mit"                    # "mit": drive runs PD + torque in N*m (recommended). "current": host PD -> current
KP, KD = (3.0, 3.0), (0.15, 0.15)                          # start low, raise slowly
FIRST_RUN_TAU_MAX = (0.5, 0.5)   # N*m. Gravity holding needs ~0.1-0.2. Keep small until torque_check.py
#                                  has confirmed the torque<->current scale (raise towards 2.5 afterwards).


def load_cal():
    try:
        return json.load(open(CAL_FILE))
    except FileNotFoundError:
        return None


def scan():
    from robstride_dynamics import RobstrideBus
    print(RobstrideBus.scan_channel(CHANNEL))


def zero():
    """Wrist in the jig = roll 0, pitch 0. Record each motor's raw position as its homing offset."""
    from robstride_dynamics import ParameterType
    bus = RobstrideSDKBus(CHANNEL, J5_ID, J6_ID, MODEL, calibration=None)
    bus.sdk.connect()
    try:
        cal = {}
        for n in bus.NAMES:
            raw = bus.sdk.read(n, ParameterType.MECHANICAL_POSITION)
            cal[n] = {"direction": 1, "homing_offset": raw}      # user = raw - offset  ->  0 in the jig
            print(f"{n}: raw position {raw:+.4f} rad -> homing_offset")
        json.dump(cal, open(CAL_FILE, "w"), indent=2)
        print(f"saved {CAL_FILE}. Re-run `hold` and check both joints read ~0 in the jig.")
    finally:
        bus.sdk.disconnect(disable_torque=False)


def hold(roll_deg, pitch_deg):
    model = DifferentialCalculatorAdapter(WristGravityModel(HandGravityParams()))   # mass = 0: pure PD
    cfg = ControllerConfig(kp=KP, kd=KD, tau_max=FIRST_RUN_TAU_MAX, onboard_pd=(MODE == "mit"))
    with RobstrideSDKBus(CHANNEL, J5_ID, J6_ID, MODEL, calibration=load_cal(), tau_limit=FIRST_RUN_TAU_MAX[0], mode=MODE) as bus:
        ctrl = ForearmGravityCompController(model, bus, cfg)
        ctrl.set_target(math.radians(roll_deg), math.radians(pitch_deg))
        print("holding; Ctrl+C to stop")
        try:
            ctrl.run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "scan": scan()
    elif cmd == "zero": zero()
    elif cmd == "hold" and len(sys.argv) == 4: hold(float(sys.argv[2]), float(sys.argv[3]))
    else: print(__doc__)
