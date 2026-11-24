"""
forearm_gravity_comp.py

Gravity-compensation + PD hold controller for the forearm wrist differential (J5, J6).

    (roll, pitch)
        |  WristGravityModel : gravity torque about the roll and pitch axes
        v
    Calculate(roll, pitch, roll_torque, pitch_torque)
        |  -> j5_angle, j6_angle, j5_torque, j6_torque   (motor side, degrees / N*m)
        v
    tau_i = tau_ff_i + Kp_i*(q_des_i - q_i) + Kd_i*(qdot_des_i - qdot_i)

Default (cfg.onboard_pd = False): the PD runs in Python on the host and the result is
sent to each RS05 as a CURRENT command (RobstrideCurrentBus converts output torque -> current).
If you ever want the drive to run the PD itself, use RobstrideMITBus with cfg.onboard_pd = True.

Units inside this file: radians and N*m. Abhinay's script uses degrees, so the
adapter converts at the boundary.

TODO:
    1. HandGravityParams : mass and centre of mass of everything past the wrist.
    2. RobstrideCurrentBus: the send-current / read calls from the robstride library.
    3. ControllerConfig  : kp, kd (inputs, you choose), real joint limits.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np

import differential_calculator as dc          # differential calculator script, unmodified


# --------------------------------------------------------------------------- #
# Interfaces
# --------------------------------------------------------------------------- #
@dataclass
class JointTargets:
    """Motor-side targets for one (roll, pitch) request."""
    q5: float      # rad
    q6: float      # rad
    tau5: float    # N*m, torque that holds this pose against gravity
    tau6: float    # N*m


class DifferentialModel(Protocol):
    def solve(self, roll: float, pitch: float) -> JointTargets: ...


class MotorBus(Protocol):
    def read(self) -> Tuple[float, float, float, float]:
        """(q5, qdot5, q6, qdot6) in rad, rad/s (motor output shaft)."""
    def send_torque(self, tau5: float, tau6: float) -> None: ...
    def send_mit(self, q_des, v_des, kp, kd, tau_ff) -> None:
        """Operation-control frame: the drive runs the PD loop itself."""


# --------------------------------------------------------------------------- #
# Gravity model: what torque does gravity put on the roll and pitch axes?
# --------------------------------------------------------------------------- #
def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass
class HandGravityParams:
    mass: float = 0.0                             # kg. 0 = gravity feedforward OFF (pure PD hold).
    #                                               Set mass and com later to switch the feedforward on.
    com: Tuple[float, float, float] = (0.10, 0.0, 0.0)  # m  centre of mass from the wrist
    #                                               centre, in the hand frame (x along the shaft)
    forearm_incline: float = 0.0                  # rad. Angle of the forearm axis above horizontal.
    #                                               TODO: 0 assumes the forearm is horizontal; it
    #                                               really depends on J1-J4.
    roll_dir: float = -1.0                        # +roll is clockwise seen from the tip = rotation
    #                                               about -x. Flip to +1 if your hardware disagrees.
    g: float = 9.81


class WristGravityModel:
    """
    Frames: x along the forearm toward the tip, z up, y left (at zero pose).
    Hand pose = Ry(-pitch) @ Rx(roll_dir*roll)  (pitch about a fixed axis, roll about the hand's shaft).
    Gravity potential U(roll, pitch) = m*g*(height of the hand's centre of mass).
    The torque needed to hold the pose is the slope of that potential: tau = dU/dq.
    """
    def __init__(self, p: HandGravityParams = HandGravityParams()):
        self.p = p

    def _U(self, roll: float, pitch: float) -> float:
        p = self.p
        pos = _rot_y(-p.forearm_incline) @ _rot_y(-pitch) @ _rot_x(p.roll_dir * roll) @ np.array(p.com)
        return p.mass * p.g * pos[2]

    def torques(self, roll: float, pitch: float, eps: float = 1e-5) -> Tuple[float, float]:
        """(roll_torque, pitch_torque) in N*m that hold the pose against gravity."""
        t_roll = (self._U(roll + eps, pitch) - self._U(roll - eps, pitch)) / (2 * eps)
        t_pitch = (self._U(roll, pitch + eps) - self._U(roll, pitch - eps)) / (2 * eps)
        return t_roll, t_pitch


# --------------------------------------------------------------------------- #
# Adapter around differential calculator
# --------------------------------------------------------------------------- #
class DifferentialAdapter:
    def __init__(self, gravity: WristGravityModel | None = None):
        self.gravity = gravity or WristGravityModel()

    def solve(self, roll: float, pitch: float) -> JointTargets:
        """roll, pitch in rad -> motor-side targets in rad and N*m."""
        t_roll, t_pitch = self.gravity.torques(roll, pitch)
        a5, a6, t5, t6 = dc.calculate(math.degrees(roll), math.degrees(pitch), t_roll, t_pitch)
        return JointTargets(math.radians(a5), math.radians(a6), t5, t6)

    @staticmethod
    def wrist_from_motors(q5: float, q6: float) -> Tuple[float, float]:
        """Inverse of the calculator's angle mapping: motor rad -> (roll, pitch) rad."""
        s5 = (math.degrees(q5) - dc.J5_ZERO) / (dc.J5_SIGN * dc.PULLEY_RATIO)   # = roll + pitch (deg)
        s6 = (math.degrees(q6) - dc.J6_ZERO) / (dc.J6_SIGN * dc.PULLEY_RATIO)   # = roll - pitch (deg)
        return math.radians((s5 + s6) / 2), math.radians((s5 - s6) / 2)


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
@dataclass
class ControllerConfig:
    kp: Tuple[float, float] = (5.0, 5.0)         # N*m/rad at the output shaft. INPUT: you choose. Start low.
    kd: Tuple[float, float] = (0.25, 0.25)       # N*m/(rad/s). INPUT: you choose.
    tau_max: Tuple[float, float] = (2.5, 2.5)    # N*m per motor (RS05 peak is 5.5; 2.5 leaves ~2x margin)
    q_min: Tuple[float, float] = (-2 * math.pi, -2 * math.pi)  # rad, motor shaft. TODO: real limits
    q_max: Tuple[float, float] = (2 * math.pi, 2 * math.pi)    # (wrist +-90 deg each = motor +-360 deg)
    max_rate: float = 1.0                        # rad/s on roll/pitch: how fast the setpoint may move
    onboard_pd: bool = False                     # False: PD here -> torque -> current. True: MIT frame
    max_pos_err: float = 0.3                     # rad (motor shaft). Caps the P torque at kp*this
    rate_hz: float = 200.0                       # host loop rate over CAN


class ForearmGravityCompController:
    def __init__(self, model: DifferentialAdapter, bus: MotorBus,
                 cfg: ControllerConfig = ControllerConfig()):
        self.model, self.bus, self.cfg = model, bus, cfg
        self.target = (0.0, 0.0)       # requested (roll, pitch), rad
        self._cmd = None               # slew-limited (roll, pitch) actually tracked
        self._qdes_prev = None

    def set_target(self, roll: float, pitch: float) -> None:
        if abs(roll) > math.pi / 2 or abs(pitch) > math.pi / 2:
            raise ValueError("roll and pitch must each be within +-90 degrees")
        self.target = (roll, pitch)

    def _slew(self, dt: float) -> Tuple[float, float]:
        step = self.cfg.max_rate * dt
        return tuple(c + max(-step, min(step, t - c)) for c, t in zip(self._cmd, self.target))

    def step(self, dt: float) -> Tuple[float, float]:
        c = self.cfg
        q5, v5, q6, v6 = self.bus.read()
        q, v = (q5, q6), (v5, v6)

        for i in range(2):
            if not (c.q_min[i] <= q[i] <= c.q_max[i]):
                self.bus.send_torque(0.0, 0.0)
                raise RuntimeError(f"J{5+i} outside limits: {q[i]:.3f} rad")

        if self._cmd is None:          # start the ramp from where the wrist actually is
            self._cmd = self.model.wrist_from_motors(q5, q6)
        self._cmd = self._slew(dt)

        tgt = self.model.solve(*self._cmd)
        q_des = (tgt.q5, tgt.q6)
        tau_ff = tuple(max(-c.tau_max[i], min(c.tau_max[i], t))
                       for i, t in enumerate((tgt.tau5, tgt.tau6)))

        if self._qdes_prev is None or dt <= 0:     # desired velocity from the setpoint ramp
            v_des = (0.0, 0.0)
        else:
            v_des = tuple((a - b) / dt for a, b in zip(q_des, self._qdes_prev))
        self._qdes_prev = q_des

        # Safety: never let the position error (and so the P torque) get large
        q_des = tuple(q[i] + max(-c.max_pos_err, min(c.max_pos_err, q_des[i] - q[i]))
                      for i in range(2))

        if c.onboard_pd and hasattr(self.bus, "send_mit"):
            self.bus.send_mit(q_des, v_des, c.kp, c.kd, tau_ff)
            return tau_ff

        taus = []
        for i in range(2):
            tau = tau_ff[i] + c.kp[i] * (q_des[i] - q[i]) + c.kd[i] * (v_des[i] - v[i])
            taus.append(max(-c.tau_max[i], min(c.tau_max[i], tau)))
        self.bus.send_torque(*taus)
        return taus[0], taus[1]

    def run(self, duration: float | None = None) -> None:
        dt_nom = 1.0 / self.cfg.rate_hz
        t_end = None if duration is None else time.perf_counter() + duration
        last = time.perf_counter()
        try:
            while t_end is None or time.perf_counter() < t_end:
                now = time.perf_counter()
                dt, last = now - last, now
                self.step(dt if dt > 0 else dt_nom)
                time.sleep(max(0.0, dt_nom - (time.perf_counter() - now)))
        finally:
            try:
                self.bus.send_torque(0.0, 0.0)       # fail safe: go limp
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Robstride RS05 current-mode bus (what you use: command current, read pos/vel)
# --------------------------------------------------------------------------- #
@dataclass
class RS05:
    gear_ratio: float = 7.75        # internal reduction (datasheet)
    kt_rotor: float = 0.0866        # N*m per A of iq at the ROTOR. Derived from the datasheet's
    #                                 7.4 Vrms/kRPM back-EMF constant (equivalently 0.94 N*m/Arms at the
    #                                 output = 7.75 x 0.0866 x sqrt(2)). TODO: verify against your library's
    #                                 current units (peak vs rms) with the check described in the notes.
    efficiency: float = 1.0         # set ~0.8-0.9 if measured output torque comes out low

    def amps_per_output_nm(self) -> float:
        return 1.0 / (self.gear_ratio * self.kt_rotor * self.efficiency)


class RobstrideCurrentBus:
    """
    Torque (N*m at the OUTPUT shaft) -> rotor current -> your library.
        iq = tau_out / (gear_ratio * kt_rotor * efficiency)
    TODO: pass two functions built from the robstride library:
        send_current_fn(motor_id, iq_amps) -> None   # current / torque command
        read_fn(motor_id) -> (position_rad, velocity_rad_s)   # output-shaft feedback
    motor_ids = (J5_can_id, J6_can_id), from can_test.py.
    Note: iq = 0 means the joint free-wheels, so on any error the wrist goes limp and will droop.
    """
    def __init__(self, motor_ids, send_current_fn, read_fn, motor: RS05 = RS05(), tau_limit: float = 2.5):
        self.ids, self.send_i, self.read_fn = motor_ids, send_current_fn, read_fn
        self.motor, self.tau_limit = motor, tau_limit

    def read(self):
        (q5, v5), (q6, v6) = (self.read_fn(m) for m in self.ids)
        return q5, v5, q6, v6

    def send_torque(self, tau5, tau6):
        k = self.motor.amps_per_output_nm()
        for m, t in zip(self.ids, (tau5, tau6)):
            t = max(-self.tau_limit, min(self.tau_limit, t))      # second clamp, independent of the controller
            self.send_i(m, t * k)


# --------------------------------------------------------------------------- #
# Robstride bus (plug in the client calls from Robstride_Bringup)
# --------------------------------------------------------------------------- #
class RobstrideMITBus:
    """
    TODO: pass two functions built from the Robstride_Bringup client:
        send_fn(motor_id, p_des, v_des, kp, kd, t_ff) -> None   # one MIT/operation frame
        read_fn(motor_id) -> (position_rad, velocity_rad_s)     # latest feedback
    motor_ids = (J5_can_id, J6_can_id). Positions must use the same zero as the calculator
    (J5_ZERO / J6_ZERO in differential_calculator.py): zero the motors in the straight pose.
    """
    def __init__(self, motor_ids, send_fn, read_fn):
        self.ids, self.send_fn, self.read_fn = motor_ids, send_fn, read_fn

    def read(self):
        (q5, v5), (q6, v6) = (self.read_fn(m) for m in self.ids)
        return q5, v5, q6, v6

    def send_mit(self, q_des, v_des, kp, kd, tau_ff):
        for i, m in enumerate(self.ids):
            self.send_fn(m, q_des[i], v_des[i], kp[i], kd[i], tau_ff[i])

    def send_torque(self, tau5, tau6):          # pure torque = MIT frame with kp = kd = 0
        for m, t in zip(self.ids, (tau5, tau6)):
            self.send_fn(m, 0.0, 0.0, 0.0, 0.0, t)


if __name__ == "__main__":
    # Quick look at what the calculator + gravity model produce
    model = DifferentialAdapter(WristGravityModel(HandGravityParams(com=(0.10, 0.0, 0.02))))
    for roll_deg, pitch_deg in [(0, 0), (0, 30), (20, 30), (-15, 10)]:
        t = model.solve(math.radians(roll_deg), math.radians(pitch_deg))
        print(f"roll {roll_deg:+4d} pitch {pitch_deg:+4d} deg -> "
              f"J5 {math.degrees(t.q5):+7.1f} deg {t.tau5:+.3f} N*m | "
              f"J6 {math.degrees(t.q6):+7.1f} deg {t.tau6:+.3f} N*m")