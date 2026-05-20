#!/usr/bin/env python3
"""
Go2 CPT (Cone Penetration Test) deployment script.

Sequence:
  1. Stand  — ramp all 4 legs to default pose
  2. Lift   — FR leg retracts upward (foot clears the soil surface)
  3. Probe  — FR leg extends straight down; torque jump marks soil contact;
              continued penetration tracks resistance vs depth
  4. Hold   — pause at max depth
  5. Retract — FR leg returns to stand pose

FL, RL, RR are locked at standing pose with high stiffness throughout.
tau_est from FR joints is logged to CSV.  The torque jump at first
contact gives a clean depth=0 reference for the soil profile.

Usage (on robot):
  python go2_cpt_probe.py
  python go2_cpt_probe.py eth0              # specify DDS network interface
  python go2_cpt_probe.py --lift 0.06       # lift 60 mm (default 80 mm)
  python go2_cpt_probe.py --speed 0.003     # 3 mm/s penetration (default 5)
  python go2_cpt_probe.py --depth 0.05      # probe only 50 mm below surface
  python go2_cpt_probe.py --output run1.csv # custom CSV name
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import torch

# ── Geometry ────────────────────────────────────────────────────────────────────
L_THIGH = 0.213   # m, thigh link
L_CALF  = 0.213   # m, calf link

# ── SDK motor order: FR(0-2)  FL(3-5)  RR(6-8)  RL(9-11) ──────────────────────
# Mirrors go2_policy_walk.py DEFAULT_DOF_POS
DEFAULT_DOF_POS = [
    0.0,  0.8, -1.5,   # FR: hip, thigh, calf
    0.0,  0.8, -1.5,   # FL: hip, thigh, calf
    0.0,  1.0, -1.5,   # RR: hip, thigh, calf
    0.0,  1.0, -1.5,   # RL: hip, thigh, calf
]

# FR motor indices in SDK order
FR_HIP_IDX   = 0
FR_THIGH_IDX = 1
FR_CALF_IDX  = 2
LOCK_IDXS    = [3,4,5, 6,7,8, 9,10,11]   # FL, RR, RL

# Joint limits from go2.xml
FR_THIGH_LO, FR_THIGH_HI = -1.5708,  3.4907
FR_CALF_LO,  FR_CALF_HI  = -2.7227, -0.83776

# ── PD gains ────────────────────────────────────────────────────────────────────
STAND_KP   = 40.0    # ramp-to-stand phase
STAND_KD   = 0.5

LOCK_KP    = 80.0    # FL, RR, RL during probe
LOCK_KD    = 2.0

PROBE_KP   = 60.0    # FR leg during probe
PROBE_KD   = 1.0

# ── Timing ──────────────────────────────────────────────────────────────────────
POLICY_HZ   = 50
POLICY_DT   = 1.0 / POLICY_HZ
LOWCMD_HZ   = 500

STAND_SECONDS    = 4.0     # ramp duration
LIFT_TIME_S      = 2.0     # s — ramp FR foot from ground to lifted height
SAFETY_HOLD_S    = 1.0     # hold at max depth before withdrawing

MAX_STEP_RAD     = 0.10    # rad per control step (slew limit during stand ramp)

# Safety: abort probe if body tilt exceeds this (rad) — ~15°
MAX_TILT_RAD     = 0.26

# ── CPT defaults (overridable via CLI) ──────────────────────────────────────────
DEFAULT_LIFT_M    = 0.08    # 80 mm above ground contact
DEFAULT_SPEED_M_S = 0.005   # 5 mm/s
DEFAULT_DEPTH_M   = 0.07    # 70 mm below soil surface
MAX_DEPTH_M       = 0.075   # hard cap (joint limit safety)


# ══════════════════════════════════════════════════════════════════════════════
# Kinematics (sagittal plane, FR leg, hip frame)
# ══════════════════════════════════════════════════════════════════════════════

def leg_fk(theta: float, phi: float) -> tuple[float, float]:
    """
    FR leg forward kinematics in the sagittal (x-z) plane.

    MuJoCo Ry(θ) convention:  Ry(θ)*[0,0,-L] = [-L*sin θ, 0, -L*cos θ]

        foot_x = -L_THIGH*sin(θ) - L_CALF*sin(θ+φ)
        foot_z = -L_THIGH*cos(θ) - L_CALF*cos(θ+φ)

    Returns (x_foot, z_foot) in FR_hip frame (x fwd, z negative downward).
    Verified: θ=0.8, φ=-1.5  →  x≈-0.016 m, z≈-0.311 m.
    """
    x = -L_THIGH * math.sin(theta) - L_CALF * math.sin(theta + phi)
    z = -L_THIGH * math.cos(theta) - L_CALF * math.cos(theta + phi)
    return x, z


def leg_ik(x_foot: float, z_foot: float) -> tuple[float, float]:
    """
    2-link sagittal IK for FR leg (knee-backward, φ < 0).

    Returns (theta_thigh, phi_calf).
    """
    X = -x_foot
    Z = -z_foot
    r2      = X**2 + Z**2
    cos_phi = max(-1.0, min(1.0, (r2 - L_THIGH**2 - L_CALF**2)
                             / (2.0 * L_THIGH * L_CALF)))
    phi     = -math.acos(cos_phi)
    A       = L_THIGH + L_CALF * math.cos(phi)
    B       = L_CALF  * math.sin(phi)
    theta   = math.atan2(X, Z) - math.atan2(B, A)
    return theta, phi


# ══════════════════════════════════════════════════════════════════════════════
# Helpers shared with go2_policy_walk.py
# ══════════════════════════════════════════════════════════════════════════════

def slew_limit(prev: float, new: float, max_step: float) -> float:
    return prev + max(-max_step, min(max_step, new - prev))


def quat_to_rpy(q_wxyz) -> tuple[float, float, float]:
    """Convert quaternion [w,x,y,z] to roll, pitch, yaw (rad)."""
    w, x, y, z = q_wxyz
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


# ══════════════════════════════════════════════════════════════════════════════
# DDS initialisation (mirrors go2_policy_walk.py)
# ══════════════════════════════════════════════════════════════════════════════

def init_dds(iface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if iface:
        ChannelFactoryInitialize(0, iface)
        print(f"[cpt] DDS interface: {iface}")
    else:
        ChannelFactoryInitialize(0)
        print("[cpt] DDS interface: default")


def wait_for_lowstate() -> dict:
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    latest = {"msg": None}

    def cb(msg: LowState_):
        latest["msg"] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(cb, 10)
    print("[cpt] Waiting for rt/lowstate ...")
    while latest["msg"] is None:
        time.sleep(0.05)
    print("[cpt] Got first lowstate.")
    return latest


def release_highlevel() -> None:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    sc  = SportClient();  sc.SetTimeout(5.0);  sc.Init()
    msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()

    print("[cpt] Releasing high-level control ...")
    status, result = msc.CheckMode()
    while result.get("name"):
        sc.StandDown()
        msc.ReleaseMode()
        time.sleep(1.0)
        status, result = msc.CheckMode()
    print("[cpt] High-level control released.")


def lowstate_to_motors(msg) -> list[dict]:
    """Extract per-motor state from LowState message."""
    motors = []
    for i in range(12):
        ms = msg.motor_state[i]
        motors.append({
            "q":       float(ms.q),
            "dq":      float(ms.dq),
            "tau_est": float(ms.tau_est),
        })
    return motors


# ══════════════════════════════════════════════════════════════════════════════
# Low-level command writer (500 Hz thread)
# ══════════════════════════════════════════════════════════════════════════════

def make_lowcmd_writer(pub, low_cmd, crc, shared: dict):
    """Return the 500-Hz callback that sends joint position commands."""
    import unitree_legged_const as go2

    def write_lowcmd():
        tq = shared["target_q"]
        kp = shared["kp"]
        kd = shared["kd"]
        if tq is None:
            return
        for i in range(12):
            low_cmd.motor_cmd[i].mode = 0x01
            low_cmd.motor_cmd[i].q    = tq[i]
            low_cmd.motor_cmd[i].dq   = 0.0
            low_cmd.motor_cmd[i].kp   = kp[i]
            low_cmd.motor_cmd[i].kd   = kd[i]
            low_cmd.motor_cmd[i].tau  = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    return write_lowcmd


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Go2 CPT probe — vertical FR-leg penetration test")
    parser.add_argument("iface", nargs="?", default=None,
                        help="DDS network interface (e.g. eth0)")
    parser.add_argument("--lift",  type=float, default=DEFAULT_LIFT_M,
                        help="FR foot lift height m (default 0.08 = 80 mm)")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED_M_S,
                        help="Penetration speed m/s (default 0.005)")
    parser.add_argument("--depth", type=float, default=DEFAULT_DEPTH_M,
                        help="Max penetration m below surface (default 0.07 = 70 mm)")
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: cpt_<timestamp>.csv)")
    args = parser.parse_args()

    lift_height_m = args.lift
    lift_step_m   = (lift_height_m / LIFT_TIME_S) * POLICY_DT   # m per step upward
    max_depth     = min(args.depth, MAX_DEPTH_M)
    probe_step    = args.speed * POLICY_DT                       # m per step downward
    csv_path      = args.output or f"cpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"\n{'='*52}")
    print("  Go2 CPT Probe Deployment")
    print(f"  Lift   : {lift_height_m*1e3:.0f} mm")
    print(f"  Speed  : {args.speed*1e3:.1f} mm/s")
    print(f"  Depth  : {max_depth*1e3:.0f} mm below surface")
    print(f"  Output : {csv_path}")
    print(f"{'='*52}\n")

    # ── DDS + robot setup ─────────────────────────────────────────────────────
    init_dds(args.iface)
    latest = wait_for_lowstate()
    release_highlevel()

    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread
    import unitree_legged_const as go2

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc     = CRC()
    low_cmd = unitree_go_msg_dds__LowCmd_()
    low_cmd.head[0]     = 0xFE
    low_cmd.head[1]     = 0xEF
    low_cmd.level_flag  = 0xFF
    low_cmd.gpio        = 0

    for i in range(20):
        low_cmd.motor_cmd[i].mode = 0x01
        low_cmd.motor_cmd[i].q   = go2.PosStopF
        low_cmd.motor_cmd[i].dq  = go2.VelStopF
        low_cmd.motor_cmd[i].kp  = 0.0
        low_cmd.motor_cmd[i].kd  = 0.0
        low_cmd.motor_cmd[i].tau = 0.0

    # Shared state for the 500-Hz writer
    shared = {
        "target_q": None,
        "kp": [STAND_KP] * 12,
        "kd": [STAND_KD] * 12,
    }

    # ── read initial joint positions ──────────────────────────────────────────
    raw0    = lowstate_to_motors(latest["msg"])
    start_q = [m["q"] for m in raw0]
    shared["target_q"] = list(start_q)

    # ── start 500-Hz writer ───────────────────────────────────────────────────
    writer_cb = make_lowcmd_writer(pub, low_cmd, crc, shared)
    writer = RecurrentThread(interval=1.0/LOWCMD_HZ,
                             target=writer_cb, name="lowcmd_writer")
    writer.Start()
    print(f"[cpt] LowCmd writer started at {LOWCMD_HZ} Hz.")

    # ── PHASE 1: ramp to standing pose ────────────────────────────────────────
    print(f"\n[cpt] Ramping to stand pose ({STAND_SECONDS:.0f}s) ...")
    ramp_steps = max(1, int(STAND_SECONDS * POLICY_HZ))
    prev_q     = list(start_q)

    for k in range(ramp_steps):
        alpha = (k + 1) / float(ramp_steps)
        for i in range(12):
            desired = (1.0 - alpha) * start_q[i] + alpha * DEFAULT_DOF_POS[i]
            desired = slew_limit(prev_q[i], desired, MAX_STEP_RAD)
            prev_q[i] = desired
        shared["target_q"] = list(prev_q)
        time.sleep(POLICY_DT)

    print("[cpt] Stand pose reached.")

    # ── PHASE 2: user confirmation ────────────────────────────────────────────
    print("\n*** ROBOT STANDING — READY FOR CPT PROBE ***")
    print("  Probe leg : FR (front-right)")
    print(f"  Speed     : {args.speed*1e3:.1f} mm/s")
    print(f"  Depth     : {max_depth*1e3:.0f} mm")
    print("  Type 'go' + Enter to start probe.  Anything else aborts.\n")
    user_in = input("> ").strip().lower()
    if user_in != "go":
        print("[cpt] Aborted. Holding stand pose. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        return

    # ── switch to probe gains ─────────────────────────────────────────────────
    kp_list = [LOCK_KP] * 12
    kd_list = [LOCK_KD] * 12
    kp_list[FR_HIP_IDX]   = PROBE_KP
    kp_list[FR_THIGH_IDX] = PROBE_KP
    kp_list[FR_CALF_IDX]  = PROBE_KP
    kd_list[FR_HIP_IDX]   = PROBE_KD
    kd_list[FR_THIGH_IDX] = PROBE_KD
    kd_list[FR_CALF_IDX]  = PROBE_KD
    shared["kp"] = kp_list
    shared["kd"] = kd_list

    # ── initial FR foot position from actual joint angles ─────────────────────
    m_now  = lowstate_to_motors(latest["msg"])
    theta0 = m_now[FR_THIGH_IDX]["q"]
    phi0   = m_now[FR_CALF_IDX]["q"]
    x0, z0 = leg_fk(theta0, phi0)
    z_lifted = z0 + lift_height_m   # target z after lifting (less negative = higher)
    print(f"\n[cpt] FR foot at ground (hip frame): x={x0*1e3:+.1f} mm  z={z0*1e3:+.1f} mm")
    print(f"[cpt] FR foot lifted target:         x={x0*1e3:+.1f} mm  z={z_lifted*1e3:+.1f} mm")

    # current target_q starts from DEFAULT_DOF_POS (stand pose)
    current_q = list(DEFAULT_DOF_POS)

    # ── PHASE 3: lift + probe state machine ──────────────────────────────────
    # "lift"  — FR foot rises from z0 to z_lifted
    # "probe" — FR foot descends from z_lifted downward through z0 (contact)
    #            and continues to z0 - max_depth (penetration)
    print("[cpt] Lifting FR leg.  Ctrl+C to abort.\n")

    phase           = "lift"
    lift_z_current  = z0         # tracks foot z during lift
    probe_z_current = z_lifted   # tracks foot z during probe
    log_rows: list[dict] = []
    step = 0

    try:
        next_tick = time.monotonic()

        while phase != "done":
            # precise 50-Hz timing
            now   = time.monotonic()
            sleep = next_tick - now
            if sleep > 0:
                time.sleep(sleep)
            next_tick += POLICY_DT

            # read sensors
            motors = lowstate_to_motors(latest["msg"])
            imu    = latest["msg"].imu_state
            quat   = [imu.quaternion[0], imu.quaternion[1],
                      imu.quaternion[2], imu.quaternion[3]]
            roll, pitch, _ = quat_to_rpy(quat)

            # tipping safety check
            tilt = math.sqrt(roll**2 + pitch**2)
            if tilt > MAX_TILT_RAD:
                print(f"\n[cpt] ABORT: tilt {math.degrees(tilt):.1f}° "
                      f"exceeds limit ({math.degrees(MAX_TILT_RAD):.0f}°).")
                break

            # ── state machine ──────────────────────────────────────────────
            if phase == "lift":
                lift_z_current = min(lift_z_current + lift_step_m, z_lifted)
                theta, phi = leg_ik(x0, lift_z_current)
                if lift_z_current >= z_lifted:
                    phase = "probe"
                    probe_z_current = z_lifted
                    print(f"\n[cpt] FR leg lifted {lift_height_m*1e3:.0f} mm. "
                          f"Starting probe ...")

            elif phase == "probe":
                probe_z_current = max(probe_z_current - probe_step,
                                      z0 - max_depth)
                theta, phi = leg_ik(x0, probe_z_current)
                if probe_z_current <= z0 - max_depth:
                    phase = "done"

            # enforce joint limits and send command
            theta = max(FR_THIGH_LO, min(FR_THIGH_HI, theta))
            phi   = max(FR_CALF_LO,  min(FR_CALF_HI,  phi))
            current_q[FR_HIP_IDX]   = 0.0
            current_q[FR_THIGH_IDX] = theta
            current_q[FR_CALF_IDX]  = phi
            shared["target_q"] = list(current_q)

            # log — depth_from_surface_mm: negative = above soil, positive = below
            depth_mm = (z0 - probe_z_current) * 1e3 if phase != "lift" else (z0 - lift_z_current) * 1e3
            tau_hip   = motors[FR_HIP_IDX]["tau_est"]
            tau_thigh = motors[FR_THIGH_IDX]["tau_est"]
            tau_calf  = motors[FR_CALF_IDX]["tau_est"]
            log_rows.append({
                "time_s":               round(time.monotonic(), 4),
                "step":                 step,
                "phase":                phase,
                "depth_from_surface_mm":round(depth_mm, 2),
                "fr_thigh_cmd_rad":     round(theta, 4),
                "fr_calf_cmd_rad":      round(phi, 4),
                "fr_thigh_actual_rad":  round(motors[FR_THIGH_IDX]["q"], 4),
                "fr_calf_actual_rad":   round(motors[FR_CALF_IDX]["q"], 4),
                "fr_thigh_dq_rad_s":    round(motors[FR_THIGH_IDX]["dq"], 4),
                "fr_calf_dq_rad_s":     round(motors[FR_CALF_IDX]["dq"], 4),
                "tau_fr_hip_Nm":        round(tau_hip, 3),
                "tau_fr_thigh_Nm":      round(tau_thigh, 3),
                "tau_fr_calf_Nm":       round(tau_calf, 3),
                "roll_rad":             round(roll, 4),
                "pitch_rad":            round(pitch, 4),
            })

            if step % POLICY_HZ == 0:
                print(f"\r  [{phase:5s}]  step={step:04d}  "
                      f"depth={depth_mm:+7.1f}mm  "
                      f"τ_thigh={tau_thigh:+6.2f}Nm  "
                      f"τ_calf={tau_calf:+6.2f}Nm  "
                      f"roll={math.degrees(roll):+5.1f}°  ",
                      end="", flush=True)
            step += 1

    except KeyboardInterrupt:
        print("\n[cpt] Interrupted by user.")

    print(f"\n[cpt] Probe done. "
          f"Final depth: {(z0 - probe_z_current)*1e3:.1f} mm below surface")

    # ── PHASE 4: hold at final depth briefly ──────────────────────────────────
    print(f"[cpt] Holding {SAFETY_HOLD_S:.1f}s ...")
    time.sleep(SAFETY_HOLD_S)

    # ── PHASE 5: retract back to stand pose ───────────────────────────────────
    print("[cpt] Retracting to stand pose ...")
    retract_steps = ramp_steps
    start_q_retract = list(current_q)

    for k in range(retract_steps):
        alpha = (k + 1) / float(retract_steps)
        for i in range(12):
            desired = (1.0 - alpha) * start_q_retract[i] + alpha * DEFAULT_DOF_POS[i]
            current_q[i] = desired
        shared["target_q"] = list(current_q)
        shared["kp"] = [STAND_KP] * 12
        shared["kd"] = [STAND_KD] * 12
        time.sleep(POLICY_DT)

    print("[cpt] Retracted to stand pose.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    print(f"\n[cpt] Writing {len(log_rows)} rows → {csv_path}")
    if log_rows:
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(log_rows)
        print(f"[cpt] Saved: {csv_path}")
    else:
        print("[cpt] No data logged.")

    # ── Safe stop ─────────────────────────────────────────────────────────────
    print("[cpt] Sending safe-stop packets ...")
    for _ in range(200):
        for i in range(12):
            low_cmd.motor_cmd[i].q   = go2.PosStopF
            low_cmd.motor_cmd[i].dq  = go2.VelStopF
            low_cmd.motor_cmd[i].kp  = 0.0
            low_cmd.motor_cmd[i].kd  = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)
        time.sleep(0.002)
    print("[cpt] Done.")


if __name__ == "__main__":
    main()
