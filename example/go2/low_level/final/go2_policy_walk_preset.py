"""Go2 real-robot walk with preset velocity, fixed duration, and data logging.

Derived from go2_policy_walk.py. Instead of keyboard velocity control the robot
walks for a fixed duration at a preset [vx, vy, wz] command, then automatically
returns to stand and saves the recorded data.

A live matplotlib window shows 15 subplots (5 rows × 3 cols):
  Row 0 : vx_cmd, vy_cmd, wz (cmd dashed / gyro actual solid)
  Row 1 : FR hip | FR thigh | FR calf    torques  (tau_est)
  Row 2 : FL hip | FL thigh | FL calf    torques
  Row 3 : RR hip | RR thigh | RR calf    torques
  Row 4 : RL hip | RL thigh | RL calf    torques

On exit the recorded policy-phase data is saved to:
  data/go2_robot_walk_vx+0.30_vy+0.00_wz+0.00_20260518_120000.npz

Note: body linear velocity (vx_act, vy_act) is not available from the Go2
lowstate — those plots show the commanded value only. wz_act is read from
the IMU gyroscope (body-frame yaw rate).

Usage:
    cd /path/to/deploy/example/go2/low_level/final
    python go2_policy_walk_preset.py [network_interface]

    # Example: override velocity at the command line
    python go2_policy_walk_preset.py eth0 --vx 0.4 --duration 20
"""
# matplotlib must be imported and configured before any other GUI toolkit
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import torch
import numpy as np

# ============================================================
# CHECKPOINT
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH  = os.path.join(SCRIPT_DIR, "walk.pt")

# ============================================================
# PRESET VELOCITY COMMAND — edit these three values
# ============================================================
PRESET_VX = 0.2    # forward velocity  m/s   (positive = forward)
PRESET_VY = 0.0    # lateral velocity  m/s   (positive = left)
PRESET_WZ = 0.0    # yaw rate          rad/s (positive = CCW)

# ============================================================
# RUN DURATION (policy phase only; stand ramp is extra)
# ============================================================
PRESET_DURATION_S = 10.0   # seconds

# ============================================================
# SAVE DIRECTORY
# ============================================================
SAVE_DIR = os.path.join(SCRIPT_DIR, "data")

# ============================================================
# TIMING
# ============================================================
POLICY_HZ  = 50.0
LOWCMD_HZ  = 500.0
RECORD_HZ  = 500.0   # torque / state recording frequency

# ============================================================
# STAND PHASE
# ============================================================
STAND_SECONDS    = 4.0
STAND_KP         = 40.0
STAND_KD         = 0.5
TRANSITION_SECS  = 2.0   # return-to-stand ramp after policy

# ============================================================
# PLS — must match training
# ============================================================
PLS_ENABLE          = True
PLS_KP_DEFAULT      = 40.0
PLS_KP_ACTION_SCALE = 20.0
PLS_KP_RANGE        = [20.0, 60.0]
KP_FACTOR           = 1.0
KD_FACTOR           = 1.0

POLICY_KP_FALLBACK = 40.0
POLICY_KD_FALLBACK = 2.0

# ============================================================
# POLICY DIMS — must match training
# ============================================================
ACTION_CLIP      = 100.0
ACTION_SCALE     = 0.25
MAX_STEP_RAD     = 0.1
NUM_POS_ACTIONS  = 12
NUM_ACT          = NUM_POS_ACTIONS + (4 if PLS_ENABLE else 0)  # 16
NUM_OBS          = 3 + 3 + 3 + 12 + 12 + NUM_ACT              # 49
OBS_SCALES       = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}

# ============================================================
# JOINT / LEG LAYOUT
# ============================================================
JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
LEG_NAMES    = ["FR", "FL", "RR", "RL"]
LEG_JOINT_MAP = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

DEFAULT_DOF_POS = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=torch.float32,
)

# ============================================================
# PLOT CONFIG
# ============================================================
PLOT_WINDOW = 200   # rolling window (steps); 200 × 0.02 s = 4 s
PLOT_HZ     = 20    # plot refresh rate (Hz)

_LEG_COLORS = [
    (0.85, 0.10, 0.10),   # FR — red
    (0.10, 0.35, 0.85),   # FL — blue
    (0.90, 0.50, 0.00),   # RR — orange
    (0.10, 0.65, 0.20),   # RL — green
]

# ============================================================
# HEADING LOCK — prevents yaw drift when wz preset ≈ 0
# ============================================================
HEADING_KP     = 2.8
MAX_WZ         = 1.5
_heading_state = {"target_yaw": None, "locked": False}


def _yaw_from_quat_wxyz(q_wxyz):
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def make_command_list(raw=None, vx=None, vy=None, wz=None):
    """Return [vx, vy, wz_effective] with heading-lock when wz ≈ 0."""
    vx = PRESET_VX if vx is None else vx
    vy = PRESET_VY if vy is None else vy
    wz = PRESET_WZ if wz is None else wz

    if raw is None or abs(vx) + abs(vy) < 0.05:
        _heading_state["locked"] = False
        _heading_state["target_yaw"] = None
        return [vx, vy, wz]

    current_yaw = _yaw_from_quat_wxyz(raw["imu"]["quat_wxyz"])

    if abs(wz) > 0.01:
        _heading_state["target_yaw"] = current_yaw
        _heading_state["locked"] = False
        return [vx, vy, wz]

    if not _heading_state["locked"] or _heading_state["target_yaw"] is None:
        _heading_state["target_yaw"] = current_yaw
        _heading_state["locked"] = True

    heading_err = _wrap_to_pi(_heading_state["target_yaw"] - current_yaw)
    wz_eff = float(max(-MAX_WZ, min(MAX_WZ, HEADING_KP * heading_err)))
    return [vx, vy, wz_eff]


# ============================================================
# PLS
# ============================================================
def compute_pls_kp_kd(stiffness_actions_4):
    kp_leg = PLS_KP_DEFAULT + stiffness_actions_4 * PLS_KP_ACTION_SCALE
    kp_leg = torch.clamp(kp_leg, PLS_KP_RANGE[0], PLS_KP_RANGE[1])
    kp_12  = torch.zeros(12, dtype=torch.float32)
    for leg, joints in enumerate(LEG_JOINT_MAP):
        for j in joints:
            kp_12[j] = kp_leg[leg]
    kd_12 = 0.2 * torch.sqrt(kp_12)
    return kp_12 * KP_FACTOR, kd_12 * KD_FACTOR


# ============================================================
# QUATERNION / OBSERVATION HELPERS
# ============================================================
def quat_conj(q):
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], dtype=torch.float32)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return torch.tensor([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=torch.float32)


def rotate_vec_by_quat(v, q):
    vq = torch.tensor([0.0, v[0], v[1], v[2]], dtype=torch.float32)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]


def projected_gravity(q_wxyz):
    q  = torch.tensor(q_wxyz, dtype=torch.float32)
    g  = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    return rotate_vec_by_quat(g, quat_conj(q))


def build_obs(raw, command_3, last_action):
    gyro   = torch.tensor(raw["imu"]["gyro_rad_s"], dtype=torch.float32)
    proj_g = projected_gravity(raw["imu"]["quat_wxyz"])
    q      = torch.tensor([m["q_rad"]    for m in raw["motors"]], dtype=torch.float32)
    dq     = torch.tensor([m["dq_rad_s"] for m in raw["motors"]], dtype=torch.float32)
    cmd    = torch.tensor(command_3, dtype=torch.float32)
    cmd_sc = torch.tensor(
        [OBS_SCALES["lin_vel"], OBS_SCALES["lin_vel"], OBS_SCALES["ang_vel"]],
        dtype=torch.float32,
    )
    obs = torch.cat([
        gyro * OBS_SCALES["ang_vel"],
        proj_g,
        cmd * cmd_sc,
        (q - DEFAULT_DOF_POS) * OBS_SCALES["dof_pos"],
        dq * OBS_SCALES["dof_vel"],
        last_action,
    ], dim=0)
    if obs.shape[0] != NUM_OBS:
        raise RuntimeError(f"obs dim mismatch: expected {NUM_OBS}, got {obs.shape[0]}")
    return obs


# ============================================================
# LOWSTATE → RAW DICT  (includes tau_est for torque logging)
# ============================================================
def lowstate_to_raw(low_state):
    imu      = low_state.imu_state
    quat     = list(imu.quaternion)   # [w, x, y, z]
    gyro     = list(imu.gyroscope)
    motors   = []
    for i in range(12):
        ms = low_state.motor_state[i]
        motors.append({
            "q_rad":    float(ms.q),
            "dq_rad_s": float(ms.dq),
            "tau_est":  float(ms.tau_est),
        })
    return {
        "imu": {
            "quat_wxyz":  [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
            "gyro_rad_s": [float(gyro[0]), float(gyro[1]), float(gyro[2])],
        },
        "motors": motors,
    }


# ============================================================
# POLICY LOADER
# ============================================================
def load_policy(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["model_state_dict"]
    try:
        from rsl_rl.modules import ActorCritic
    except Exception:
        from rsl_rl.modules.actor_critic import ActorCritic
    num_critic_obs = sd["critic.0.weight"].shape[1]
    policy = ActorCritic(
        num_actor_obs=NUM_OBS, num_critic_obs=num_critic_obs, num_actions=NUM_ACT,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", init_noise_std=1.0,
    )
    policy.load_state_dict(sd, strict=True)
    policy.eval()
    print(f"  Loaded: actor_obs={NUM_OBS}  critic_obs={num_critic_obs}  actions={NUM_ACT}")
    return policy


# ============================================================
# SLEW LIMIT
# ============================================================
def slew_limit(prev_q, new_q, max_step):
    return prev_q + torch.clamp(new_q - prev_q, -max_step, max_step)


# ============================================================
# SHARED DATA BUFFERS
# ============================================================
_plot_lock    = threading.Lock()
_plot_running = threading.Event()
_plot_running.set()

_buf_vx_cmd  = deque([0.0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
_buf_vy_cmd  = deque([0.0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
_buf_wz_cmd  = deque([0.0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
_buf_wz_act  = deque([0.0] * PLOT_WINDOW, maxlen=PLOT_WINDOW)
_buf_torques = [deque([0.0] * PLOT_WINDOW, maxlen=PLOT_WINDOW) for _ in range(12)]

# Full history (grows during policy phase at RECORD_HZ; saved on exit)
_hist_time    = []
_hist_vx_cmd  = []
_hist_vy_cmd  = []
_hist_wz_cmd  = []
_hist_wz_act  = []
_hist_torques = [[] for _ in range(12)]
_hist_dof_pos = [[] for _ in range(12)]
_hist_dof_vel = [[] for _ in range(12)]

# Command state shared between policy loop (writer) and recorder thread (reader)
_cmd_state        = [0.0, 0.0, 0.0]   # [vx, vy, wz]
_recorder_running = threading.Event()


def _push_to_plot(vx_cmd, vy_cmd, wz_cmd, wz_act, torques_12):
    """Update rolling plot buffers — called at POLICY_HZ by the policy loop."""
    with _plot_lock:
        _buf_vx_cmd.append(float(vx_cmd))
        _buf_vy_cmd.append(float(vy_cmd))
        _buf_wz_cmd.append(float(wz_cmd))
        _buf_wz_act.append(float(wz_act))
        for i, tau in enumerate(torques_12):
            _buf_torques[i].append(float(tau))


def _push_to_history(t, vx_cmd, vy_cmd, wz_cmd, wz_act, torques_12, dof_pos_12, dof_vel_12):
    """Append one sample to the saved history — called at RECORD_HZ by the recorder thread."""
    with _plot_lock:
        _hist_time.append(float(t))
        _hist_vx_cmd.append(float(vx_cmd))
        _hist_vy_cmd.append(float(vy_cmd))
        _hist_wz_cmd.append(float(wz_cmd))
        _hist_wz_act.append(float(wz_act))
        for i in range(12):
            _hist_torques[i].append(float(torques_12[i]))
            _hist_dof_pos[i].append(float(dof_pos_12[i]))
            _hist_dof_vel[i].append(float(dof_vel_12[i]))


# ============================================================
# LIVE PLOT — 15 subplots in 5 × 3 grid
# ============================================================
def _plot_thread_fn(vx, vy, wz, duration):
    xs = list(range(PLOT_WINDOW))
    tick_pos    = [0, 50, 100, 150, 199]
    tick_labels = [f"-{(PLOT_WINDOW - 1 - p) / POLICY_HZ:.1f}s" for p in tick_pos]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Go2 Robot Walk (real)    "
        f"vx={vx:+.3f} m/s    vy={vy:+.3f} m/s    wz={wz:+.3f} rad/s    "
        f"duration={duration:.0f} s",
        fontsize=10,
    )
    gs_layout = gridspec.GridSpec(5, 3, hspace=0.60, wspace=0.38)

    def _setup_ax(ax, title, ylabel, ylim):
        ax.set_xlim(0, PLOT_WINDOW - 1)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6)

    # ── Row 0: velocity / yaw ───────────────────────────────────────
    ax_vx = fig.add_subplot(gs_layout[0, 0])
    ax_vy = fig.add_subplot(gs_layout[0, 1])
    ax_wz = fig.add_subplot(gs_layout[0, 2])

    _setup_ax(ax_vx, "vx  (cmd only — no sensor)", "m/s",   (-1.3, 1.3))
    _setup_ax(ax_vy, "vy  (cmd only — no sensor)", "m/s",   (-1.3, 1.3))
    _setup_ax(ax_wz, "wz  (cmd dashed / gyro solid)", "rad/s", (-2.0, 2.0))

    ln_vx_cmd, = ax_vx.plot(xs, list(_buf_vx_cmd), "tomato",       ls="--", lw=1.5, label="cmd")
    ln_vy_cmd, = ax_vy.plot(xs, list(_buf_vy_cmd), "steelblue",    ls="--", lw=1.5, label="cmd")
    ln_wz_cmd, = ax_wz.plot(xs, list(_buf_wz_cmd), "mediumpurple", ls="--", lw=1.0, label="cmd")
    ln_wz_act, = ax_wz.plot(xs, list(_buf_wz_act), "mediumpurple", ls="-",  lw=1.5, label="gyro")
    ax_vx.legend(fontsize=6, loc="upper right")
    ax_vy.legend(fontsize=6, loc="upper right")
    ax_wz.legend(fontsize=6, loc="upper right")

    # ── Rows 1–4: torques ───────────────────────────────────────────
    _JOINT_TYPES = ["hip", "thigh", "calf"]
    _LS = ["-", "--", ":"]
    tau_lines = []

    for leg_i, (leg_name, color) in enumerate(zip(LEG_NAMES, _LEG_COLORS)):
        row_lines = []
        for jt_i, (jtype, ls) in enumerate(zip(_JOINT_TYPES, _LS)):
            ax = fig.add_subplot(gs_layout[leg_i + 1, jt_i])
            _setup_ax(ax, f"{leg_name}  {jtype}", "τ (N·m)", (-40, 40))
            joint_idx = leg_i * 3 + jt_i
            ln, = ax.plot(xs, list(_buf_torques[joint_idx]),
                          color=color, ls=ls, lw=1.2)
            row_lines.append(ln)
        tau_lines.append(row_lines)

    plt.ion()
    plt.show(block=False)

    interval = 1.0 / PLOT_HZ
    while _plot_running.is_set():
        with _plot_lock:
            vx_d = list(_buf_vx_cmd)
            vy_d = list(_buf_vy_cmd)
            wc_d = list(_buf_wz_cmd)
            wa_d = list(_buf_wz_act)
            t_d  = [list(b) for b in _buf_torques]

        ln_vx_cmd.set_ydata(vx_d)
        ln_vy_cmd.set_ydata(vy_d)
        ln_wz_cmd.set_ydata(wc_d)
        ln_wz_act.set_ydata(wa_d)
        for leg_i in range(4):
            for jt_i in range(3):
                tau_lines[leg_i][jt_i].set_ydata(t_d[leg_i * 3 + jt_i])

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(interval)

    plt.close(fig)


# ============================================================
# RECORDER THREAD — 500 Hz torque / state capture
# ============================================================
def _recorder_thread_fn(latest, t_start):
    interval  = 1.0 / RECORD_HZ
    next_tick = time.monotonic()
    while _recorder_running.is_set():
        now        = time.monotonic()
        sleep_time = next_tick - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_tick += interval

        t       = time.monotonic() - t_start
        raw     = lowstate_to_raw(latest["msg"])
        wz_act  = raw["imu"]["gyro_rad_s"][2]
        taus    = [m["tau_est"]  for m in raw["motors"]]
        dof_pos = [m["q_rad"]   for m in raw["motors"]]
        dof_vel = [m["dq_rad_s"] for m in raw["motors"]]
        with _plot_lock:
            vx_cmd, vy_cmd, wz_cmd = _cmd_state[0], _cmd_state[1], _cmd_state[2]
        _push_to_history(t, vx_cmd, vy_cmd, wz_cmd, wz_act, taus, dof_pos, dof_vel)


# ============================================================
# SAVE NPZ
# ============================================================
def save_npz(vx, vy, wz):
    if not _hist_time:
        print("  No data recorded — skipping save.")
        return
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"go2_robot_walk_vx{vx:+.2f}_vy{vy:+.2f}_wz{wz:+.2f}_{ts}.npz"
    fpath = os.path.join(SAVE_DIR, fname)
    np.savez(
        fpath,
        time        = np.array(_hist_time,    dtype=np.float32),
        vx_cmd      = np.array(_hist_vx_cmd,  dtype=np.float32),
        vy_cmd      = np.array(_hist_vy_cmd,  dtype=np.float32),
        wz_cmd      = np.array(_hist_wz_cmd,  dtype=np.float32),
        wz_act      = np.array(_hist_wz_act,  dtype=np.float32),
        torques     = np.array(_hist_torques, dtype=np.float32),   # (12, T)
        dof_pos     = np.array(_hist_dof_pos, dtype=np.float32),   # (12, T)
        dof_vel     = np.array(_hist_dof_vel, dtype=np.float32),   # (12, T)
        joint_names = np.array(JOINT_NAMES),
        preset_vx   = np.float32(vx),
        preset_vy   = np.float32(vy),
        preset_wz   = np.float32(wz),
    )
    print(f"\nSaved {len(_hist_time)} steps → {fpath}")
    return fpath


# ============================================================
# DDS HELPERS
# ============================================================
def init_dds(iface=None):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if iface:
        ChannelFactoryInitialize(0, iface)
        print(f"DDS interface: {iface}")
    else:
        ChannelFactoryInitialize(0)
        print("DDS interface: default")


def wait_for_lowstate():
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    latest = {"msg": None}
    def cb(msg: LowState_):
        latest["msg"] = msg
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(cb, 10)
    print("Waiting for rt/lowstate ...")
    while latest["msg"] is None:
        time.sleep(0.05)
    print("Got first lowstate.")
    return latest


def release_highlevel():
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    sc  = SportClient();  sc.SetTimeout(5.0);  sc.Init()
    msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
    print("Releasing high-level control ...")
    status, result = msc.CheckMode()
    while result.get("name"):
        sc.StandDown()
        msc.ReleaseMode()
        time.sleep(1.0)
        status, result = msc.CheckMode()
    print("High-level control released.")


# ============================================================
# MAIN RUN
# ============================================================
def run(policy, vx, vy, wz, duration, iface=None):
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread
    import unitree_legged_const as go2

    init_dds(iface)
    latest = wait_for_lowstate()
    release_highlevel()

    # ── Publisher setup ──────────────────────────────────────────────
    pub     = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    crc     = CRC()
    low_cmd = unitree_go_msg_dds__LowCmd_()
    low_cmd.head[0] = 0xFE
    low_cmd.head[1] = 0xEF
    low_cmd.level_flag = 0xFF
    low_cmd.gpio = 0
    for i in range(20):
        low_cmd.motor_cmd[i].mode = 0x01
        low_cmd.motor_cmd[i].q   = go2.PosStopF
        low_cmd.motor_cmd[i].dq  = go2.VelStopF
        low_cmd.motor_cmd[i].kp  = 0.0
        low_cmd.motor_cmd[i].kd  = 0.0
        low_cmd.motor_cmd[i].tau = 0.0

    shared = {
        "target_q":    None,
        "kp_per_joint": torch.full((12,), STAND_KP, dtype=torch.float32),
        "kd_per_joint": torch.full((12,), STAND_KD, dtype=torch.float32),
    }

    raw0    = lowstate_to_raw(latest["msg"])
    start_q = torch.tensor([m["q_rad"] for m in raw0["motors"]], dtype=torch.float32)
    shared["target_q"] = start_q.clone()

    def write_lowcmd():
        tq     = shared["target_q"]
        kp_arr = shared["kp_per_joint"]
        kd_arr = shared["kd_per_joint"]
        for i in range(12):
            low_cmd.motor_cmd[i].mode = 0x01
            low_cmd.motor_cmd[i].q   = float(tq[i])
            low_cmd.motor_cmd[i].dq  = 0.0
            low_cmd.motor_cmd[i].kp  = float(kp_arr[i])
            low_cmd.motor_cmd[i].kd  = float(kd_arr[i])
            low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.crc = crc.Crc(low_cmd)
        pub.Write(low_cmd)

    writer = RecurrentThread(interval=1.0 / LOWCMD_HZ, target=write_lowcmd, name="lowcmd_writer")
    writer.Start()
    print(f"LowCmd writer started at {LOWCMD_HZ} Hz.")

    # ── Stand ramp ───────────────────────────────────────────────────
    print("\nRamping to STAND pose ...")
    ramp_steps = max(1, int(STAND_SECONDS * POLICY_HZ))
    prev_q = start_q.clone()
    for k in range(ramp_steps):
        alpha   = (k + 1) / float(ramp_steps)
        desired = slew_limit((1 - alpha) * start_q + alpha * DEFAULT_DOF_POS, prev_q, MAX_STEP_RAD)
        shared["target_q"] = desired.clone()
        prev_q = desired.clone()
        time.sleep(1.0 / POLICY_HZ)
    print("Stand pose reached.")

    # ── Safety confirmation ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  PRESET: vx={vx:+.3f} m/s  vy={vy:+.3f} m/s  wz={wz:+.3f} rad/s")
    print(f"  DURATION: {duration:.1f} s  ({int(duration * POLICY_HZ)} steps)")
    print(f"  PLS: {'ENABLED' if PLS_ENABLE else 'DISABLED'}")
    print("  Type 'go' and press Enter to start. Anything else aborts.")
    print(f"{'='*55}")
    if input("> ").strip().lower() != "go":
        print("Aborted.")
        _plot_running.clear()
        return

    # ── Start 500 Hz recorder ────────────────────────────────────────
    t_start = time.monotonic()
    _recorder_running.set()
    rec_thread = threading.Thread(
        target=_recorder_thread_fn, args=(latest, t_start), daemon=True,
    )
    rec_thread.start()
    print(f"  Recorder started at {RECORD_HZ:.0f} Hz.")

    # ── Start live plot ──────────────────────────────────────────────
    plot_thread = threading.Thread(
        target=_plot_thread_fn, args=(vx, vy, wz, duration), daemon=True,
    )
    plot_thread.start()

    # ── Policy phase ─────────────────────────────────────────────────
    print(f"\nPolicy running for {duration:.1f} s ...")
    shared["kp_per_joint"][:] = POLICY_KP_FALLBACK
    shared["kd_per_joint"][:] = POLICY_KD_FALLBACK
    _heading_state["locked"]     = False
    _heading_state["target_yaw"] = None

    last_action   = torch.zeros(NUM_ACT, dtype=torch.float32)
    prev_target_q = DEFAULT_DOF_POS.clone()

    dt         = 1.0 / POLICY_HZ
    max_steps  = int(duration * POLICY_HZ)
    next_tick  = time.monotonic()

    try:
        for step in range(max_steps):
            now        = time.monotonic()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_tick += dt

            raw     = lowstate_to_raw(latest["msg"])
            command = make_command_list(raw, vx, vy, wz)
            obs     = build_obs(raw, command, last_action)

            with torch.no_grad():
                action_raw = policy.act_inference(obs.unsqueeze(0)).squeeze(0)

            action_clip   = torch.clamp(action_raw, -ACTION_CLIP, ACTION_CLIP)
            pos_action    = action_clip[:NUM_POS_ACTIONS]
            policy_target = DEFAULT_DOF_POS + ACTION_SCALE * pos_action
            target_q      = slew_limit(prev_target_q, policy_target, MAX_STEP_RAD)
            prev_target_q = target_q.clone()
            shared["target_q"] = target_q.clone()

            if PLS_ENABLE and action_clip.shape[0] > NUM_POS_ACTIONS:
                kp_12, kd_12 = compute_pls_kp_kd(action_clip[NUM_POS_ACTIONS:])
                shared["kp_per_joint"] = kp_12.clone()
                shared["kd_per_joint"] = kd_12.clone()

            last_action = action_clip.clone()

            # ── Update plot buffers and shared command (recorder handles history at 500 Hz) ──
            t      = step * dt
            wz_act = raw["imu"]["gyro_rad_s"][2]
            taus   = [m["tau_est"] for m in raw["motors"]]
            with _plot_lock:
                _cmd_state[0] = command[0]
                _cmd_state[1] = command[1]
                _cmd_state[2] = command[2]
            _push_to_plot(command[0], command[1], command[2], wz_act, taus)

            if step % 50 == 0:
                pct = 100.0 * step / max_steps
                print(
                    f"\r[{step:5d}/{max_steps}  {t:5.1f}s  {pct:4.1f}%]"
                    f"  cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}]"
                    f"  wz_gyro={wz_act:+.3f}  tau0={taus[0]:+.1f}N·m  ",
                    end="", flush=True,
                )

    except KeyboardInterrupt:
        print("\n  Interrupted early.")

    # ── Stop 500 Hz recorder ─────────────────────────────────────────
    _recorder_running.clear()
    rec_thread.join(timeout=2.0)

    # ── Transition back to stand ─────────────────────────────────────
    print("\nReturning to STAND pose ...")
    trans_steps = max(1, int(TRANSITION_SECS * POLICY_HZ))
    stand_start = prev_target_q.clone()
    shared["kp_per_joint"][:] = POLICY_KP_FALLBACK
    shared["kd_per_joint"][:] = POLICY_KD_FALLBACK
    for k in range(trans_steps):
        alpha   = (k + 1) / float(trans_steps)
        desired = slew_limit((1 - alpha) * stand_start + alpha * DEFAULT_DOF_POS,
                             prev_target_q, MAX_STEP_RAD)
        shared["kp_per_joint"] = ((1 - alpha) * torch.full((12,), POLICY_KP_FALLBACK)
                                  + alpha * torch.full((12,), STAND_KP))
        shared["kd_per_joint"] = ((1 - alpha) * torch.full((12,), POLICY_KD_FALLBACK)
                                  + alpha * torch.full((12,), STAND_KD))
        shared["target_q"] = desired.clone()
        prev_target_q = desired.clone()
        time.sleep(dt)
    print("Stand pose restored.")

    # ── Hold stand until user quits ─────────────────────────────────
    shared["target_q"]       = DEFAULT_DOF_POS.clone()
    shared["kp_per_joint"][:] = STAND_KP
    shared["kd_per_joint"][:] = STAND_KD
    print("\nHolding STAND pose. Press Ctrl+C to exit and save data.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # ── Safe stop (only on explicit Ctrl+C exit) ─────────────────────
    print("\nSending safe-stop packets ...")
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

    _plot_running.clear()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Go2 real-robot walk with preset velocity.")
    parser.add_argument("iface",      nargs="?",  default=None,
                        help="Network interface (e.g. eth0); omit for default DDS")
    parser.add_argument("--vx",       type=float, default=PRESET_VX,
                        help=f"Forward velocity m/s  (default {PRESET_VX})")
    parser.add_argument("--vy",       type=float, default=PRESET_VY,
                        help=f"Lateral velocity m/s  (default {PRESET_VY})")
    parser.add_argument("--wz",       type=float, default=PRESET_WZ,
                        help=f"Yaw rate rad/s         (default {PRESET_WZ})")
    parser.add_argument("--duration", type=float, default=PRESET_DURATION_S,
                        help=f"Policy run time s      (default {PRESET_DURATION_S})")
    parser.add_argument("--ckpt",     type=str,   default=CKPT_PATH,
                        help="Path to walk.pt")
    args = parser.parse_args()

    vx, vy, wz = args.vx, args.vy, args.wz

    print(f"\n{'='*55}")
    print("  Go2 Policy Walk — Preset Velocity")
    print(f"  vx={vx:+.3f}  vy={vy:+.3f}  wz={wz:+.3f}   duration={args.duration:.1f} s")
    print(f"  PLS: {'ENABLED' if PLS_ENABLE else 'DISABLED'}  "
          f"KP_FACTOR={KP_FACTOR}  KD_FACTOR={KD_FACTOR}")
    print(f"  Save dir: {SAVE_DIR}")
    print(f"{'='*55}\n")

    policy = load_policy(args.ckpt)

    try:
        run(policy, vx, vy, wz, args.duration, iface=args.iface)
    finally:
        save_npz(vx, vy, wz)
        print("Done.")


if __name__ == "__main__":
    main()
