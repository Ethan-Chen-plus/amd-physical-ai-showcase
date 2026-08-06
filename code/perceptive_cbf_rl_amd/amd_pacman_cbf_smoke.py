#!/usr/bin/env python3
"""AMD-side PAC-MAN control-path validation.

This executable keeps the upstream PAC-MAN predictive CBF projection, but
replaces the CUDA-only mjlab/MuJoCo-Warp simulator with a small CPU MuJoCo
scene. PyTorch performs the batched CBF tensor calculation on the selected
device, so an AMD ROCm run exercises the real ROCm tensor path while MuJoCo
remains a portable physics and rendering backend.

The result is intentionally a control-path validation, not a reproduction of
the upstream G1 AMP policy or its 19/20 hardware benchmark. The output JSON
records that boundary explicitly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path

# Select a headless backend before importing MuJoCo's renderer. EGL is
# available on the AMD cloud image; callers can override it with MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - the runtime can use OpenCV instead.
    imageio = None

try:
    import cv2
except Exception:  # pragma: no cover - imageio is the normal path.
    cv2 = None


UPSTREAM_COMMIT = "2d4266978805e8272daa7f029a8bca91cf45e1ba"
DT = 0.02
SAFE_RADIUS = 0.55
BALL_RADIUS = 0.08
ROBOT_RADIUS = 0.28
XML = r"""
<mujoco model="pacman_amd_cbf_proxy">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.02" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="360"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="256" height="256" rgb1="0.11 0.13 0.16" rgb2="0.17 0.20 0.24"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light name="key" pos="2 -3 5" dir="-0.2 0.3 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="-3 2 3" dir="0.4 -0.3 -1" diffuse="0.35 0.42 0.55"/>
    <geom name="floor" type="plane" size="12 12 0.1" material="floor_mat"/>
    <geom name="lane_left" type="box" pos="0 -1.8 0.015" size="4.8 0.02 0.015" rgba="0.15 0.72 0.72 0.65"/>
    <geom name="lane_right" type="box" pos="0 1.8 0.015" size="4.8 0.02 0.015" rgba="0.15 0.72 0.72 0.65"/>
    <body name="agent" mocap="true" pos="0 0 1.1">
      <geom name="torso" type="capsule" fromto="0 0 0.65 0 0 1.55" size="0.28" rgba="0.20 0.78 0.78 1"/>
      <geom name="head" type="sphere" pos="0 0 1.72" size="0.18" rgba="0.95 0.72 0.25 1"/>
      <site name="agent_center" pos="0 0 1.1" size="0.03" rgba="0.95 0.95 0.95 1"/>
    </body>
    <body name="ball" pos="-4 0 1.6">
      <joint name="ball_free" type="free"/>
      <geom name="ball_geom" type="sphere" size="0.08" rgba="0.96 0.25 0.35 1"/>
    </body>
    <body name="impact_marker" pos="0 0 0.04">
      <geom type="cylinder" size="0.42 0.025" rgba="0.96 0.25 0.35 0.20"/>
    </body>
    <camera name="overview" pos="4.8 -6.6 3.8" euler="67 0 36" fovy="48"/>
    <camera name="top" pos="0 -0.1 7.2" euler="0 0 0" fovy="52"/>
  </worldbody>
</mujoco>
"""


def predictive_dodge_filter(
    u_nom: torch.Tensor,
    robot_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    side: torch.Tensor,
    *,
    safe_radius: float = SAFE_RADIUS,
    ball_radius: float = BALL_RADIUS,
    alpha: float = 2.0,
    sense_radius: float = 6.0,
    momentum_time: float = 0.25,
    comfort_buffer: float = 0.15,
    t_floor: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the upstream predictive perpendicular CBF projection.

    Inputs are batched world-frame positions and velocities. The nominal
    command is projected onto the half-space required to clear the predicted
    horizontal ball path. The return values are safe command, threat mask,
    barrier value, time-to-impact, and escape direction.
    """
    rp = robot_pos[:, :2]
    bp = ball_pos[:, :2]
    vb = ball_vel[:, :2]
    speed = torch.linalg.vector_norm(vb, dim=-1)
    vdir = vb / (speed[:, None] + 1e-6)
    normal = torch.stack((-vdir[:, 1], vdir[:, 0]), dim=-1)
    to_robot = rp - bp
    along = torch.sum(to_robot * vdir, dim=-1)
    perp = torch.sum(to_robot * normal, dim=-1)
    airborne = ball_pos[:, 2] > ball_radius + 0.05
    threat = airborne & (speed > 0.5) & (along > 0.0) & (along < sense_radius)

    escape = side[:, None] * normal
    signed_side = torch.sum(to_robot * escape, dim=-1)
    clearance = safe_radius - signed_side
    barrier = signed_side - safe_radius
    time_to_impact = along / (speed + 1e-6)
    usable = torch.clamp(time_to_impact - momentum_time - comfort_buffer, min=t_floor)
    required = torch.maximum(alpha * clearance, clearance / usable)
    nominal_projection = torch.sum(u_nom * escape, dim=-1)
    correction = torch.clamp(required - nominal_projection, min=0.0)
    safe = u_nom + correction[:, None] * escape
    safe = torch.where(threat[:, None], safe, u_nom)
    return safe, threat, barrier, time_to_impact, escape


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "rocm":
        if not torch.cuda.is_available():
            raise RuntimeError("--device rocm requested, but torch.cuda is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_ball_state(data: mujoco.MjData, start: np.ndarray, velocity: np.ndarray) -> None:
    data.qpos[:3] = start
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[:3] = velocity
    data.qvel[3:6] = 0.0


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if imageio is not None:
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return
    if cv2 is None:
        raise RuntimeError("Neither imageio nor OpenCV is available for video output")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def render_frame(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    renderer = render_frame.renderer
    renderer.update_scene(data, camera="overview")
    overview = renderer.render().copy()
    renderer.update_scene(data, camera="top")
    top = renderer.render().copy()
    return np.concatenate((overview, top), axis=1)


def run_episode(model: mujoco.MjModel, device: torch.device, seed: int, render: bool) -> tuple[dict, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    agent_pos = np.array([0.0, 0.0, 1.1], dtype=np.float32)
    side_value = 1.0 if (seed % 2 == 0) else -1.0
    start = np.array([-4.2, rng.normal(0.0, 0.015), 1.55 + rng.uniform(-0.04, 0.05)], dtype=np.float64)
    duration = 2.2
    target_z = 1.10
    flight_time = 1.0
    gravity = -9.81
    velocity = np.array([
        (0.0 - start[0]) / flight_time,
        rng.normal(0.0, 0.02),
        (target_z - start[2] - 0.5 * gravity * flight_time * flight_time) / flight_time,
    ])
    set_ball_state(data, start, velocity)
    mocap_id = int(model.body("agent").mocapid[0])
    data.mocap_pos[mocap_id] = agent_pos
    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    frames: list[np.ndarray] = []
    min_clearance = float("inf")
    max_speed = 0.0
    threat_steps = 0
    collision = False
    for _ in range(round(duration / DT)):
        ball_pos_np = data.qpos[:3].copy()
        ball_vel_np = data.qvel[:3].copy()
        robot_t = torch.as_tensor(agent_pos[None, :], dtype=torch.float32, device=device)
        ball_t = torch.as_tensor(ball_pos_np[None, :], dtype=torch.float32, device=device)
        velocity_t = torch.as_tensor(ball_vel_np[None, :], dtype=torch.float32, device=device)
        nominal_t = torch.zeros((1, 2), dtype=torch.float32, device=device)
        side_t = torch.tensor([side_value], dtype=torch.float32, device=device)
        safe_t, threat_t, barrier_t, tti_t, escape_t = predictive_dodge_filter(
            nominal_t, robot_t, ball_t, velocity_t, side_t
        )
        safe_np = safe_t.detach().cpu().numpy()[0]
        speed = float(np.linalg.norm(safe_np))
        if speed > 1.6:
            safe_np *= 1.6 / speed
            speed = 1.6
        agent_pos[:2] += safe_np * DT
        data.mocap_pos[mocap_id] = agent_pos
        mujoco.mj_step(model, data)

        distance = float(np.linalg.norm(agent_pos - data.qpos[:3]))
        clearance = distance - (ROBOT_RADIUS + BALL_RADIUS)
        min_clearance = min(min_clearance, clearance)
        collision |= clearance <= 0.0
        threat_steps += int(threat_t.item())
        max_speed = max(max_speed, speed)
        if render and (_ % 2 == 0):
            frames.append(render_frame(model, data))

    result = {
        "seed": seed,
        "success": not collision,
        "collision": collision,
        "min_clearance_m": round(min_clearance, 5),
        "max_filtered_speed_mps": round(max_speed, 5),
        "threat_steps": threat_steps,
        "escape_side": int(side_value),
        "evaluation_type": "predictive_cbf_control_path_proxy",
    }
    return result, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/perceptive_cbf_rl_amd"))
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "rocm", "cpu"), default="auto")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(XML)
    render_frame.renderer = mujoco.Renderer(model, height=270, width=480)

    episodes: list[dict] = []
    all_frames: list[np.ndarray] = []
    for seed in range(args.episodes):
        result, frames = run_episode(model, device, seed, render=not args.no_video)
        episodes.append(result)
        if seed == 0:
            all_frames = frames

    successes = sum(int(item["success"]) for item in episodes)
    result = {
        "schema": "amd-pacman-cbf-control-validation/v1",
        "status": "control_path_validated",
        "upstream_repository": "https://github.com/lzyang2000/perceptive_cbf_rl",
        "upstream_commit": UPSTREAM_COMMIT,
        "task": "PAC-MAN predictive perpendicular CBF on a MuJoCo projectile proxy",
        "evaluation_scope": {
            "controller": "predictive_perpendicular_cbf",
            "scene": "portable_mujoco_projectile",
            "is_rocm_policy_tensor_path": device.type == "cuda",
            "evidence": "fixed_seed_clearance_and_video",
        },
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "torch_device": str(device),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "mujoco": mujoco.__version__,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "protocol": {
            "episodes": args.episodes,
            "seed_policy": "seed 0..N-1",
            "physics_timestep_s": DT,
            "camera_views": ["overview", "top"],
            "safe_radius_m": SAFE_RADIUS,
            "video_fps": 25,
        },
        "summary": {
            "successes": successes,
            "episodes": len(episodes),
            "success_rate": successes / len(episodes),
            "all_proxy_episodes_safe": successes == len(episodes),
        },
        "episodes": episodes,
    }
    (args.output_dir / "eval_info.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "command": " ".join(sys.argv),
                "script": "amd_pacman_cbf_smoke.py",
                "upstream_commit": UPSTREAM_COMMIT,
                "output": "eval_info.json",
                "video": None if args.no_video else "pacman-cbf-amd-proxy.mp4",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if all_frames and not args.no_video:
        write_video(all_frames, args.output_dir / "pacman-cbf-amd-proxy.mp4", fps=25)

    print(json.dumps({"output_dir": str(args.output_dir), "device": str(device), **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
