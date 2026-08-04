#!/usr/bin/env python3
"""Replay the upstream Unitree G1 asset with the predictive CBF controller.

The runner loads the pinned G1 MJCF and mesh assets from the upstream
repository, injects a projectile and two cameras, and replays the planar
predictive perpendicular CBF command around the frozen G1 pose.  MuJoCo is
used as the portable physics and renderer backend; the CBF calculation is
NumPy-only so the replay is usable on AMD hosts without the upstream
CUDA-specific mjlab/MuJoCo-Warp stack.

This is an AMD-portable G1 asset replay and controller demonstration.  It is
not the upstream AMP training benchmark or a replacement for the official
G1 deployment path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


UPSTREAM_COMMIT = "2d4266978805e8272daa7f029a8bca91cf45e1ba"
DT = 0.02
SAFE_RADIUS = 0.55
BALL_RADIUS = 0.08
ROBOT_RADIUS = 0.28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scene(xml_path: Path) -> str:
    """Add the projectile, marker, lights, and replay cameras to the G1 XML."""
    source = xml_path.read_text(encoding="utf-8")
    source = source.replace(
        'meshdir="assets"',
        f'meshdir="{(xml_path.parent / "assets").resolve()}"',
        1,
    )
    prefix, final_worldbody = source.rsplit("<worldbody>", 1)
    extra = """
    <light name="replay_key" pos="2 -3 5" dir="-0.2 0.3 -1" diffuse="0.9 0.9 0.9"/>
    <light name="replay_fill" pos="-3 2 3" dir="0.4 -0.3 -1" diffuse="0.35 0.42 0.55"/>
    <geom name="replay_lane_left" type="box" pos="0 -1.8 0.015" size="4.8 0.02 0.015" rgba="0.15 0.72 0.72 0.65"/>
    <geom name="replay_lane_right" type="box" pos="0 1.8 0.015" size="4.8 0.02 0.015" rgba="0.15 0.72 0.72 0.65"/>
    <body name="dodge_ball" pos="-4 0 1.6">
      <joint name="dodge_ball_free" type="free"/>
      <geom name="dodge_ball_geom" type="sphere" size="0.08" rgba="0.96 0.25 0.35 1"/>
    </body>
    <body name="impact_marker" pos="0 0 0.04">
      <geom type="cylinder" size="0.42 0.025" rgba="0.96 0.25 0.35 0.20"/>
    </body>
    <camera name="replay_overview" pos="3.8 -5.2 2.6" euler="68 0 36" fovy="48"/>
    <camera name="replay_top" pos="0 -0.1 7.2" euler="0 0 0" fovy="52"/>
"""
    final_worldbody = final_worldbody.replace("</worldbody>", extra + "</worldbody>", 1)
    return prefix + "<worldbody>" + final_worldbody


def cbf_command(robot_xy: np.ndarray, ball_pos: np.ndarray, ball_vel: np.ndarray, side: float) -> tuple[np.ndarray, bool]:
    """Project a nominal planar command away from the predicted ball path."""
    speed = float(np.linalg.norm(ball_vel[:2]))
    if speed < 1e-6:
        return np.zeros(2, dtype=np.float64), False
    direction = ball_vel[:2] / speed
    normal = np.array([-direction[1], direction[0]])
    to_robot = robot_xy - ball_pos[:2]
    along = float(np.dot(to_robot, direction))
    airborne = ball_pos[2] > BALL_RADIUS + 0.05
    threat = airborne and speed > 0.5 and 0.0 < along < 6.0
    nominal = np.array([0.0, -0.16 * robot_xy[1]], dtype=np.float64)
    if not threat:
        return nominal, False
    escape = side * normal
    signed_side = float(np.dot(to_robot, escape))
    clearance = SAFE_RADIUS - signed_side
    time_to_impact = along / (speed + 1e-6)
    usable = max(time_to_impact - 0.25 - 0.15, 0.10)
    required = max(2.0 * clearance, clearance / usable)
    correction = max(required - float(np.dot(nominal, escape)), 0.0)
    command = nominal + correction * escape
    norm = float(np.linalg.norm(command))
    if norm > 1.2:
        command *= 1.2 / norm
    return command, True


def draw_overlay(frame: np.ndarray, title: str, step: int, threat: bool, clearance: float) -> np.ndarray:
    """Add compact replay metadata without covering the G1 silhouette."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((18, 18, 430, 78), radius=12, fill=(8, 12, 18, 220))
    draw.text((32, 28), title, fill=(235, 242, 248, 255))
    state = "CBF ACTIVE" if threat else "MONITORING"
    color = (72, 212, 180, 255) if threat else (247, 194, 92, 255)
    draw.text((32, 51), f"{state}  |  frame {step:03d}  |  clearance {clearance:.2f} m", fill=color)
    return np.asarray(image)


def run_episode(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    overview_camera: mujoco.MjvCamera,
    top_camera: mujoco.MjvCamera,
    seed: int,
    duration: float,
) -> tuple[dict, list[np.ndarray]]:
    """Replay one projectile and return CBF metrics plus synchronized views."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    robot_joint = model.joint("floating_base_joint")
    ball_joint = model.joint("dodge_ball_free")
    robot_qpos = int(robot_joint.qposadr[0])
    ball_qpos = int(ball_joint.qposadr[0])
    robot_xy = np.array([0.0, 0.0], dtype=np.float64)
    data.qpos[robot_qpos : robot_qpos + 7] = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]
    flight_time = 1.45
    start = np.array([-4.4, rng.normal(0.0, 0.04), 1.68 + rng.uniform(-0.04, 0.05)], dtype=np.float64)
    target_z = 1.05
    velocity = np.array([
        (0.0 - start[0]) / flight_time,
        rng.normal(0.0, 0.025),
        (target_z - start[2] + 4.905 * flight_time * flight_time) / flight_time,
    ])
    side = 1.0 if seed % 2 == 0 else -1.0
    frames: list[np.ndarray] = []
    min_clearance = float("inf")
    threat_steps = 0
    max_speed = 0.0
    for step in range(round(duration / DT)):
        time = step * DT
        ball_pos = start + velocity * time + np.array([0.0, 0.0, -4.905 * time * time])
        ball_vel = velocity + np.array([0.0, 0.0, -9.81 * time])
        command, threat = cbf_command(robot_xy, ball_pos, ball_vel, side)
        robot_xy += command * DT
        data.qpos[robot_qpos : robot_qpos + 3] = [robot_xy[0], robot_xy[1], 0.793]
        data.qpos[robot_qpos + 3 : robot_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[ball_qpos : ball_qpos + 3] = ball_pos
        data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        clearance = float(np.linalg.norm(np.array([robot_xy[0], robot_xy[1], 0.793]) - ball_pos) - ROBOT_RADIUS - BALL_RADIUS)
        min_clearance = min(min_clearance, clearance)
        threat_steps += int(threat)
        max_speed = max(max_speed, float(np.linalg.norm(command)))
        if step % 2 == 0:
            renderer.update_scene(data, camera=overview_camera)
            overview = renderer.render().copy()
            renderer.update_scene(data, camera=top_camera)
            top = renderer.render().copy()
            frame = np.concatenate((overview, top), axis=1)
            frames.append(draw_overlay(frame, "Unitree G1 · predictive CBF replay", step, threat, clearance))
    return {
        "seed": seed,
        "clearance_preserved": min_clearance > 0.0,
        "min_clearance_m": round(min_clearance, 5),
        "max_filtered_speed_mps": round(max_speed, 5),
        "threat_steps": threat_steps,
        "escape_side": int(side),
        "evaluation_type": "upstream_g1_asset_portable_cbf_replay",
    }, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pacman_g1_amd_replay"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--duration", type=float, default=2.2)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_scene(args.upstream_xml))
    renderer = mujoco.Renderer(model, height=420, width=640)
    overview_camera = mujoco.MjvCamera()
    overview_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    overview_camera.lookat[:] = [0.0, 0.0, 0.85]
    overview_camera.distance = 4.8
    overview_camera.azimuth = 135.0
    overview_camera.elevation = -18.0
    top_camera = mujoco.MjvCamera()
    top_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    top_camera.lookat[:] = [0.0, 0.0, 0.0]
    top_camera.distance = 4.8
    top_camera.azimuth = 90.0
    top_camera.elevation = -88.0
    episodes: list[dict] = []
    first_frames: list[np.ndarray] = []
    for seed in range(args.episodes):
        result, frames = run_episode(model, renderer, overview_camera, top_camera, seed, args.duration)
        episodes.append(result)
        if seed == 0:
            first_frames = frames

    video_path = args.output_dir / "unitree-g1-predictive-cbf-amd-replay.mp4"
    imageio.mimsave(video_path, first_frames, fps=args.fps, macro_block_size=1)
    successes = sum(int(item["clearance_preserved"]) for item in episodes)
    result = {
        "schema": "amd-pacman-g1-asset-replay/v1",
        "status": "g1_asset_replay_complete",
        "upstream_repository": "https://github.com/lzyang2000/perceptive_cbf_rl",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_xml": str(args.upstream_xml),
        "runtime": {
            "backend": "MuJoCo 3.11 portable renderer + NumPy CBF",
            "platform": os.uname().sysname,
            "official_cuda_mjlab_stack": False,
        },
        "task": "Unitree G1 predictive perpendicular CBF obstacle-avoidance replay",
        "protocol": {
            "episodes": args.episodes,
            "seeds": list(range(args.episodes)),
            "duration_s": args.duration,
            "views": ["replay_overview", "replay_top"],
        },
        "summary": {
            "clearance_preserved": successes,
            "episodes": args.episodes,
            "min_clearance_m": min(item["min_clearance_m"] for item in episodes),
        },
        "episodes": episodes,
        "artifacts": {"video": video_path.name},
        "boundaries": [
            "The replay uses the upstream G1 MJCF and mesh assets.",
            "The portable path does not run upstream AMP training or MuJoCo-Warp.",
            "The result is a controller and rendering demonstration, not the upstream paper benchmark.",
        ],
    }
    (args.output_dir / "eval_info.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(os.sys.argv),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_xml_sha256": sha256_file(args.upstream_xml),
        "video_sha256": sha256_file(video_path),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "video": str(video_path), "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
