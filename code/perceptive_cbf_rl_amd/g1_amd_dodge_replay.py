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
SAFE_RADIUS = 0.55
BALL_RADIUS = 0.08
ROBOT_RADIUS = 0.28
GHOST_COUNT = 12
ROOT_Z = 0.793


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_scene(xml_path: Path) -> str:
    """Add cinematic replay geometry while preserving the pinned G1 assets."""
    source = xml_path.read_text(encoding="utf-8")
    source = source.replace(
        'meshdir="assets"',
        f'meshdir="{(xml_path.parent / "assets").resolve()}"',
        1,
    )
    source = source.replace(
        '<global azimuth="-140" elevation="-20"/>',
        '<global azimuth="-140" elevation="-20" offwidth="1280" offheight="720"/>',
        1,
    )
    prefix, final_worldbody = source.rsplit("<worldbody>", 1)
    ghost_bodies = "\n".join(
        f"""<body name="prediction_{index}" pos="-6 0 0.1">
      <joint name="prediction_{index}_free" type="free"/>
      <geom type="sphere" size="0.035" contype="0" conaffinity="0"
            rgba="1.0 0.62 0.18 {0.50 - index * 0.025:.3f}"/>
    </body>"""
        for index in range(GHOST_COUNT)
    )
    extra = f"""
    <light name="replay_key" pos="2 -3 5" dir="-0.2 0.3 -1" diffuse="0.9 0.9 0.9"/>
    <light name="replay_fill" pos="-3 2 3" dir="0.4 -0.3 -1" diffuse="0.35 0.42 0.55"/>
    <geom name="replay_lane_left" type="box" pos="0 -1.8 0.015" size="4.8 0.02 0.015" rgba="0.15 0.72 0.72 0.65"/>
    <geom name="replay_lane_right" type="box" pos="0 1.8 0.015" size="4.8 0.02 0.015" rgba="0.15 0.72 0.72 0.65"/>
    <body name="dodge_ball" pos="-4 0 1.6">
      <joint name="dodge_ball_free" type="free"/>
      <geom name="dodge_ball_geom" type="sphere" size="0.08" contype="0" conaffinity="0" rgba="0.96 0.25 0.35 1"/>
    </body>
    <body name="safety_disc" pos="0 0 0.035">
      <joint name="safety_disc_free" type="free"/>
      <geom type="cylinder" size="0.55 0.018" contype="0" conaffinity="0" rgba="0.22 0.72 0.65 0.20"/>
    </body>
    {ghost_bodies}
"""
    final_worldbody = final_worldbody.replace("</worldbody>", extra + "</worldbody>", 1)
    return prefix + "<worldbody>" + final_worldbody


def projectile_state(time: float, duration: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, shallow thrown-ball arc for one showcase seed."""
    phase = float(np.clip(time / duration, 0.0, 1.0))
    lateral = 0.04 * np.sin(seed * 1.7)
    start = np.array([-4.4, lateral, 1.48 + 0.03 * (seed % 3)], dtype=np.float64)
    end = np.array([2.25, lateral + 0.025 * np.cos(seed), 0.38], dtype=np.float64)
    position = (1.0 - phase) * start + phase * end
    position[2] += 0.72 * 4.0 * phase * (1.0 - phase)
    velocity = (end - start) / duration
    velocity[2] += 0.72 * 4.0 * (1.0 - 2.0 * phase) / duration
    return position, velocity


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
    nominal = np.array([0.0, -0.22 * robot_xy[1]], dtype=np.float64)
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
    if norm > 1.05:
        command *= 1.05 / norm
    return command, True


def apply_sidestep_pose(model: mujoco.MjModel, data: mujoco.MjData, command: np.ndarray, time: float) -> None:
    """Animate a readable kinematic sidestep pose around the official G1 rest pose."""
    strength = float(np.clip(abs(command[1]) / 1.05, 0.0, 1.0))
    direction = float(np.sign(command[1])) if strength > 0.02 else 0.0
    pulse = np.sin(time * np.pi * 2.4)
    offsets = {
        "left_hip_roll_joint": -0.17 * direction * strength,
        "right_hip_roll_joint": -0.13 * direction * strength,
        "left_ankle_roll_joint": 0.12 * direction * strength,
        "right_ankle_roll_joint": 0.10 * direction * strength,
        "left_hip_pitch_joint": -0.08 * strength + 0.035 * pulse * strength,
        "right_hip_pitch_joint": -0.08 * strength - 0.035 * pulse * strength,
        "left_knee_joint": 0.23 * strength - 0.04 * pulse * strength,
        "right_knee_joint": 0.23 * strength + 0.04 * pulse * strength,
        "left_ankle_pitch_joint": -0.12 * strength,
        "right_ankle_pitch_joint": -0.12 * strength,
        "waist_roll_joint": 0.09 * direction * strength,
        "left_shoulder_roll_joint": 0.12 * direction * strength,
        "right_shoulder_roll_joint": 0.12 * direction * strength,
        "left_shoulder_pitch_joint": 0.08 * pulse * strength,
        "right_shoulder_pitch_joint": -0.08 * pulse * strength,
    }
    for name, offset in offsets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            qpos_address = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_address] = model.qpos0[qpos_address] + offset


def draw_overlay(
    primary: np.ndarray,
    top: np.ndarray,
    segment: int,
    segments: int,
    seed: int,
    step: int,
    total_steps: int,
    threat: bool,
    clearance: float,
    camera_label: str,
) -> np.ndarray:
    """Compose a 720p evidence frame with camera inset and clearance telemetry."""
    image = Image.fromarray(primary)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    inset_width = 368
    inset_height = 207
    inset = Image.fromarray(top).resize((inset_width, inset_height), Image.Resampling.LANCZOS)
    inset_x = width - inset_width - 28
    inset_y = 28
    draw.rounded_rectangle((inset_x - 5, inset_y - 5, width - 23, inset_y + inset_height + 5), radius=14, fill=(8, 12, 18, 235))
    image.paste(inset, (inset_x, inset_y))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((24, 24, 545, 108), radius=14, fill=(8, 12, 18, 224))
    draw.text((42, 38), "Unitree G1 · AMD portable predictive-CBF replay", fill=(238, 244, 248, 255))
    state = "CBF ACTIVE" if threat else "MONITORING"
    color = (72, 212, 180, 255) if threat else (247, 194, 92, 255)
    draw.text((42, 67), f"{state}  ·  seed {seed}  ·  clearance {clearance:.2f} m", fill=color)
    draw.text((inset_x + 12, inset_y + 10), "TOP VIEW · trajectory + safety zone", fill=(238, 244, 248, 255))

    footer_top = height - 88
    draw.rounded_rectangle((24, footer_top, width - 24, height - 22), radius=12, fill=(8, 12, 18, 218))
    draw.text((42, footer_top + 13), f"SHOT {segment + 1}/{segments}  ·  {camera_label}", fill=(238, 244, 248, 255))
    draw.text((42, footer_top + 38), "RED projectile   AMBER predicted path   TEAL safety envelope", fill=(182, 196, 207, 255))
    progress_x0 = 455
    progress_x1 = width - 52
    progress_y = footer_top + 34
    progress = step / max(total_steps - 1, 1)
    draw.rounded_rectangle((progress_x0, progress_y, progress_x1, progress_y + 9), radius=4, fill=(60, 70, 78, 255))
    draw.rounded_rectangle((progress_x0, progress_y, progress_x0 + int((progress_x1 - progress_x0) * progress), progress_y + 9), radius=4, fill=(72, 212, 180, 255))
    return np.asarray(image)


def run_episode(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    primary_camera: mujoco.MjvCamera,
    top_camera: mujoco.MjvCamera,
    seed: int,
    duration: float,
    fps: int,
    segment: int,
    segments: int,
    camera_label: str,
    render_video: bool,
) -> tuple[dict, list[np.ndarray]]:
    """Replay one deterministic projectile and optionally render one segment."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    robot_joint = model.joint("floating_base_joint")
    ball_joint = model.joint("dodge_ball_free")
    disc_joint = model.joint("safety_disc_free")
    robot_qpos = int(robot_joint.qposadr[0])
    ball_qpos = int(ball_joint.qposadr[0])
    disc_qpos = int(disc_joint.qposadr[0])
    prediction_qpos = [int(model.joint(f"prediction_{index}_free").qposadr[0]) for index in range(GHOST_COUNT)]
    robot_xy = np.array([0.0, 0.0], dtype=np.float64)
    data.qpos[robot_qpos : robot_qpos + 7] = [0.0, 0.0, ROOT_Z, 1.0, 0.0, 0.0, 0.0]
    side = 1.0 if seed % 2 == 0 else -1.0
    frames: list[np.ndarray] = []
    min_clearance = float("inf")
    threat_steps = 0
    max_speed = 0.0
    max_lateral_displacement = 0.0
    total_steps = round(duration * fps)
    for step in range(total_steps):
        time = step / fps
        ball_pos, ball_vel = projectile_state(time, duration, seed)
        command, threat = cbf_command(robot_xy, ball_pos, ball_vel, side)
        robot_xy += command / fps
        data.qpos[robot_qpos : robot_qpos + 3] = [robot_xy[0], robot_xy[1], ROOT_Z]
        data.qpos[robot_qpos + 3 : robot_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[ball_qpos : ball_qpos + 3] = ball_pos
        data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[disc_qpos : disc_qpos + 3] = [robot_xy[0], robot_xy[1], 0.035]
        data.qpos[disc_qpos + 3 : disc_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        for ghost_index, ghost_qpos in enumerate(prediction_qpos):
            future_time = min(time + 0.12 * (ghost_index + 1), duration)
            ghost_pos, _ = projectile_state(future_time, duration, seed)
            data.qpos[ghost_qpos : ghost_qpos + 3] = ghost_pos
            data.qpos[ghost_qpos + 3 : ghost_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        apply_sidestep_pose(model, data, command, time)
        mujoco.mj_forward(model, data)
        clearance = float(np.linalg.norm(np.array([robot_xy[0], robot_xy[1], ROOT_Z]) - ball_pos) - ROBOT_RADIUS - BALL_RADIUS)
        min_clearance = min(min_clearance, clearance)
        threat_steps += int(threat)
        max_speed = max(max_speed, float(np.linalg.norm(command)))
        max_lateral_displacement = max(max_lateral_displacement, abs(float(robot_xy[1])))
        if render_video:
            primary_camera.lookat[:] = [robot_xy[0], robot_xy[1] * 0.35, 0.9]
            renderer.update_scene(data, camera=primary_camera)
            primary = renderer.render().copy()
            renderer.update_scene(data, camera=top_camera)
            top = renderer.render().copy()
            frames.append(
                draw_overlay(
                    primary,
                    top,
                    segment,
                    segments,
                    seed,
                    step,
                    total_steps,
                    threat,
                    clearance,
                    camera_label,
                )
            )
    return {
        "seed": seed,
        "clearance_preserved": min_clearance > 0.0,
        "min_clearance_m": round(min_clearance, 5),
        "max_filtered_speed_mps": round(max_speed, 5),
        "max_lateral_displacement_m": round(max_lateral_displacement, 5),
        "threat_steps": threat_steps,
        "escape_side": int(side),
        "duration_s": duration,
        "evaluation_type": "upstream_g1_asset_portable_cbf_replay",
    }, frames


def make_camera(azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    """Create a deterministic free camera for a replay segment."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.9]
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-xml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pacman_g1_amd_replay"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per cinematic segment")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--showcase-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    if not 12.0 <= args.duration * len(args.showcase_seeds) <= 20.0:
        raise SystemExit("cinematic duration must be between 12 and 20 seconds")
    if any(seed < 0 or seed >= args.episodes for seed in args.showcase_seeds):
        raise SystemExit("showcase seeds must be included in --episodes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_scene(args.upstream_xml))
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera_specs = [
        ("THREE-QUARTER", make_camera(132.0, -16.0, 5.2)),
        ("PROJECTILE PROFILE", make_camera(90.0, -10.0, 5.5)),
        ("LOW FRONT", make_camera(205.0, -8.0, 4.8)),
    ]
    top_camera = make_camera(90.0, -88.0, 5.4)
    showcase_order = {seed: index for index, seed in enumerate(args.showcase_seeds)}
    episodes: list[dict] = []
    video_segments: dict[int, list[np.ndarray]] = {}
    for seed in range(args.episodes):
        segment = showcase_order.get(seed)
        render_video = segment is not None
        camera_label, primary_camera = camera_specs[segment % len(camera_specs)] if render_video else camera_specs[0]
        result, frames = run_episode(
            model,
            renderer,
            primary_camera,
            top_camera,
            seed,
            args.duration,
            args.fps,
            segment or 0,
            len(args.showcase_seeds),
            camera_label,
            render_video,
        )
        episodes.append(result)
        if render_video:
            video_segments[segment] = frames

    video_path = args.output_dir / "unitree-g1-predictive-cbf-amd-replay.mp4"
    cinematic_frames = [frame for segment in range(len(args.showcase_seeds)) for frame in video_segments[segment]]
    imageio.mimsave(
        video_path,
        cinematic_frames,
        fps=args.fps,
        macro_block_size=1,
        quality=8,
        codec="libx264",
        ffmpeg_params=["-threads", "1", "-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"],
    )
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
            "showcase_seeds": args.showcase_seeds,
            "segment_duration_s": args.duration,
            "video_duration_s": args.duration * len(args.showcase_seeds),
            "video_resolution": [1280, 720],
            "views": [label for label, _ in camera_specs] + ["top-view inset"],
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
