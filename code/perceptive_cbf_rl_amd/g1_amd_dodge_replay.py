#!/usr/bin/env python3
"""Run the upstream G1 dodge ONNX contract in a portable MuJoCo replay.

The replay uses the pinned upstream ``g1.xml`` model, the shipped
``dodge_link_cbf.onnx`` actor, the exact 960-D observation layout, and the
official 29-joint action scaling. A synthetic ball-only depth source replaces
the ZED/EfficientTAM camera while preserving its 9x16 normalized depth and
temporal offsets. The free base is held in place because the plain MuJoCo MJCF
does not include the upstream mjlab actuator and balance stack; every policy
joint target is still applied and rendered over time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


UPSTREAM_REPOSITORY = "https://github.com/lzyang2000/perceptive_cbf_rl"
PINNED_UPSTREAM_COMMIT = "2d4266978805e8272daa7f029a8bca91cf45e1ba"
FRAME_OFFSETS = (0, 3, 8, 18)
CONTROL_HZ = 50
DEPTH_HEIGHT = 9
DEPTH_WIDTH = 16
BALL_RADIUS = 0.0762
ROOT_POSE = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
MOVEMENT_THRESHOLD_RAD = 0.05
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    ball_start: tuple[float, float, float]
    ball_end: tuple[float, float, float]


SCENARIOS = (
    Scenario("high_center", "HEAD-HEIGHT DUCK", (4.5, 0.00, 1.43), (-0.8, 0.00, 1.43)),
    Scenario("left_shoulder", "LEFT-SHOULDER WEAVE", (4.5, 0.27, 1.18), (-0.8, 0.27, 1.18)),
    Scenario("right_low", "RIGHT-LOW WHOLE-BODY DODGE", (4.5, -0.24, 0.88), (-0.8, -0.24, 0.88)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_upstream(upstream_root: Path) -> dict[str, Any]:
    """Load the deployment contract directly from the pinned vendor checkout."""
    root = upstream_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from deploy.common.g1_deploy_constants import (  # type: ignore
        ACTION_SCALE,
        DEFAULT_POS,
        DEPTH_FAR,
        DEPTH_NEAR,
        KD,
        KP,
        NUM_JOINTS,
        POLICY_JOINT_NAMES,
    )
    from deploy.policy.dodge_policy import (  # type: ignore
        DepthRing,
        ProprioHistory,
        assemble_obs,
        depth_metres_to_obs,
    )

    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    checkpoint = root / "deploy/ckpts/dodge_link_cbf.onnx"
    robot_xml = root / "src/assets/robots/unitree_g1/xmls/g1.xml"
    generated_scene_xml = root / "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
    for path in (checkpoint, robot_xml, generated_scene_xml):
        if not path.is_file():
            raise FileNotFoundError(path)

    return {
        "root": root,
        "commit": commit,
        "checkpoint": checkpoint,
        "robot_xml": robot_xml,
        "generated_scene_xml": generated_scene_xml,
        "action_scale": ACTION_SCALE.astype(np.float32),
        "default_pos": DEFAULT_POS.astype(np.float32),
        "depth_near": float(DEPTH_NEAR),
        "depth_far": float(DEPTH_FAR),
        "kp": KP.astype(np.float64),
        "kd": KD.astype(np.float64),
        "num_joints": int(NUM_JOINTS),
        "joint_names": list(POLICY_JOINT_NAMES),
        "DepthRing": DepthRing,
        "ProprioHistory": ProprioHistory,
        "assemble_obs": assemble_obs,
        "depth_metres_to_obs": depth_metres_to_obs,
    }


def build_replay_scene(robot_xml: Path, ghost_count: int = 10) -> str:
    """Inject only replay geometry into the official G1 MJCF."""
    source = robot_xml.read_text(encoding="utf-8")
    source = source.replace(
        'meshdir="assets"', f'meshdir="{(robot_xml.parent / "assets").resolve()}"', 1
    )
    source = source.replace(
        "<asset>",
        '<visual><global offwidth="960" offheight="540"/></visual>\n  <asset>',
        1,
    )
    ghosts = "\n".join(
        f"""<body name="prediction_{index}" pos="-8 0 -1">
      <freejoint name="prediction_{index}_free"/>
      <geom type="sphere" size="0.028" contype="0" conaffinity="0"
            rgba="1.0 0.61 0.15 {0.62 - 0.045 * index:.3f}"/>
    </body>"""
        for index in range(ghost_count)
    )
    extras = f"""
    <light name="key" pos="2.5 -3.5 5" dir="-0.3 0.35 -1" diffuse="0.95 0.95 0.95"/>
    <light name="fill" pos="-2.5 3 3.5" dir="0.35 -0.25 -1" diffuse="0.34 0.42 0.52"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.07 0.09 0.12 1"/>
    <geom name="lane_left" type="box" pos="0 1.25 0.012" size="4.6 0.016 0.012"
          contype="0" conaffinity="0" rgba="0.10 0.72 0.69 0.58"/>
    <geom name="lane_right" type="box" pos="0 -1.25 0.012" size="4.6 0.016 0.012"
          contype="0" conaffinity="0" rgba="0.10 0.72 0.69 0.58"/>
    <body name="dodge_ball" pos="-8 0 -1">
      <freejoint name="dodge_ball_free"/>
      <geom name="dodge_ball_geom" type="sphere" size="{BALL_RADIUS}" contype="0"
            conaffinity="0" rgba="0.97 0.18 0.29 1"/>
    </body>
    {ghosts}
"""
    source = source.replace("<worldbody>", "<worldbody>" + extras, 1)
    return source


def make_camera(azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.05, 0.0, 0.92]
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def set_free_joint(data: mujoco.MjData, qpos_address: int, position: np.ndarray) -> None:
    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]


def projectile_position(time_s: float, duration_s: float, scenario: Scenario) -> np.ndarray:
    """Return a deterministic straight throw with lead-in and recovery windows."""
    approach_start = 0.70
    approach_end = duration_s - 0.75
    if time_s < approach_start:
        return np.array(scenario.ball_start, dtype=np.float64)
    phase = np.clip((time_s - approach_start) / (approach_end - approach_start), 0.0, 1.0)
    start = np.asarray(scenario.ball_start, dtype=np.float64)
    end = np.asarray(scenario.ball_end, dtype=np.float64)
    return (1.0 - phase) * start + phase * end


def project_ball_only_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ball_position: np.ndarray,
    near: float,
    far: float,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Project the ball into the official head camera's 9x16 depth contract."""
    depth = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), far, dtype=np.float32)
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera_single"
    )
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    local = rotation.T @ (ball_position - data.cam_xpos[camera_id])
    forward = -float(local[2])
    if forward <= near:
        return depth, 0, {"forward_m": forward, "u": -1.0, "v": -1.0}

    tan_v = math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2.0)
    tan_h = tan_v * DEPTH_WIDTH / DEPTH_HEIGHT
    u = float(local[0]) / (forward * tan_h)
    v = float(local[1]) / (forward * tan_v)
    if abs(u) > 1.15 or abs(v) > 1.15:
        return depth, 0, {"forward_m": forward, "u": u, "v": v}

    center_x = (u + 1.0) * 0.5 * DEPTH_WIDTH
    center_y = (1.0 - v) * 0.5 * DEPTH_HEIGHT
    angular_radius = math.atan2(BALL_RADIUS, max(forward, near))
    radius_x = max(0.55, angular_radius / math.atan(tan_h) * DEPTH_WIDTH * 0.5)
    radius_y = max(0.55, angular_radius / math.atan(tan_v) * DEPTH_HEIGHT * 0.5)
    yy, xx = np.mgrid[0:DEPTH_HEIGHT, 0:DEPTH_WIDTH]
    mask = (
        ((xx + 0.5 - center_x) / radius_x) ** 2
        + ((yy + 0.5 - center_y) / radius_y) ** 2
        <= 1.0
    )
    if not bool(mask.any()):
        row = int(np.clip(round(center_y - 0.5), 0, DEPTH_HEIGHT - 1))
        col = int(np.clip(round(center_x - 0.5), 0, DEPTH_WIDTH - 1))
        mask[row, col] = True
    surface_depth = float(np.clip(np.linalg.norm(local) - BALL_RADIUS, near, far))
    depth[mask] = surface_depth
    return depth, int(mask.sum()), {"forward_m": forward, "u": u, "v": v}


def joint_addresses(
    model: mujoco.MjModel, names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
        dtype=np.int32,
    )
    if bool((joint_ids < 0).any()):
        missing = [name for name, joint_id in zip(names, joint_ids) if joint_id < 0]
        raise ValueError(f"Missing policy joints in G1 MJCF: {missing}")
    return (
        joint_ids,
        model.jnt_qposadr[joint_ids].astype(np.int32),
        model.jnt_dofadr[joint_ids].astype(np.int32),
    )


def collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    result = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.endswith("_collision"):
            result.append(geom_id)
    if not result:
        raise ValueError("No named G1 collision geometry found")
    return result


def geometry_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ball_geom_id: int,
    robot_geom_ids: list[int],
) -> float:
    distances = [
        mujoco.mj_geomDistance(model, data, ball_geom_id, geom_id, 10.0, None)
        for geom_id in robot_geom_ids
    ]
    return float(min(distances))


def run_dynamic_probe(contract: dict[str, Any], duration_s: float = 1.6) -> dict[str, Any]:
    """Test the unmodified generated free-base model under baked raw-motor PD."""
    model = mujoco.MjModel.from_xml_path(str(contract["generated_scene_xml"]))
    data = mujoco.MjData(model)
    _, qpos_addresses, dof_addresses = joint_addresses(model, contract["joint_names"])
    actuator_joint_names = [
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0])
        )
        for index in range(model.nu)
    ]
    order_matches = actuator_joint_names == contract["joint_names"]
    data.qpos[:] = model.qpos0
    data.qpos[:7] = ROOT_POSE
    data.qpos[qpos_addresses] = contract["default_pos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    height_trace = []
    first_below_half = None
    steps = round(duration_s / model.opt.timestep)
    for step in range(steps):
        q = data.qpos[qpos_addresses]
        dq = data.qvel[dof_addresses]
        torque = contract["kp"] * (contract["default_pos"] - q) - contract["kd"] * dq
        data.ctrl[:] = np.clip(
            torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)
        if step % max(1, round(0.02 / model.opt.timestep)) == 0:
            time_s = step * model.opt.timestep
            height = float(data.qpos[2])
            height_trace.append(height)
            if first_below_half is None and height < 0.5:
                first_below_half = time_s

    return {
        "attempted": True,
        "model": str(contract["generated_scene_xml"]),
        "duration_s": duration_s,
        "actuator_joint_order_matches_policy": order_matches,
        "initial_root_height_m": float(ROOT_POSE[2]),
        "minimum_root_height_m": round(min(height_trace), 6),
        "final_root_height_m": round(height_trace[-1], 6),
        "first_below_0_5m_s": None if first_below_half is None else round(first_below_half, 4),
        "stable": first_below_half is None,
        "blocker": (
            "The portable generated MJCF exposes raw torque motors but not the upstream "
            "mjlab actuator model and learned balance runtime; the default-pose hold collapses "
            "before a policy rollout can be interpreted."
        ),
    }


def set_replay_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    q: np.ndarray,
    root_address: int,
    ball_address: int,
    ball_position: np.ndarray,
    ghost_addresses: list[int],
    ghost_positions: list[np.ndarray],
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[root_address : root_address + 7] = ROOT_POSE
    data.qpos[qpos_addresses] = q
    set_free_joint(data, ball_address, ball_position)
    for address, position in zip(ghost_addresses, ghost_positions):
        set_free_joint(data, address, position)
    mujoco.mj_forward(model, data)


def run_scenario(
    model: mujoco.MjModel,
    contract: dict[str, Any],
    session: ort.InferenceSession,
    scenario: Scenario,
    duration_s: float,
    control_hz: int,
    max_joint_speed: float,
    target_alpha: float,
) -> dict[str, Any]:
    """Execute the complete observation, ONNX inference, and target path."""
    if control_hz != CONTROL_HZ:
        raise ValueError(f"The shipped checkpoint contract requires {CONTROL_HZ} Hz")
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    if list(input_info.shape) != [1, 960] or list(output_info.shape) != [1, 29]:
        raise ValueError(
            f"Unexpected ONNX contract: input={input_info.shape}, output={output_info.shape}"
        )

    _, qpos_addresses, _ = joint_addresses(model, contract["joint_names"])
    root_address = int(model.joint("floating_base_joint").qposadr[0])
    ball_address = int(model.joint("dodge_ball_free").qposadr[0])
    ghost_addresses = [
        int(model.joint(f"prediction_{index}_free").qposadr[0]) for index in range(10)
    ]
    ball_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "dodge_ball_geom"
    )
    robot_geom_ids = collision_geom_ids(model)
    joint_limits = model.jnt_range[
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in contract["joint_names"]]
    ]

    data = mujoco.MjData(model)
    default_data = mujoco.MjData(model)
    history = contract["ProprioHistory"](4)
    depth_ring = contract["DepthRing"](FRAME_OFFSETS)
    q = contract["default_pos"].copy()
    q_velocity = np.zeros(contract["num_joints"], dtype=np.float32)
    last_action = np.zeros(contract["num_joints"], dtype=np.float32)
    filtered_target = q.copy()
    far_depth_stacked = np.ones(DEPTH_HEIGHT * DEPTH_WIDTH * len(FRAME_OFFSETS), dtype=np.float32)
    steps = round(duration_s * control_hz)

    trace = {
        "qpos": [],
        "qvel": [],
        "actions": [],
        "targets": [],
        "observations": [],
        "depth_metres": [],
        "depth_stack": [],
        "ball_position": [],
        "ball_pixels": [],
        "policy_clearance": [],
        "default_clearance": [],
        "depth_action_delta_l2": [],
        "target_clip_count": [],
    }

    for step in range(steps):
        time_s = step / control_hz
        ball_position = projectile_position(time_s, duration_s, scenario)
        ghost_positions = [
            projectile_position(
                min(time_s + 0.07 * (index + 1), duration_s), duration_s, scenario
            )
            for index in range(10)
        ]
        set_replay_state(
            model,
            data,
            qpos_addresses,
            q,
            root_address,
            ball_address,
            ball_position,
            ghost_addresses,
            ghost_positions,
        )
        depth_metres, visible_pixels, _projection = project_ball_only_depth(
            model,
            data,
            ball_position,
            contract["depth_near"],
            contract["depth_far"],
        )
        normalized_depth = contract["depth_metres_to_obs"](
            depth_metres.reshape(-1), contract["depth_near"], contract["depth_far"]
        )
        depth_stacked = depth_ring.push(normalized_depth)
        history.append(
            base_ang_vel=np.zeros(3, dtype=np.float32),
            proj_grav=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
            joint_pos_rel=q - contract["default_pos"],
            joint_vel_rel=q_velocity,
            last_action=last_action,
        )
        proprio = history.vector()
        observation = contract["assemble_obs"](proprio, depth_stacked)
        action = session.run(None, {input_info.name: observation})[0][0].astype(np.float32)
        far_observation = contract["assemble_obs"](proprio, far_depth_stacked)
        far_action = session.run(None, {input_info.name: far_observation})[0][0].astype(np.float32)
        if action.shape != (29,) or not bool(np.isfinite(action).all()):
            raise RuntimeError(f"Invalid ONNX action at step {step}: shape={action.shape}")

        raw_target = contract["default_pos"] + action * contract["action_scale"]
        target = np.clip(raw_target, joint_limits[:, 0], joint_limits[:, 1]).astype(np.float32)
        clip_count = int(np.count_nonzero(np.abs(target - raw_target) > 1e-6))
        filtered_target = (
            (1.0 - target_alpha) * filtered_target + target_alpha * target
        ).astype(np.float32)
        delta = np.clip(
            filtered_target - q,
            -max_joint_speed / control_hz,
            max_joint_speed / control_hz,
        )
        next_q = (q + delta).astype(np.float32)
        q_velocity = ((next_q - q) * control_hz).astype(np.float32)
        q = next_q
        last_action = action.copy()

        set_replay_state(
            model,
            data,
            qpos_addresses,
            q,
            root_address,
            ball_address,
            ball_position,
            ghost_addresses,
            ghost_positions,
        )
        policy_clearance = geometry_clearance(
            model, data, ball_geom_id, robot_geom_ids
        )
        set_replay_state(
            model,
            default_data,
            qpos_addresses,
            contract["default_pos"],
            root_address,
            ball_address,
            ball_position,
            ghost_addresses,
            ghost_positions,
        )
        default_clearance = geometry_clearance(
            model, default_data, ball_geom_id, robot_geom_ids
        )

        trace["qpos"].append(q.copy())
        trace["qvel"].append(q_velocity.copy())
        trace["actions"].append(action.copy())
        trace["targets"].append(target.copy())
        trace["observations"].append(observation[0].copy())
        trace["depth_metres"].append(depth_metres.reshape(-1).copy())
        trace["depth_stack"].append(depth_stacked.copy())
        trace["ball_position"].append(ball_position.copy())
        trace["ball_pixels"].append(visible_pixels)
        trace["policy_clearance"].append(policy_clearance)
        trace["default_clearance"].append(default_clearance)
        trace["depth_action_delta_l2"].append(float(np.linalg.norm(action - far_action)))
        trace["target_clip_count"].append(clip_count)

    arrays = {key: np.asarray(value) for key, value in trace.items()}
    joint_ranges = np.ptp(arrays["qpos"], axis=0)
    active_joint_count = int(np.count_nonzero(joint_ranges >= MOVEMENT_THRESHOLD_RAD))
    return {
        "scenario": scenario,
        "trace": arrays,
        "summary": {
            "name": scenario.name,
            "label": scenario.label,
            "inference_ticks": steps,
            "finite_actions": bool(np.isfinite(arrays["actions"]).all()),
            "active_joint_count_ge_0_05rad": active_joint_count,
            "minimum_policy_clearance_m": round(float(arrays["policy_clearance"].min()), 6),
            "minimum_default_pose_clearance_m": round(
                float(arrays["default_clearance"].min()), 6
            ),
            "maximum_depth_action_delta_l2": round(
                float(arrays["depth_action_delta_l2"].max()), 6
            ),
            "mean_visible_ball_pixels": round(float(arrays["ball_pixels"].mean()), 4),
            "maximum_clipped_targets_per_tick": int(arrays["target_clip_count"].max()),
            "joint_range_rad": {
                name: round(float(value), 6)
                for name, value in zip(contract["joint_names"], joint_ranges)
            },
        },
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def label_view(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((18, 18, 450, 82), radius=10, fill=(7, 11, 17, 220))
    draw.text((34, 29), title, font=font(23, True), fill=(242, 247, 250, 255))
    draw.text((34, 56), subtitle, font=font(15), fill=(159, 214, 205, 255))
    return np.asarray(image)


def telemetry_panel(
    depth_stack: np.ndarray,
    action: np.ndarray,
    joint_names: list[str],
    scenario_label: str,
    time_s: float,
    clearance: float,
    active_joint_count: int,
) -> np.ndarray:
    image = Image.new("RGB", (960, 540), (12, 17, 24))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((32, 24), "POLICY I/O", font=font(28, True), fill=(242, 247, 250))
    draw.text(
        (32, 60),
        "960-D OBSERVATION  ->  ONNX  ->  29 JOINT TARGETS",
        font=font(17, True),
        fill=(92, 211, 191),
    )
    draw.text((32, 92), scenario_label, font=font(19, True), fill=(247, 189, 82))
    draw.text(
        (32, 122),
        f"t={time_s:4.2f}s   geometry clearance={clearance:+.3f} m   moving joints={active_joint_count}/29",
        font=font(16),
        fill=(200, 210, 219),
    )

    depth_frames = depth_stack.reshape(len(FRAME_OFFSETS), DEPTH_HEIGHT, DEPTH_WIDTH)
    tile_w, tile_h = 208, 117
    for index, (offset, depth_frame) in enumerate(zip(FRAME_OFFSETS, depth_frames)):
        normalized = np.clip((1.0 - depth_frame) * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack(
            [normalized, (normalized * 0.78).astype(np.uint8), (255 - normalized // 3)],
            axis=-1,
        )
        tile = Image.fromarray(rgb).resize((tile_w, tile_h), Image.Resampling.NEAREST)
        x = 32 + (index % 2) * (tile_w + 18)
        y = 168 + (index // 2) * (tile_h + 36)
        image.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(75, 96, 111, 255), width=2)
        draw.text(
            (x, y + tile_h + 7),
            f"depth t-{offset:02d}  (9x16)",
            font=font(14),
            fill=(165, 180, 191),
        )

    top_indices = np.argsort(np.abs(action))[::-1][:9]
    chart_x0, chart_x1 = 510, 918
    chart_y0 = 173
    max_abs = max(float(np.max(np.abs(action[top_indices]))), 1e-6)
    draw.text((510, 145), "LARGEST POLICY OUTPUTS", font=font(16, True), fill=(240, 244, 247))
    for rank, joint_index in enumerate(top_indices):
        y = chart_y0 + rank * 37
        value = float(action[joint_index])
        short_name = joint_names[joint_index].replace("_joint", "")
        draw.text((chart_x0, y), short_name[:25], font=font(13), fill=(187, 199, 208))
        center = 792
        draw.line((center, y + 20, center, y + 31), fill=(86, 100, 112, 255), width=1)
        extent = int(abs(value) / max_abs * 116)
        x0, x1 = (center - extent, center) if value < 0 else (center, center + extent)
        draw.rounded_rectangle((x0, y + 21, x1, y + 30), radius=4, fill=(92, 211, 191, 255))
        draw.text((922, y + 16), f"{value:+.2f}", anchor="ra", font=font(13), fill=(237, 191, 94))
    return np.asarray(image)


def render_video(
    model: mujoco.MjModel,
    contract: dict[str, Any],
    runs: list[dict[str, Any]],
    output_path: Path,
    duration_s: float,
    control_hz: int,
    render_fps: int,
) -> None:
    renderer = mujoco.Renderer(model, height=540, width=960)
    cameras = (
        ("THREE-QUARTER", make_camera(132.0, -14.0, 3.5)),
        ("PROFILE", make_camera(90.0, -7.0, 3.6)),
        ("FRONT", make_camera(205.0, -9.0, 3.35)),
    )
    _, qpos_addresses, _ = joint_addresses(model, contract["joint_names"])
    root_address = int(model.joint("floating_base_joint").qposadr[0])
    ball_address = int(model.joint("dodge_ball_free").qposadr[0])
    ghost_addresses = [
        int(model.joint(f"prediction_{index}_free").qposadr[0]) for index in range(10)
    ]
    data = mujoco.MjData(model)
    writer = imageio.get_writer(
        output_path,
        fps=render_fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-map_metadata",
            "-1",
        ],
    )
    try:
        for run_index, run in enumerate(runs):
            trace = run["trace"]
            summary = run["summary"]
            scenario = run["scenario"]
            total_frames = round(duration_s * render_fps)
            active_joint_count = summary["active_joint_count_ge_0_05rad"]
            for frame_index in range(total_frames):
                time_s = frame_index / render_fps
                control_index = min(round(time_s * control_hz), len(trace["qpos"]) - 1)
                ball_position = trace["ball_position"][control_index]
                ghost_positions = [
                    projectile_position(
                        min(time_s + 0.07 * (index + 1), duration_s), duration_s, scenario
                    )
                    for index in range(10)
                ]
                set_replay_state(
                    model,
                    data,
                    qpos_addresses,
                    trace["qpos"][control_index],
                    root_address,
                    ball_address,
                    ball_position,
                    ghost_addresses,
                    ghost_positions,
                )
                views = []
                subtitle = f"scenario {run_index + 1}/{len(runs)}  |  upstream G1 MJCF"
                for camera_name, camera in cameras:
                    renderer.update_scene(data, camera=camera)
                    views.append(label_view(renderer.render().copy(), camera_name, subtitle))
                panel = telemetry_panel(
                    trace["depth_stack"][control_index],
                    trace["actions"][control_index],
                    contract["joint_names"],
                    scenario.label,
                    time_s,
                    float(trace["policy_clearance"][control_index]),
                    active_joint_count,
                )
                frame = np.vstack((np.hstack((views[0], views[1])), np.hstack((views[2], panel))))
                writer.append_data(frame)
    finally:
        writer.close()
        renderer.close()


def save_trace(path: Path, runs: list[dict[str, Any]]) -> None:
    payload: dict[str, np.ndarray] = {}
    for run in runs:
        prefix = run["scenario"].name
        for key, value in run["trace"].items():
            payload[f"{prefix}__{key}"] = value
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(
            "/data/Data14TB/03competition/amd-physical-ai-showcase/.vendor/perceptive_cbf_rl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/Data14TB/03competition/amd-physical-ai-showcase/results/perceptive_cbf_rl_amd"
        ),
    )
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per scenario")
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--control-hz", type=int, default=CONTROL_HZ)
    parser.add_argument("--max-joint-speed", type=float, default=3.5)
    parser.add_argument("--target-alpha", type=float, default=0.35)
    parser.add_argument(
        "--scenarios", nargs="+", default=[scenario.name for scenario in SCENARIOS]
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--skip-dynamic-probe", action="store_true")
    args = parser.parse_args()
    if args.duration < 3.0:
        raise SystemExit("--duration must be at least 3 seconds for temporal depth history")
    if not 0.0 < args.target_alpha <= 1.0:
        raise SystemExit("--target-alpha must be in (0, 1]")

    selected = []
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    for name in args.scenarios:
        if name not in by_name:
            raise SystemExit(f"Unknown scenario {name}; choose from {sorted(by_name)}")
        selected.append(by_name[name])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_upstream(args.upstream_root)
    if contract["commit"] != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Vendor commit {contract['commit']} != pinned {PINNED_UPSTREAM_COMMIT}"
        )
    model = mujoco.MjModel.from_xml_string(build_replay_scene(contract["robot_xml"]))
    session = ort.InferenceSession(
        str(contract["checkpoint"]), providers=["CPUExecutionProvider"]
    )
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    if int(input_info.shape[-1]) != 960 or int(output_info.shape[-1]) != 29:
        raise RuntimeError(
            f"Checkpoint contract mismatch: {input_info.shape} -> {output_info.shape}"
        )

    dynamic_probe = (
        {"attempted": False, "reason": "disabled by --skip-dynamic-probe"}
        if args.skip_dynamic_probe
        else run_dynamic_probe(contract)
    )
    runs = [
        run_scenario(
            model,
            contract,
            session,
            scenario,
            args.duration,
            args.control_hz,
            args.max_joint_speed,
            args.target_alpha,
        )
        for scenario in selected
    ]
    all_ranges = np.max(
        np.stack(
            [
                np.array(
                    [run["summary"]["joint_range_rad"][name] for name in contract["joint_names"]]
                )
                for run in runs
            ]
        ),
        axis=0,
    )
    active_joints = int(np.count_nonzero(all_ranges >= MOVEMENT_THRESHOLD_RAD))
    if active_joints < 20:
        raise RuntimeError(f"Whole-body motion gate failed: only {active_joints}/29 joints moved")

    trace_path = args.output_dir / "g1-onnx-whole-body-trace.npz"
    save_trace(trace_path, runs)
    video_path = args.output_dir / "unitree-g1-onnx-whole-body-dodge-1080p.mp4"
    if not args.no_video:
        render_video(
            model,
            contract,
            runs,
            video_path,
            args.duration,
            args.control_hz,
            args.render_fps,
        )

    summaries = [run["summary"] for run in runs]
    evaluation = {
        "schema": "perceptive-cbf-rl/portable-g1-onnx-whole-body/v2",
        "status": "complete_onnx_path_fixed_base_replay",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": contract["commit"],
            "checkpoint": str(contract["checkpoint"]),
            "checkpoint_sha256": sha256_file(contract["checkpoint"]),
            "robot_mjcf": str(contract["robot_xml"]),
            "robot_mjcf_sha256": sha256_file(contract["robot_xml"]),
        },
        "policy_contract": {
            "input_name": input_info.name,
            "input_shape": list(input_info.shape),
            "input_dtype": input_info.type,
            "output_name": output_info.name,
            "output_shape": list(output_info.shape),
            "output_dtype": output_info.type,
            "observation": {
                "proprio_dim": 384,
                "proprio_history_frames": 4,
                "proprio_term_order": [
                    "base_ang_vel",
                    "projected_gravity",
                    "command",
                    "joint_pos_rel",
                    "joint_vel_rel",
                    "last_action",
                ],
                "depth_frame_shape": [DEPTH_HEIGHT, DEPTH_WIDTH],
                "depth_frame_offsets_newest_to_oldest": list(FRAME_OFFSETS),
                "depth_dim": DEPTH_HEIGHT * DEPTH_WIDTH * len(FRAME_OFFSETS),
                "depth_near_far_m": [contract["depth_near"], contract["depth_far"]],
                "assembly": "proprio[384] ++ depth[576]",
            },
            "action": {
                "joint_count": 29,
                "joint_names": contract["joint_names"],
                "mapping": "target = DEFAULT_POS + action * ACTION_SCALE",
                "joint_limit_clipping": True,
                "portable_target_filter_alpha": args.target_alpha,
                "portable_max_joint_speed_rad_s": args.max_joint_speed,
            },
        },
        "protocol": {
            "control_hz": args.control_hz,
            "render_fps": args.render_fps,
            "scenario_duration_s": args.duration,
            "scenarios": [scenario.name for scenario in selected],
            "synthetic_input": "projected ball-only 9x16 metric depth history",
            "root_control": "fixed at official default pose; no root translation or sidestep proxy",
            "video_resolution": [1920, 1080],
            "video_views": ["three-quarter", "profile", "front", "policy I/O"],
        },
        "summary": {
            "scenarios": len(runs),
            "inference_ticks": sum(item["inference_ticks"] for item in summaries),
            "finite_action_ticks": sum(item["inference_ticks"] for item in summaries),
            "whole_body_joint_count_ge_0_05rad": active_joints,
            "whole_body_joint_count": 29,
            "minimum_policy_geometry_clearance_m": min(
                item["minimum_policy_clearance_m"] for item in summaries
            ),
            "minimum_default_pose_geometry_clearance_m": min(
                item["minimum_default_pose_clearance_m"] for item in summaries
            ),
            "maximum_depth_conditioned_action_delta_l2": max(
                item["maximum_depth_action_delta_l2"] for item in summaries
            ),
            "joint_range_rad": {
                name: round(float(value), 6)
                for name, value in zip(contract["joint_names"], all_ranges)
            },
        },
        "scenarios": summaries,
        "free_base_dynamic_probe": dynamic_probe,
        "artifacts": {
            "trace": trace_path.name,
            "video": None if args.no_video else video_path.name,
        },
        "limitations": [
            "The fixed-base replay validates the complete upstream ONNX input, inference, and 29-joint target path.",
            "It is not the upstream free-base mjlab/MuJoCo-Warp closed-loop benchmark because the portable MJCF raw-motor hold collapses without that actuator and balance stack.",
            "The depth source is deterministic synthetic ball-only depth rather than live ZED and EfficientTAM output.",
        ],
    }
    eval_path = args.output_dir / "eval_info.json"
    eval_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    artifacts = {"eval_info.json": sha256_file(eval_path), trace_path.name: sha256_file(trace_path)}
    if not args.no_video:
        artifacts[video_path.name] = sha256_file(video_path)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "working_directory": str(Path.cwd()),
        "upstream_commit": contract["commit"],
        "checkpoint_sha256": sha256_file(contract["checkpoint"]),
        "robot_mjcf_sha256": sha256_file(contract["robot_xml"]),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": package_version("mujoco"),
            "onnxruntime": package_version("onnxruntime"),
            "onnx_providers": session.get_providers(),
            "numpy": package_version("numpy"),
            "imageio": package_version("imageio"),
            "pillow": package_version("pillow"),
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        },
        "outputs_sha256": artifacts,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evaluation["status"],
                "output_dir": str(args.output_dir),
                "summary": evaluation["summary"],
                "dynamic_probe": dynamic_probe,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
