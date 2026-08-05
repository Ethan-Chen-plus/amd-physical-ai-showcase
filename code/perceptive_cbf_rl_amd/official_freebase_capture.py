#!/usr/bin/env python3
"""Capture PAC-MAN's official free-base dodge loop as a reproducible MP4.

The script runs the shipped ``dodge_link_cbf.onnx`` policy through the
upstream UDP deployment contract while mjlab supplies Unitree G1 dynamics,
head-camera depth, ball throws, contacts, and balance control. It records three
deterministic threat layouts: a central rising ball, a left-side descending
ball, and a right-side descending ball.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FRAME_OFFSETS = (0, 3, 8, 18)
CONTROL_HZ = 50
VIDEO_FPS = 25
PINNED_UPSTREAM_COMMIT = "2d4266978805e8272daa7f029a8bca91cf45e1ba"
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True)
class DodgeMode:
    name: str
    title: str
    caption: str
    high_throw: bool
    aim_offset_y: float


MODES = (
    DodgeMode(
        name="duck",
        title="DUCK",
        caption="A rising center-line threat triggers a compact whole-body crouch.",
        high_throw=True,
        aim_offset_y=0.0,
    ),
    DodgeMode(
        name="sidestep_right",
        title="SIDESTEP RIGHT",
        caption="A left-side threat is cleared with a lateral whole-body response.",
        high_throw=False,
        aim_offset_y=0.22,
    ),
    DodgeMode(
        name="sidestep_left",
        title="SIDESTEP LEFT",
        caption="A right-side threat produces the mirrored evasive response.",
        high_throw=False,
        aim_offset_y=-0.22,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def depth_panel(depth_m: np.ndarray) -> Image.Image:
    """Render the 9x16 ball-only depth sent to the policy."""
    depth = np.asarray(depth_m, dtype=np.float32).reshape(9, 16)
    panel = np.zeros((9, 16, 3), dtype=np.uint8)
    threat = depth < 4.95
    intensity = np.clip((5.0 - depth) / 4.9, 0.0, 1.0)
    panel[..., 0] = np.where(threat, 240, 16)
    panel[..., 1] = np.where(threat, 72 + 120 * intensity, 25)
    panel[..., 2] = np.where(threat, 46, 34)
    return Image.fromarray(panel, mode="RGB").resize((352, 198), Image.Resampling.NEAREST)


def compose_frame(
    frame: np.ndarray,
    depth_m: np.ndarray,
    mode: DodgeMode,
    elapsed_s: float,
    root_y: float,
    head_z: float,
    phase: str,
) -> np.ndarray:
    """Build a 1920x1080 evidence frame around the native simulation view."""
    canvas = Image.new("RGB", (1920, 1080), (11, 15, 20))
    scene = Image.fromarray(frame).convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
    canvas.paste(scene, (0, 0))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, 1920, 172), fill=(9, 14, 20, 218))
    draw.rectangle((0, 888, 1920, 1080), fill=(9, 14, 20, 226))
    draw.rounded_rectangle((1516, 196, 1892, 438), radius=8, fill=(8, 13, 18, 222), outline=(93, 207, 190, 180), width=2)
    draw.text((64, 34), "PAC-MAN · PERCEPTIVE CBF-RL", font=font(FONT_BOLD, 24), fill=(111, 224, 207, 255))
    draw.text((64, 70), mode.title, font=font(FONT_BOLD, 56), fill=(255, 255, 255, 255))
    draw.text((64, 136), "Unitree G1 · free-base dynamics · zero velocity command", font=font(FONT_REGULAR, 22), fill=(202, 212, 223, 255))
    draw.text((1540, 212), "BALL-ONLY DEPTH · 9×16", font=font(FONT_BOLD, 17), fill=(226, 232, 239, 255))
    draw.text((64, 922), mode.caption, font=font(FONT_BOLD, 30), fill=(255, 255, 255, 255))
    draw.text((64, 975), f"PHASE  {phase.upper()}   ·   TIME  {elapsed_s:04.1f}s", font=font(FONT_REGULAR, 22), fill=(184, 196, 207, 255))
    draw.text((1320, 924), f"ROOT Y  {root_y:+.3f} m", font=font(FONT_BOLD, 23), fill=(111, 224, 207, 255))
    draw.text((1320, 968), f"HEAD Z  {head_z:.3f} m", font=font(FONT_BOLD, 23), fill=(111, 224, 207, 255))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    canvas.paste(depth_panel(depth_m), (1528, 236))
    return np.asarray(canvas)


def load_environment(task_id: str, seed: int, width: int, height: int):
    """Build the upstream play environment with deterministic throw geometry."""
    import torch

    import src.tasks.amp_loco.config.g1 as _g1  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import enable_depth_aug_preview
    from src.tasks.amp_loco.mdp.events import reset_to_default_stand

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = seed
    env_cfg.depth_frame_offsets = FRAME_OFFSETS
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    env_cfg.viewer.distance = 3.4
    env_cfg.viewer.elevation = -6.0
    env_cfg.viewer.azimuth = 90.0
    env_cfg.episode_length_s = 8.0

    stand_event = env_cfg.events["reset_from_motion"]
    stand_event.func = reset_to_default_stand
    stand_event.params = {}

    throw_event = env_cfg.events["throw_ball_on_dwell"]
    throw_event.params["aim_noise_scale"] = 0.0
    throw_event.params["lead_target"] = False
    throw_event.params["throw_interval_range"] = (30.0, 31.0)

    enable_depth_aug_preview(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
    env._depth_aug_display_enabled = True
    env._dodge_throw_paused = True
    env.reset()
    return env, torch


def body_index(robot: Any, body_name: str) -> int:
    ids, _ = robot.find_bodies(body_name)
    if not ids:
        raise ValueError(f"Body not found: {body_name}")
    return int(ids[0])


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    upstream_root = args.upstream_root.resolve()
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    commit = subprocess.check_output(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream {PINNED_UPSTREAM_COMMIT}, found {commit}")

    from deploy.common.g1_deploy_constants import NUM_JOINTS, POLICY_JOINT_NAMES
    from deploy.common.udp_sync import UDP_HW_PORT, UDP_HOST, create_udp_socket
    from deploy.sim.sim_node import UdpDodgeBridge

    checkpoint = upstream_root / "deploy/ckpts/dodge_link_cbf.onnx"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "pacman-g1-freebase-dodge-modes-1080p.mp4"

    policy_log = output_dir / "policy.log"
    policy_cmd = [
        str(args.policy_python),
        str(upstream_root / "deploy/policy/dodge_policy.py"),
        str(checkpoint),
        "--frame-offsets",
        str(FRAME_OFFSETS).replace(" ", ""),
    ]
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = str(upstream_root)
    with policy_log.open("w", encoding="utf-8") as log_handle:
        policy = subprocess.Popen(
            policy_cmd,
            cwd=upstream_root,
            env=env_vars,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    env = None
    writer = None
    state_sock = depth_sock = action_sock = None
    try:
        time.sleep(2.0)
        if policy.poll() is not None:
            raise RuntimeError(
                f"Policy process exited with code {policy.returncode}; see {policy_log}"
            )
        env, torch = load_environment(args.task_id, args.seed, 960, 540)
        robot = env.scene["robot"]
        if list(robot.joint_names) != list(POLICY_JOINT_NAMES):
            raise RuntimeError("Policy and environment joint orders differ")

        depth_key = "head_depth_single"
        state_sock = create_udp_socket(UDP_HOST, 0)
        depth_sock = create_udp_socket(UDP_HOST, 0)
        action_sock = create_udp_socket(UDP_HOST, UDP_HW_PORT)
        bridge = UdpDodgeBridge(
            env,
            torch,
            "cuda:0",
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
            depth_key,
            state_sock,
            depth_sock,
            action_sock,
            9 * 16,
            True,
        )
        bridge.zero_command = True
        bridge.depth_decim = 1

        torso_idx = body_index(robot, "torso_link")
        writer = imageio.get_writer(
            str(video_path),
            fps=VIDEO_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

        mode_results: list[dict[str, Any]] = []
        for mode_index, mode in enumerate(MODES):
            env.reset(seed=args.seed + mode_index)
            env._dodge_throw_paused = True
            env._dodge_throw_aim_offset_b = (0.0, mode.aim_offset_y)
            bridge.reset()
            bridge._prev_len = -1

            root_y_trace: list[float] = []
            head_z_trace: list[float] = []
            terminated_count = 0
            throw_step = args.warmup_steps
            for step in range(args.steps_per_mode):
                if step == throw_step:
                    env._dodge_throw_force_high = mode.high_throw
                    env._dodge_throw_once = True

                action = bridge(None)
                _, _, terminated, truncated, _ = env.step(action)
                terminated_count += int(bool(terminated[0] or truncated[0]))

                root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
                torso_pos = robot.data.body_link_pos_w[0, torso_idx].detach().cpu().numpy()
                root_y_trace.append(float(root_pos[1]))
                head_z_trace.append(float(torso_pos[2] + 0.45))

                if step % (CONTROL_HZ // VIDEO_FPS) == 0:
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("Offscreen renderer returned no frame")
                    depth_m = bridge._depth_metres()
                    if step < throw_step:
                        phase = "stand"
                    elif step < throw_step + 45:
                        phase = "react"
                    else:
                        phase = "recover"
                    writer.append_data(
                        compose_frame(
                            frame,
                            depth_m,
                            mode,
                            step / CONTROL_HZ,
                            root_y_trace[-1],
                            head_z_trace[-1],
                            phase,
                        )
                    )

            baseline_head = float(np.median(head_z_trace[: max(10, throw_step // 2)]))
            mode_results.append(
                {
                    "mode": mode.name,
                    "high_throw": mode.high_throw,
                    "aim_offset_y_m": mode.aim_offset_y,
                    "steps": args.steps_per_mode,
                    "resets": terminated_count,
                    "root_y_min_m": min(root_y_trace),
                    "root_y_max_m": max(root_y_trace),
                    "root_y_range_m": max(root_y_trace) - min(root_y_trace),
                    "head_z_baseline_m": baseline_head,
                    "head_z_min_m": min(head_z_trace),
                    "head_drop_m": baseline_head - min(head_z_trace),
                }
            )

        writer.close()
        writer = None
        result = {
            "schema": "perceptive-cbf-rl/upstream-freebase-sim2sim/v1",
            "status": "complete",
            "task": args.task_id,
            "upstream_repository": "https://github.com/lzyang2000/perceptive_cbf_rl",
            "upstream_commit": commit,
            "checkpoint": "deploy/ckpts/dodge_link_cbf.onnx",
            "checkpoint_sha256": sha256_file(checkpoint),
            "observation_dim": 960,
            "action_dim": int(NUM_JOINTS),
            "frame_offsets": list(FRAME_OFFSETS),
            "control_hz": CONTROL_HZ,
            "video_fps": VIDEO_FPS,
            "zero_velocity_command": True,
            "free_base": True,
            "modes": mode_results,
            "video": video_path.name,
        }
        (output_dir / "eval_info.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest = {
            video_path.name: sha256_file(video_path),
            "eval_info.json": sha256_file(output_dir / "eval_info.json"),
            "dodge_link_cbf.onnx": sha256_file(checkpoint),
        }
        (output_dir / "sha256sum.txt").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in manifest.items()),
            encoding="utf-8",
        )
        run_manifest = {
            "schema": "perceptive-cbf-rl/upstream-freebase-run/v1",
            "command": (
                f"MUJOCO_GL=egl {args.policy_python} "
                "code/perceptive_cbf_rl_amd/official_freebase_capture.py"
            ),
            "upstream_commit": commit,
            "task": args.task_id,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mjlab": "1.5.3",
                "mujoco": "3.10.0",
                "mujoco_warp": "3.10.0.3",
                "warp": "1.15.0",
                "onnxruntime": "1.23.2",
                "renderer": "EGL",
                "simulation_device": "cuda:0 (upstream reference runtime)",
            },
            "policy_contract": {
                "observation_dim": 960,
                "action_dim": int(NUM_JOINTS),
                "frame_offsets": list(FRAME_OFFSETS),
                "control_hz": CONTROL_HZ,
            },
            "outputs_sha256": manifest,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        if writer is not None:
            writer.close()
        for sock in (state_sock, depth_sock, action_sock):
            if sock is not None:
                sock.close()
        if env is not None:
            env.close()
        policy.terminate()
        try:
            policy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            policy.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(".vendor/perceptive_cbf_rl"),
    )
    parser.add_argument(
        "--policy-python",
        type=Path,
        default=Path("/data/Data14TB/envs/perceptive-cbf-rl-official/bin/python"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/perceptive-cbf-rl-freebase"),
    )
    parser.add_argument(
        "--task-id",
        default="Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--steps-per-mode", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_capture(args), indent=2, ensure_ascii=False))
