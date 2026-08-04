# PAC-MAN on AMD: predictive CBF control-path port

This folder records the AMD-side port boundary for
[lzyang2000/perceptive_cbf_rl](https://github.com/lzyang2000/perceptive_cbf_rl).
The upstream project is a Unitree G1 humanoid dodgeball system built around
`mjlab`, MuJoCo Warp, AMP and `rsl_rl`; its published hardware result belongs
to the upstream project and is not reused as an AMD result.

## What is reproduced here

`amd_pacman_cbf_smoke.py` preserves the upstream predictive perpendicular CBF
calculation:

1. estimate the horizontal ball trajectory from position and velocity;
2. gate threats by airborne state, approach direction and sensing radius;
3. select a latched perpendicular escape direction;
4. project the nominal velocity onto the time-aware CBF half-space.

MuJoCo CPU provides a small portable projectile scene and two camera renders.
PyTorch performs the batched CBF tensor calculation. On the AMD cloud runtime,
`torch.cuda` is the ROCm device path even though PyTorch keeps the historical
CUDA API name.

The output is a **control-path validation**, not a full PAC-MAN policy
reproduction. It does not claim the upstream G1 AMP training result, the
`mjlab`/MuJoCo-Warp CUDA simulator, ZED/EfficientTAM deployment, or the
upstream 19/20 hardware benchmark.

## Unitree G1 asset replay

The showcase page includes a portable replay using the pinned Unitree G1 MJCF
and mesh assets. It preserves the predictive perpendicular CBF geometry and
combines three-quarter, projectile-profile, and low-front shots with a
synchronized top-view inset. The renderer overlays the predicted projectile
path, safety envelope, filtered velocity, and live clearance.

Fetch the pinned upstream assets and run the replay with the dedicated AMD
environment:

```bash
git clone --depth 1 https://github.com/lzyang2000/perceptive_cbf_rl.git \
  .vendor/perceptive_cbf_rl
git -C .vendor/perceptive_cbf_rl checkout 2d4266978805e8272daa7f029a8bca91cf45e1ba

MUJOCO_GL=egl \
  /data/Data14TB/envs/pacman-g1-replay/bin/python \
  code/perceptive_cbf_rl_amd/g1_amd_dodge_replay.py \
  --upstream-xml .vendor/perceptive_cbf_rl/src/assets/robots/unitree_g1/xmls/scene_g1.xml \
  --output-dir results/pacman_g1_amd_replay \
  --episodes 8
```

The run produces `eval_info.json`, `run_manifest.json`, and
`unitree-g1-predictive-cbf-amd-replay.mp4`. The public evidence uses eight
fixed seeds, preserves clearance in 8/8 replays, and records a 0.42 m minimum
clearance. The 15-second showcase video presents the same run through three
cinematic shots and a synchronized top-view trajectory.

## AMD run

The reproducible AMD runtime used for the evidence run was:

```bash
/workspace/envs/robowits/bin/python \
  amd_pacman_cbf_smoke.py \
  --output-dir /tmp/perceptive_cbf_rl_amd \
  --episodes 12 \
  --device rocm
```

The command writes:

- `eval_info.json`: runtime, protocol, per-seed outcomes and boundary labels;
- `run_manifest.json`: exact command and upstream commit;
- `pacman-cbf-amd-proxy.mp4`: overview/top-view control-path replay.

For a CPU-only development check:

```bash
python amd_pacman_cbf_smoke.py --device cpu --episodes 2 --no-video
```

## Full migration boundary

The next engineering step for a full AMD reproduction is to replace the
upstream CUDA-only `mjlab`/MuJoCo-Warp vector simulator with an AMD-compatible
batched simulator, then port the G1 AMP training and depth observation path.
The CBF control term is already isolated and validated independently so that
simulator work does not obscure safety-logic regressions.
