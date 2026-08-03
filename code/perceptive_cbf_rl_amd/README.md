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
