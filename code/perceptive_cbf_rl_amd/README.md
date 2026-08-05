# Unitree G1 free-base whole-body dodging

This directory ports the deployment contract from
[`lzyang2000/perceptive_cbf_rl`](https://github.com/lzyang2000/perceptive_cbf_rl)
into a deterministic free-base simulation and an AMD-portable policy probe.

## Official free-base path

`official_freebase_capture.py` runs the upstream task, robot, controller, depth
pipeline, UDP deployment contract, and shipped ONNX actor together. With a zero
velocity command, the same policy produces three responses to deterministic
ball trajectories:

- a compact whole-body duck;
- a right sidestep;
- a left sidestep.

The capture uses the official `mjlab` and `mujoco-warp` stack because the
balance controller and free-base actuators are part of that runtime. It is kept
in an isolated environment so the existing ROCm inference environment remains
unchanged.

```bash
MUJOCO_GL=egl \
  PYTHONPATH=.vendor/perceptive_cbf_rl \
  /data/Data14TB/envs/perceptive-cbf-rl-official/bin/python \
  code/perceptive_cbf_rl_amd/official_freebase_capture.py
```

Outputs:

- `pacman-g1-freebase-dodge-modes-1080p.mp4`;
- `eval_info.json` with duck and lateral-displacement measurements;
- `run_manifest.json` with pinned versions and the policy contract;
- `sha256sum.txt` for the video, evaluation, and ONNX checkpoint.

## AMD-portable policy path

`g1_amd_dodge_replay.py` runs the pinned upstream artifacts directly:

- `deploy/ckpts/dodge_link_cbf.onnx`;
- `src/assets/robots/unitree_g1/xmls/g1.xml` and its mesh assets;
- the upstream 4-frame, term-major proprio history;
- synthetic ball-only 9x16 metric depth at offsets `(0, 3, 8, 18)`;
- the exact `384 proprio + 576 depth = 960` observation;
- ONNX inference with a 29-D output;
- `target = DEFAULT_POS + action * ACTION_SCALE` in official joint order.

This lightweight path is useful for validating the observation layout, ONNX
inference, joint order, and action scaling in the existing AMD environment.

## Runtime

The existing environment is located on the data disk:

```bash
uv pip install \
  --python /data/Data14TB/envs/pacman-g1-replay/bin/python \
  -r /data/Data14TB/03competition/amd-physical-ai-showcase/code/perceptive_cbf_rl_amd/requirements.txt
```

Run tests:

```bash
MUJOCO_GL=egl \
  /data/Data14TB/envs/pacman-g1-replay/bin/python \
  /data/Data14TB/03competition/amd-physical-ai-showcase/code/perceptive_cbf_rl_amd/test_g1_onnx_replay.py
```

Run the portable policy probe:

```bash
MUJOCO_GL=egl \
  /data/Data14TB/envs/pacman-g1-replay/bin/python \
  /data/Data14TB/03competition/amd-physical-ai-showcase/code/perceptive_cbf_rl_amd/g1_amd_dodge_replay.py \
  --upstream-root /data/Data14TB/03competition/amd-physical-ai-showcase/.vendor/perceptive_cbf_rl \
  --output-dir /data/Data14TB/03competition/amd-physical-ai-showcase/results/perceptive_cbf_rl_amd
```

`amd_pacman_cbf_smoke.py` remains available as a small predictive-CBF tensor
test; it is independent of the whole-body ONNX replay.
