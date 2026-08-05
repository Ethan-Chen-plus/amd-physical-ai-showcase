# Unitree G1 whole-body ONNX replay

This directory ports the deployment contract from
[`lzyang2000/perceptive_cbf_rl`](https://github.com/lzyang2000/perceptive_cbf_rl)
into a deterministic, portable MuJoCo replay.

## Implemented path

`g1_amd_dodge_replay.py` runs the pinned upstream artifacts directly:

- `deploy/ckpts/dodge_link_cbf.onnx`;
- `src/assets/robots/unitree_g1/xmls/g1.xml` and its mesh assets;
- the upstream 4-frame, term-major proprio history;
- synthetic ball-only 9x16 metric depth at offsets `(0, 3, 8, 18)`;
- the exact `384 proprio + 576 depth = 960` observation;
- ONNX inference with a 29-D output;
- `target = DEFAULT_POS + action * ACTION_SCALE` in official joint order.

The generated video contains three synchronized G1 views and one policy-I/O
view. The policy-I/O view shows all four depth frames and the largest ONNX
outputs. The trace archive stores every observation, action, joint target,
joint position, ball position, and geometry-clearance sample.

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

Render the complete replay:

```bash
MUJOCO_GL=egl \
  /data/Data14TB/envs/pacman-g1-replay/bin/python \
  /data/Data14TB/03competition/amd-physical-ai-showcase/code/perceptive_cbf_rl_amd/g1_amd_dodge_replay.py \
  --upstream-root /data/Data14TB/03competition/amd-physical-ai-showcase/.vendor/perceptive_cbf_rl \
  --output-dir /data/Data14TB/03competition/amd-physical-ai-showcase/results/perceptive_cbf_rl_amd
```

Outputs:

- `unitree-g1-onnx-whole-body-dodge-1080p.mp4`: 1920x1080 multi-view replay;
- `g1-onnx-whole-body-trace.npz`: complete policy input/output trace;
- `eval_info.json`: movement, clearance, contract, and limitation evidence;
- `run_manifest.json`: command, environment versions, artifact SHA256 values.

## Portable simulation boundary

The upstream generated free-base MJCF exposes raw torque motors, while the
training and native simulation path uses mjlab/MuJoCo-Warp actuator and balance
logic. The included dynamic probe applies the baked deployment PD gains to the
generated free-base model and records whether the nominal stand remains stable.
When that probe collapses, the evidence replay holds the official free base at
its nominal pose and applies every one of the 29 ONNX joint targets with a
documented target filter and rate limit. No root translation is used.

This mode validates the complete observation, inference, joint-order, scaling,
target, and rendering path. `eval_info.json` records the free-base probe and the
portable replay mode separately.

`amd_pacman_cbf_smoke.py` remains available as a small predictive-CBF tensor
test; it is independent of the whole-body ONNX replay.
