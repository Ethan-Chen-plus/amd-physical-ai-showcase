# AMD Migration Playbook

This playbook defines the reproducibility gates used by Datawhale-EAI when
porting Physical AI projects to AMD Radeon GPUs and ROCm. Import, rendering,
training, inference, and closed-loop evaluation are treated as distinct system
milestones.

The companion [Physical AI migration engineering blog](migration-blog.html)
documents the full workflow across DISCOVERSE, RoboCasa365, DexJoCo, JAX, and
policy runtimes. This file provides the compact executable acceptance contract.

## Acceptance gates

1. **Environment:** ROCm, PyTorch or JAX, and simulator dependencies import successfully.
2. **Simulation:** Official scenes reset, step, render, and export MP4 footage.
3. **Data:** Expert trajectories or official datasets pass shape, temporal-order, episode-boundary, and metadata audits.
4. **Training:** The official entry point starts, records metrics, saves checkpoints, and resumes from persistent storage.
5. **Inference:** Evaluation uses the checkpoint's normalization statistics and the matching observation/action bridge.
6. **Evaluation:** Fixed tasks and seeds produce success metrics, stage outcomes, videos, JSON records, and checksums.

## Validated ports

### ROCm JAX and OpenPI

- Installation follows the [AMD ROCm JAX 0.10.0 guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html?fam=all&os=linux&jax-ver=0.10.0&i=docker&w=compute). The reference image is `rocm/jax:rocm7.14-jax0.10.0-py3.12`.
- The RoboCasa365 Pi0.5 path validates the GPU backend, 75k checkpoint, tokenizer, normalization statistics, isolated Gemma softmax path, and OSMesa video output.
- Pi0.5 completes the matched `16 tasks x 50 episodes` protocol with `142/800 = 17.75%`, including per-task JSON, videos, and SHA-256 records.
- DexJoCo uses an isolated ROCm JAX 0.10 environment with native GPU preflight and Orbax restore before task evaluation.
- The runtime contract pins ROCm, the JAX plugin, PJRT, Python, tokenizer, and checkpoint versions together.

### DISCOVERSE

- Upstream: [DISCOVERSE](https://github.com/discoverse-dev/DISCOVERSE).
- Core runtime gates pass `18/18`, AIRBOT passes `12/12`, MMK2 passes `8/8`, and the strict `block_bridge_place` expert replay passes `31/31` with `500` expert episodes.
- The port covers MuJoCo tasks, expert trajectories, policy entry points, MP4 output, ROS 2, LiDAR, 3DGS, and ACT, Diffusion Policy, PPO, and RDT runtime paths.
- RealSense, gamepad, ROS 2 peripheral, and physical-robot paths remain tied to their corresponding hardware.

### RoboCasa365

- AMD Ryzen AI MAX+ 395 runs the official assets, scene reset, rendering, policy inference, and video export.
- Pi0.5 and GR00T use the same `16 tasks x 50 episodes` protocol: Pi0.5 reaches `142/800`, and GR00T reaches `230/800`.
- Aggregate JSON, task statistics, four-view videos, and checksums are published with the showcase.
- The high-resolution `CloseFridge` showcase records synchronized center, left, right, and wrist views at `1920x1080@20fps`.

### Every Embodied VLA

- SmolVLA, Pi0, and ACT include standard training, protected training, Notebook-native inference, strict evaluation, and video export entry points.
- Reproducible reference results are SmolVLA `57/60` and Pi0 `12/14`, with ACT training logs and checkpoint artifacts indexed alongside them.
- Each published checkpoint is linked to its training recipe, evaluation JSON, video outputs, and SHA-256 manifest.

### RoboWits

- Radeon PRO W7900 runs the official 16-D ACT configuration with a `100k` training target.
- Checkpoints are mirrored to Hugging Face every `5k` steps with training metrics and recovery metadata.
- Persistent datasets, scripts, logs, and checkpoints live under the cloud PVC workspace so instance recreation does not remove experiment state.

## Common AMD porting issues

| Layer | Typical issue | Engineering response |
|---|---|---|
| Device | The `torch.cuda` API name remains while HIP is the active backend | Record `torch.version.hip`, `rocminfo`, and the exact GPU model |
| Rendering | Headless execution and unavailable CUDA denoisers | Treat offscreen rendering and MP4 export as first-class outputs |
| Data | Misaligned action chunks, episode boundaries, or normalization statistics | Audit temporal order, dimensions, terminal padding, and statistics before training |
| JAX | Wheel, MIOpen, attention, or shared-memory constraints | Pin the full runtime matrix and record operator-level compatibility |
| Assets | Licensed BlenderKit or task-specific assets | Preserve the official task definition and document asset requirements |
| Evaluation | Warm reuse, process topology, or video paths alter random state | Fix task order, process topology, seeds, and output conventions |

## Publication contract

- The public site contains sanitized results, videos, configuration summaries, reproduction commands, and upstream links.
- Private integration repositories hold platform bridges, patches, licensed asset manifests, and detailed infrastructure logs.
- Tokens, machine addresses, personal paths, private assets, and unauthorized proxy meshes are excluded.
- Every published checkpoint, JSON file, and video has a SHA-256 record.

## DexJoCo and Pi0.5

Project page: [DexJoCo](https://dexjoco.github.io/). Source code:
[brave-eai/dexjoco](https://github.com/brave-eai/dexjoco). Policy weights:
[DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05).

The native AMD ROCm JAX 0.10 path passes GPU preflight and Orbax restoration.
The single-task `water_plant` run reaches `4/4`. The fixed-seed official 11-task
panel reaches `5/11`, while the deterministic success-seed archive contains a
successful trace for 10 of 11 tasks. Each result retains its own protocol,
denominator, videos, and task-level records.
