# RoboCasa Mobile Manipulation Mainline

## Positioning

The competition flagship is a RoboCasa-derived household manipulation track on AMD Radeon GPUs and ROCm. The existing RoboCasa365 16-task result is the verified household baseline. The mobile extension is a separate experiment until its action contract, data audit, and fixed evaluation pass.

The project must not call a scripted-base rollout an end-to-end mobile policy. A scripted base is an environment and interface gate only. The final mobile policy must expose base and arm actions through one declared action vector.

## Core Task Family

The first release uses three task families instead of attempting every RoboCasa task at once:

1. `FetchAndPlace`: navigate to an object, approach, grasp, lift, transport, and place.
2. `OpenRetrievePlace`: open a cabinet or fridge, retrieve an item, and place it at a target.
3. `MultiStationTransport`: move between two stations and complete a long-horizon transport and release.

The recovery condition is shared across the task family:

- grasp failure -> re-localize and retry;
- object slip -> detect loss and re-grasp;
- blocked path -> re-plan and continue.

## Policy Contract

The intended SmolVLA action vector contains:

```text
[base_vx, base_vy, base_yaw_rate, arm_joint_targets..., gripper_state]
```

The exact `action_dim`, joint order, control frequency, clipping, normalization statistics, camera order, and language prompt are stored in the run manifest. No mobile score is publishable while any of these fields are implicit.

## Evidence Contract

Every episode records:

- `navigation`, `approach`, `contact`, `grasp`, `lift`, `transport`, `place`, `release`, `recovery`, and `final_success`;
- fixed seed, task name, simulator commit, asset manifest, model checkpoint SHA, stats SHA, and environment versions;
- one success or failure video when video capture is enabled;
- raw `eval_info.json` plus a summary table.

The formal target is three tasks x 50 episodes. A 10-episode panel is a promotion gate, not the final score.

## Training Stages

1. Freeze the official RoboCasa household baseline.
2. Add the mobile action interface and run reset/render/controller tests.
3. Generate expert trajectories and audit action alignment, cameras, finite values, and strict success.
4. Train SmolVLA with a real progress log and checkpoint manifest.
5. Run the 10-episode promotion panel.
6. Run the fixed 3 x 50 formal evaluation.
7. Publish the English submission bundle and bilingual evidence website.

## AMD Measurement

Record GPU model, ROCm, PyTorch/JAX versions, wall time, GPU utilization, VRAM, throughput, batch size, worker count, precision, and video-on/video-off cost. The 395 remains the long-evaluation and asset/debug device; W7900 remains the large training device unless a run manifest says otherwise.

## Publication Boundary

The public website can expose validated videos, result JSON, SHA files, and migration notes. Competition-specific adapter code remains in the private source-of-truth repository until the planned release window. No smoke result, teacher-forced loss, or scripted recovery is presented as final policy success.
