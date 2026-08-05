#!/usr/bin/env python3
"""Contract and movement tests for the portable G1 ONNX replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[1] / ".vendor" / "perceptive_cbf_rl"
SPEC = importlib.util.spec_from_file_location("g1_replay", HERE / "g1_amd_dodge_replay.py")
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


class G1OnnxReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = R.load_upstream(UPSTREAM)
        cls.model = mujoco.MjModel.from_xml_string(
            R.build_replay_scene(cls.contract["robot_xml"])
        )
        cls.session = ort.InferenceSession(
            str(cls.contract["checkpoint"]), providers=["CPUExecutionProvider"]
        )

    def test_exact_onnx_contract(self) -> None:
        self.assertEqual(self.session.get_inputs()[0].shape, [1, 960])
        self.assertEqual(self.session.get_outputs()[0].shape, [1, 29])
        self.assertEqual(len(self.contract["joint_names"]), 29)

    def test_depth_changes_policy_output(self) -> None:
        input_name = self.session.get_inputs()[0].name
        far = np.ones((1, 960), dtype=np.float32)
        far[:, :384] = 0.0
        looming = far.copy()
        for offset in range(4):
            base = 384 + offset * 144
            looming[0, base + 4 * 16 + 8] = 0.15
        far_action = self.session.run(None, {input_name: far})[0]
        looming_action = self.session.run(None, {input_name: looming})[0]
        self.assertEqual(looming_action.shape, (1, 29))
        self.assertTrue(np.isfinite(looming_action).all())
        self.assertGreater(float(np.linalg.norm(looming_action - far_action)), 0.1)

    def test_whole_body_trace_has_no_root_shift(self) -> None:
        run = R.run_scenario(
            self.model,
            self.contract,
            self.session,
            R.SCENARIOS[0],
            duration_s=3.0,
            control_hz=50,
            max_joint_speed=3.5,
            target_alpha=0.35,
        )
        trace = run["trace"]
        ranges = np.ptp(trace["qpos"], axis=0)
        self.assertEqual(trace["observations"].shape[1], 960)
        self.assertEqual(trace["actions"].shape[1], 29)
        self.assertTrue(np.isfinite(trace["actions"]).all())
        self.assertGreaterEqual(int(np.count_nonzero(ranges >= 0.05)), 20)
        self.assertEqual(run["summary"]["inference_ticks"], 150)


if __name__ == "__main__":
    unittest.main(verbosity=2)
