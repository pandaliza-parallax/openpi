# StandardBots RO1 — pi0.5 fine-tuning (Parallax)

End-to-end recipe to fine-tune **π₀.₅ (pi05)** on the StandardBots RO1 (6-DoF arm + gripper,
single wrist RealSense). Built for a single 32GB GPU (RTX 5090) using the **JAX** path.

## Robot spec

| | layout | dim |
|---|---|---|
| Camera | 1 wrist RealSense RGB (640×480, resized to 224×224 by the model) | — |
| `state` | `[joint_1..joint_6 (rad), gripper_width (m)]` | 7 |
| `actions` | `[joint_1..joint_6 target (rad), gripper_width (m)]` (absolute joint targets) | 7 |

Defined in [`src/openpi/policies/standardbot_policy.py`](../../src/openpi/policies/standardbot_policy.py)
and `LeRobotStandardbotDataConfig` in [`src/openpi/training/config.py`](../../src/openpi/training/config.py).
If you move to Cartesian/tooltip control, change the state/action layout in both places.

## Configs

| Config | What trains | Fits 32GB? |
|---|---|---|
| `pi05_standardbot` | full fine-tune (needs ~70GB — multi-GPU) | ✗ |
| `pi05_standardbot_lora` | **LoRA** on Gemma VL + action expert, full-trains SigLIP vision encoder, freezes the 2.9B LLM (~467M / 13.7% trainable) | ✓ |

Use **`pi05_standardbot_lora`** on your single 5090. To shrink trainable params further you can
also freeze the vision encoder; ask if you want that variant added.

## 1. Collect & convert data

You have no demos yet. Once you log teleop/rollout episodes, fill in `iter_episodes()` in
[`convert_standardbot_data_to_lerobot.py`](convert_standardbot_data_to_lerobot.py) (a worked
`.npz` example is in its docstring), then:

```bash
uv run examples/standardbot/convert_standardbot_data_to_lerobot.py --data_dir /path/to/raw/demos
```

This writes a LeRobot dataset to `$HF_LEROBOT_HOME` under `parallax/standardbot` (matching the
config `repo_id`). Each frame stores `image`, `state`, `actions`, and a `task` string (the
language instruction — pulled into the prompt via `prompt_from_task=True`).

## 2. Compute normalization stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_standardbot_lora
```

## 3. Train (LoRA)

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_standardbot_lora --exp-name=ro1_v1 --overwrite
```

Checkpoints land in `checkpoints/pi05_standardbot_lora/ro1_v1/`. Progress logs to the console
and Weights & Biases. The pi05 base weights auto-download from `gs://openpi-assets` on first run.

## 4. Serve & run inference

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_standardbot_lora \
  --policy.dir=checkpoints/pi05_standardbot_lora/ro1_v1/20000
```

The server listens on port 8000. From your robot runtime, send observations with keys
`image` (HWC uint8), `state` (7,), and `prompt`, then apply the returned `actions` chunk. See
[`docs/remote_inference.md`](../../docs/remote_inference.md) for a minimal websocket client.

## Notes
- **Delta actions:** the RO1's `move_joints` takes absolute joint targets, so the config converts
  the 6 joint dims to deltas (gripper stays absolute) to match the pi05 base. Set
  `use_delta_joint_actions=False` if you ever log delta actions directly.
- **PyTorch path:** LoRA is JAX-only in openpi (and torch here is cu126, which doesn't support
  the 5090's sm_120) — stay on JAX.
