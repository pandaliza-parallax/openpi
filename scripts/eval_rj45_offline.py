"""Offline open-loop action-error eval for the RJ45 sbot policy.

Runs one or more trained checkpoints over a held-out LeRobot dataset and reports the
error between the predicted action chunk and the ground-truth actions. This is an
open-loop proxy (no robot / sim): it measures how well each checkpoint reproduces the
recorded actions, which is enough to rank checkpoints and spot overfitting.

Usage:
    uv run scripts/eval_rj45_offline.py \
        --config pi05_rj45_sbot_lora \
        --exp rj45_difix_run1 \
        --eval-repo-id Parallax-Worlds/rj45_sbot_difix_eval \
        --steps 5000 10000 14999
"""

import dataclasses
import pathlib

import jax
import numpy as np
import tyro

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# Raw LeRobot keys -> the observation/* keys the policy's input transform expects.
_REPACK = {
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/state": "state",
    "prompt": "prompt",
}

# 7-D action layout: [dpos(3), drotvec(3), gripper(1)].
_GROUPS = {"dpos": slice(0, 3), "drot": slice(3, 6), "gripper": slice(6, 7)}


def _to_numpy(x):
    arr = np.asarray(x)
    return arr


def _normalize_actions(x, stats):
    """Quantile-normalize actions to ~[-1, 1] using the same formula as transforms.Normalize."""
    d = x.shape[-1]
    q01, q99 = np.asarray(stats.q01)[:d], np.asarray(stats.q99)[:d]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def main(
    config: str = "pi05_rj45_sbot_lora",
    exp: str = "rj45_difix_run1",
    eval_repo_id: str = "Parallax-Worlds/rj45_sbot_difix_eval",
    steps: tuple[int, ...] = (5000, 10000, 14999),
    max_frames: int | None = None,
    seed: int = 0,
    csv: str | None = None,
) -> None:
    train_config = _config.get_config(config)
    ckpt_root = pathlib.Path(train_config.checkpoint_base_dir) / config / exp

    # Point the data pipeline at the held-out eval dataset, reusing every other setting
    # (action_sequence_keys, prompt_from_task, quantile norm) exactly as in training.
    eval_data_factory = dataclasses.replace(train_config.data, repo_id=eval_repo_id)
    data_config = eval_data_factory.create(train_config.assets_dirs, train_config.model)
    action_horizon = train_config.model.action_horizon

    # Norm stats are saved in each checkpoint under the *training* asset_id (not the eval repo).
    train_data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    asset_id = train_data_config.asset_id

    # Raw samples: image, wrist_image, state, and actions already stacked into a
    # (action_horizon, action_dim) chunk via delta_timestamps; prompt from the task.
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, train_config.model)
    n = len(dataset)
    order = np.random.default_rng(seed).permutation(n)
    if max_frames is not None:
        order = order[:max_frames]
    print(f"Eval dataset {eval_repo_id}: {n} frames; scoring {len(order)} of them.\n")

    csv_rows = []  # (step, frame_index, nmae, raw_mae, dpos, drot, gripper) if --csv is set
    results = {}
    for step in steps:
        ckpt_dir = ckpt_root / str(step)
        if not ckpt_dir.exists():
            print(f"[skip] checkpoint {step} not found at {ckpt_dir}")
            continue
        print(f"Loading checkpoint {step} ...")
        policy = _policy_config.create_trained_policy(
            train_config,
            ckpt_dir,
            # Repack raw LeRobot keys -> observation/* before the policy's own transforms.
            repack_transforms=_transforms.Group(inputs=[_transforms.RepackTransform(_REPACK)]),
        )
        action_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", asset_id)["actions"]

        abs_err = []  # per-frame mean |pred - gt| over valid (non-padded) timesteps, normalized space
        raw_err = []  # same, raw action units (secondary, for reference)
        group_err = {g: [] for g in _GROUPS}
        for i in order:
            sample = dataset[int(i)]
            obs = {k: _to_numpy(v) for k, v in sample.items()}
            gt = obs.pop("actions").astype(np.float32)  # (action_horizon, 7)
            pad = obs.pop("actions_is_pad", None)
            valid = ~np.asarray(pad) if pad is not None else np.ones(gt.shape[0], dtype=bool)
            if not valid.any():
                continue
            pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)  # (action_horizon, 7)
            # Primary metric: error in normalized action space (scale-free, comparable across dims).
            err_n = np.abs(_normalize_actions(pred[valid], action_stats) - _normalize_actions(gt[valid], action_stats))
            err_raw = np.abs(pred[valid] - gt[valid])
            abs_err.append(err_n.mean())
            raw_err.append(err_raw.mean())
            per_group = {g: float(err_n[:, sl].mean()) for g, sl in _GROUPS.items()}
            for g in _GROUPS:
                group_err[g].append(per_group[g])
            if csv is not None:
                csv_rows.append(
                    (step, int(i), float(err_n.mean()), float(err_raw.mean()), *[per_group[g] for g in _GROUPS])
                )

        results[step] = {
            "nmae": float(np.mean(abs_err)),
            "raw_mae": float(np.mean(raw_err)),
            **{g: float(np.mean(v)) for g, v in group_err.items()},
        }
        r = results[step]
        print(
            f"  step {step}:  normMAE={r['nmae']:.4f}  (raw={r['raw_mae']:.5f})  |  "
            f"dpos={r['dpos']:.4f}  drot={r['drot']:.4f}  gripper={r['gripper']:.4f}\n"
        )

    if results:
        print("=" * 72)
        print("Normalized action MAE (lower is better); raw MAE in action units for reference.")
        print(f"{'step':>8} {'normMAE':>10} {'raw_mae':>10} {'dpos':>10} {'drot':>10} {'gripper':>10}")
        print("-" * 72)
        for step, r in results.items():
            print(
                f"{step:>8} {r['nmae']:>10.4f} {r['raw_mae']:>10.5f} "
                f"{r['dpos']:>10.4f} {r['drot']:>10.4f} {r['gripper']:>10.4f}"
            )
        best = min(results.items(), key=lambda kv: kv[1]["nmae"])
        print("-" * 72)
        print(f"Lowest normalized MAE: step {best[0]} ({best[1]['nmae']:.4f})")

    if csv is not None and csv_rows:
        with open(csv, "w") as f:
            f.write("step,frame_index,nmae,raw_mae,dpos,drot,gripper\n")
            for row in csv_rows:
                f.write("{},{},{:.6f},{:.8f},{:.6f},{:.6f},{:.6f}\n".format(*row))
        print(f"\nPer-frame errors written to {csv} ({len(csv_rows)} rows).")


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "gpu")
    tyro.cli(main)
