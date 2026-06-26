"""Convert StandardBots RO1 demonstrations into a LeRobot dataset for openpi fine-tuning.

This is a SCAFFOLD. You have no demos saved yet, so the `iter_episodes` function below
contains the only TODO you need to fill in: yield one episode at a time from however you
end up logging teleop / rollout data (the bot-ctrl driver exposes joints, tooltip pose, and
gripper width; see parallax/bot-ctrl/standardbot_ctrl/standardbot.py).

Robot spec (matches src/openpi/policies/standardbot_policy.py):
  - 1 wrist RealSense RGB camera, native 640x480 (stored here; the model resizes to 224x224).
  - state  (7-dim): [joint_1..joint_6 (rad), gripper_width (m)]
  - action (7-dim): [joint_1..joint_6 target (rad), gripper_width (m)]   # absolute joint targets

Usage:
  uv run examples/standardbot/convert_standardbot_data_to_lerobot.py --data_dir /path/to/raw/demos
  # push to the Hub (optional):
  uv run examples/standardbot/convert_standardbot_data_to_lerobot.py --data_dir ... --push_to_hub

The output dataset is written to $HF_LEROBOT_HOME (default ~/.cache/huggingface/lerobot) under
REPO_NAME, which must match the `repo_id` in the pi05_standardbot / pi05_standardbot_lora configs.
"""

import dataclasses
import shutil
from typing import Iterator

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

# Must match `repo_id` in src/openpi/training/config.py (pi05_standardbot*).
REPO_NAME = "parallax/standardbot"

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
STATE_DIM = 7  # 6 joints + gripper width
ACTION_DIM = 7  # 6 joint targets + gripper width


@dataclasses.dataclass
class Frame:
    """One timestep of a demonstration."""

    image: np.ndarray  # uint8 (H, W, 3) RGB from the RealSense
    state: np.ndarray  # float32 (7,) [6 joints rad, gripper width m]
    action: np.ndarray  # float32 (7,) [6 joint targets rad, gripper width m]
    task: str  # language instruction, e.g. "insert the peg into the hole"


def iter_episodes(data_dir: str) -> Iterator[list[Frame]]:
    """Yield one episode (a list of Frames) at a time from your raw demo logs.

    TODO(parallax): implement this for your logging format. For example, if you log each
    teleop episode as an .npz with arrays ``rgb`` (T,H,W,3), ``joints`` (T,6),
    ``gripper`` (T,), ``action`` (T,7) and a ``task`` string:

        import pathlib
        for path in sorted(pathlib.Path(data_dir).glob("*.npz")):
            d = np.load(path, allow_pickle=True)
            task = str(d["task"])
            frames = []
            for t in range(len(d["rgb"])):
                state = np.concatenate([d["joints"][t], d["gripper"][t:t+1]]).astype(np.float32)
                frames.append(Frame(
                    image=d["rgb"][t].astype(np.uint8),
                    state=state,
                    action=d["action"][t].astype(np.float32),
                    task=task,
                ))
            yield frames
    """
    raise NotImplementedError(
        f"Implement iter_episodes() for your demo format under {data_dir!r}. "
        "See the docstring for an .npz example."
    )


def main(data_dir: str, *, push_to_hub: bool = False) -> None:
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # OpenPi expects proprio in `state` and actions in `actions`; images are dtype "image".
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="standardbots_ro1",
        fps=10,  # set to your teleop/control rate
        features={
            "image": {
                "dtype": "image",
                "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_ep = 0
    for episode in iter_episodes(data_dir):
        for frame in episode:
            assert frame.state.shape == (STATE_DIM,), frame.state.shape
            assert frame.action.shape == (ACTION_DIM,), frame.action.shape
            dataset.add_frame(
                {
                    "image": frame.image,
                    "state": frame.state,
                    "actions": frame.action,
                    "task": frame.task,
                }
            )
        dataset.save_episode()
        n_ep += 1

    print(f"Wrote {n_ep} episodes to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["standardbots", "ro1", "parallax"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
