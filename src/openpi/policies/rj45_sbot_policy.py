"""Policy transforms for the sbot RJ45 full-episode GS dataset (Parallax).

Produced by newton-cabling/tools/render_batch.sh (record_sbot_scene_gs.py --dump) +
tools/datagen_to_lerobot.py. Two GS cameras (front + rigid eye-in-hand wrist); proprio is the
hand grasp-point pose in the robot base frame; actions are base-frame deltas to the next frame.
Episodes cover the FULL task: home hold -> approach (gripper opens/closes) -> insertion.

  state   (10-D): [eef_pos(3), eef_rot6d(6), gripper(1)]  absolute, base frame; gripper 0..1
  actions  (7-D): [dpos(3), drotvec(3), gripper(1)]       delta to next frame; gripper = next cmd
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 10
ACTION_DIM = 7


def make_rj45_sbot_example() -> dict:
    """Random input example (inference smoke tests)."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the ethernet cable and plug it into the jack",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # LeRobot stores (C,H,W); model wants (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Rj45SbotInputs(transforms.DataTransformFn):
    """Front cam -> base view, wrist cam -> left wrist; right wrist zero-padded."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Rj45SbotOutputs(transforms.DataTransformFn):
    """Return the real 7 action dims (the model pads to its internal action dim)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
