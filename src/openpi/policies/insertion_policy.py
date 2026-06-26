"""Policy transforms for the GS-sim peg-insertion teacher dataset (Parallax).

Produced by gs-sim-vla/sample/sample_sac_teacher.py + raw_to_lerobot.py. Single third-person
GS camera; proprio is the peg pose in the hole frame; actions are task-space delta setpoints.

  state   (7-D): [x, y, z, qw, qx, qy, qz]  (peg pose in hole frame; m, unit quat)
  actions (6-D): [dx, dy, dz, drx, dry, drz] (task-space delta setpoint; m, rad)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 7
ACTION_DIM = 6


def make_insertion_example() -> dict:
    """Random input example for the insertion policy (inference smoke tests)."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "insert the peg into the hole",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # LeRobot stores (C,H,W); model wants (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class InsertionInputs(transforms.DataTransformFn):
    """Single camera -> base view; the two wrist views are zero-padded."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class InsertionOutputs(transforms.DataTransformFn):
    """Return the real 6 action dims (the model pads to its internal action dim)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
