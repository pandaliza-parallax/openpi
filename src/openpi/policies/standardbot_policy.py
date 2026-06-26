"""Policy transforms for the StandardBots RO1 (single-arm, 6-DoF + gripper).

Robot spec (see parallax/bot-ctrl/standardbot_ctrl/standardbot.py):
  - 6-DoF arm + 1-DoF parallel gripper (width 0..0.145 m).
  - Single wrist-mounted RealSense RGB camera (native 640x480; the model resizes to 224x224).
  - State  (7-dim): [joint_1..joint_6 (rad), gripper_width (m)].
  - Action (7-dim): [joint_1..joint_6 target (rad), gripper_width (m)].

If you switch to Cartesian/tooltip control, change the state/action layout here and in
``LeRobotStandardbotDataConfig`` accordingly (e.g. [xyz, rot6d, gripper]).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# StandardBots RO1: 6 arm joints + 1 gripper = 7.
STATE_DIM = 7
ACTION_DIM = 7


def make_standardbot_example() -> dict:
    """Creates a random input example for the StandardBot policy (used for inference smoke tests)."""
    return {
        "observation/state": np.random.rand(STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "insert the peg into the hole",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # LeRobot stores images as (C, H, W); the model wants (H, W, C).
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class StandardbotInputs(transforms.DataTransformFn):
    """Converts StandardBot observations into the format the model expects.

    Used for both training and inference. The RO1 has a single camera, so we feed it as the
    base (third-person) view and zero-pad the two wrist views the pi0 family supports.
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                # Single camera -> base view. Pad the wrist views with zeros.
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask out the padded views (pi0_fast does not mask, so keep True there).
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only present during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Language instruction.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class StandardbotOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the StandardBot action space (inference only)."""

    def __call__(self, data: dict) -> dict:
        # The model pads actions to its internal dim; return only the real 7 dims.
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
