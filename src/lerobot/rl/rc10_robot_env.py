import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from lerobot.envs.configs import HILSerlRobotEnvConfig

from lerobot.robots import make_robot_from_config

from lerobot.processor import RobotObservation

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.INFO)




class RC10RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._setup_spaces()

    def _get_observation(self) -> RobotObservation:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        agent_pos = np.array([obs_dict["tcp.x"], obs_dict["tcp.y"], obs_dict["tcp.z"], obs_dict["gripper.state"]])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": agent_pos, "pixels": images}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=-10,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        self.robot.reset()
        start_time = time.perf_counter()
        log_say("Reset the environment.", play_sounds=True)

        # if self.reset_pose is not None:
        #     log_say("Reset the environment.", play_sounds=True)
        #     # reset_follower_position(self.robot, np.array(self.reset_pose))
        #     log_say("Reset the environment done.", play_sounds=True)

        precise_sleep(max(self.reset_time_s - (time.perf_counter() - start_time), 0.0))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        # joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}
        action_dict = {}
        action_dict["tcp.delta_x"] = action[0]
        action_dict["tcp.delta_y"] = action[1]
        action_dict["tcp.delta_z"] = action[2]
        action_dict["gripper.state"] = action[3]
        self.robot.send_action(action_dict)

        obs = self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Display robot camera feeds."""
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions



def make_rc10_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None

    env = RC10RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
    )

    return env, teleop_device
