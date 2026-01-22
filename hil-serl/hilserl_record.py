from lerobot.robots.rc10 import RC10Config
from lerobot.teleoperators.haptic.config_haptic import HapticConfig
from lerobot.envs.configs import (HILSerlRobotEnvConfig, 
                                  HILSerlProcessorConfig,
                                  ImagePreprocessingConfig,
                                  RewardClassifierConfig,
                                  ObservationConfig,
                                  GripperConfig,
                                  ResetConfig)
from lerobot.rl.gym_manipulator import RobotEnv, make_robot_env
from lerobot.rl.my_robot_env import RC10RobotEnv, make_rc10_robot_env
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.utils import TeleopEvents
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.utils import log_say
import numpy as np

# ======================================================================================

FPS = 30

ROBOT_ACTION_SCALE = 100

EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 5

TELEOP_ACTION_SACLE = 100
TELEOP_IP = "127.0.0.1"
TELEOP_PORT = 8081

USE_GRIPPER = True
GRIPPER_PENALTY = 0.05

CLASSIFIER_PATH = ""
SUCCESS_THRESHOLD = 0.5
SUCCESS_REWARD = 1

CONTROL_MODE = "haptic"
MAX_GRIPPER_POS = 2

NAME = "rc10"
TASK_DESCRIPTION = "TASK"

MAX_EPISODES = 3
MAX_STEPS_PER_EPISODE = 150

DEVICE = "cuda"

HF_REPO_ID = "local/test_demo"

# ======================================================================================

camera_config = {"front": OpenCVCameraConfig(
        index_or_path=0, width=640, height=480, fps=FPS)
    }

robot_config = RC10Config(action_scale=ROBOT_ACTION_SCALE,
                          cameras=camera_config)

teleop_config = HapticConfig(ip=TELEOP_IP,
                             port=TELEOP_PORT,
                             scale=TELEOP_ACTION_SACLE)

# ======================================================================================

reset_config = ResetConfig(reset_time_s=RESET_TIME_SEC,
                           control_time_s=EPISODE_TIME_SEC,
                           terminate_on_success=True)

gripper_config = GripperConfig(use_gripper=USE_GRIPPER,
                               gripper_penalty=GRIPPER_PENALTY)

image_proc_config = ImagePreprocessingConfig()

observation_config = ObservationConfig(add_joint_velocity_to_observation=False,
                                       add_current_to_observation=False,
                                       display_cameras=True)

reward_classifier_config = RewardClassifierConfig(pretrained_path=CLASSIFIER_PATH,
                                                  success_threshold=SUCCESS_THRESHOLD,
                                                  success_reward=SUCCESS_REWARD)

# ======================================================================================

hilserl_processor_config = HILSerlProcessorConfig(control_mode=CONTROL_MODE,
                                                  observation=observation_config,
                                                  image_preprocessing=image_proc_config,
                                                  gripper=gripper_config,
                                                  reward_classifier=reward_classifier_config,
                                                  max_gripper_pos=MAX_GRIPPER_POS)

hilserl_robot_env_config = HILSerlRobotEnvConfig(task=TASK_DESCRIPTION,
                                                 fps=FPS,
                                                #  features=None,
                                                #  features_map=None,
                                                 max_parallel_tasks=1,
                                                 disable_env_checker=True,
                                                 robot=robot_config,
                                                 teleop=teleop_config,
                                                 processor=hilserl_processor_config,
                                                 name=NAME)

env, teleop_device = make_rc10_robot_env(hilserl_robot_env_config)

# Configure the dataset features
# action_features = hw_to_dataset_features(env.robot.action_features, "action")
# obs_features = hw_to_dataset_features(env.robot.observation_features, "observation", use_video=True)
# dataset_features = {**action_features, **obs_features, "next.reward": {"dtype": "float32", "shape": (1,)}, "next.done": {"dtype": "bool", "shape": (1,)}}
# Create the dataset where to store the data

dataset_features = {'action': {'dtype': 'float32', 'shape': (4,), 'names': ['tcp.delta_x', 'tcp.delta_y', 'tcp.delta_z', 'gripper.state']}, 
                    'observation.state': {'dtype': 'float32', 'shape': (1,3), 'names': ['tcp.x', 'tcp.y', 'tcp.z']}, 
                    'observation.image.front': {'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}, 
                    'next.reward': {'dtype': 'float32', 'shape': (1,)}, 
                    'next.done': {'dtype': 'bool', 'shape': (1,)}}

print(dataset_features)

dataset = LeRobotDataset.create(
    repo_id="local/test_data",
    fps=FPS,
    features=dataset_features,
    robot_type=env.robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# ======================================================================================

def make_policy_obs(obs, device: torch.device = "cpu"):
    return {
        "observation.state": torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(device),
        **{
            f"observation.image.{k}": torch.from_numpy(obs["pixels"][k]).float().unsqueeze(0).to(device)
            for k in obs["pixels"]
        },
    }

# ======================================================================================

for episode in range(MAX_EPISODES):

    obs, _info = env.reset()
    episode_reward = 0.0
    step = 0
    episode_transitions = []

    log_say(f"Starting episode {episode + 1}")

    while step < MAX_STEPS_PER_EPISODE:

        policy_obs = make_policy_obs(obs, device=DEVICE)
    
        action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # In HIL-SERL, human interventions come from the teleop device
        is_intervention = False
        if hasattr(teleop_device, "get_teleop_events"):
            # Real intervention detection from teleop device
            teleop_events = teleop_device.get_teleop_events()
            is_intervention = teleop_events.get(TeleopEvents.IS_INTERVENTION, False)
            if is_intervention:
                action_dict = teleop_device.get_action()
                action = [action_dict["tcp.delta_x"], action_dict["tcp.delta_y"], action_dict["tcp.delta_z"], action_dict["gripper.state"]]

        # Step environment
        next_obs, _env_reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        policy_next_obs = make_policy_obs(next_obs, device=DEVICE)
        reward = _env_reward

        if reward >= 1.0 and not done:  # success detected! halt episode
            terminated = True
            done = True

        # Store transition with intervention metadata
        transition = {
            "state": policy_obs,
            "action": action,
            "reward": float(reward) if hasattr(reward, "item") else reward,
            "next_state": policy_next_obs,
            "done": done,
            "truncated": truncated,
            "complementary_info": {
                "is_intervention": is_intervention,
            },
        }

        for k, v in policy_obs.items():
            if "image" in k:
                policy_obs[k] = v[0].cpu()/255
            else: 
                policy_obs[k] = v.cpu()

        dataset_transition = {
            **policy_obs,
            "action": action,
            "next.reward": np.array([reward], dtype=np.float32),
            "next.done": np.array([done], dtype=bool),
            "task": TASK_DESCRIPTION
        }

        dataset.add_frame(dataset_transition)
        episode_transitions.append(transition)

        episode_reward += reward
        step += 1

        obs = next_obs

        if done:
            break

    dataset.save_episode()
log_say("Stop recording")
dataset.finalize()