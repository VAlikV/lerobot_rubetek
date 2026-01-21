"""
You can also use the CLI to record data. To see the required arguments, run:
lerobot-record --help
"""
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

from lerobot.robots.rc10.rc10 import RC10
from lerobot.robots.rc10.config_rc10 import RC10Config
from lerobot.teleoperators.haptic.config_haptic import HapticConfig
from lerobot.teleoperators.haptic.teleop_haptic import Haptic

# from lerobot_robot_rc10.config_rc10 import RC10Config
# from lerobot_robot_rc10.rc10 import RC10
# from lerobot_teleop_haptic.config_haptic import HapticConfig
# from lerobot_teleop_haptic.teleop_haptic import Haptic

from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop

from lerobot.processor import RobotProcessorPipeline, ObservationProcessorStep, ActionProcessorStep, RobotActionProcessorStep, IdentityProcessorStep
from lerobot.processor.converters import observation_to_transition, transition_to_observation, robot_action_to_transition, robot_action_observation_to_transition, transition_to_robot_action

import time

NUM_EPISODES = 3
FPS = 30
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "TEST"  # provide a task description

# HF_USER = ...  # provide your Hugging Face username

teleop_ip = "127.0.0.1"
teleop_port = 8081

# Create the robot and teleoperator configurations
camera_config = {"front": OpenCVCameraConfig(
    index_or_path=0, width=640, height=480, fps=FPS)
}
robot_config = RC10Config(
    cameras=camera_config
)
teleop_config = HapticConfig(
    ip=teleop_ip,
    port=teleop_port,
    scale=100
)

# Initialize the robot and teleoperator
robot = RC10(robot_config)
teleop = Haptic(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
dataset_features = {**action_features, **obs_features}

# print(dataset_features)

# Create the dataset where to store the data
dataset = LeRobotDataset.create(
    # repo_id=f"{HF_USER}/robot-learning-tutorial-data",
    repo_id="local/test_data",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

teleop_action_processor = RobotProcessorPipeline(
    steps=[IdentityProcessorStep()],
    name="teleop_action_identity",
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

robot_action_processor = RobotProcessorPipeline(
    steps=[IdentityProcessorStep()],
    name="robot_action_identity",
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

robot_observation_processor = RobotProcessorPipeline(
    steps=[IdentityProcessorStep()],
    name="robot_observation_identity",
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,

        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,

        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if (not events["stop_recording"]) and \
        (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,

            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,

            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.finalize()
# dataset.push_to_hub()