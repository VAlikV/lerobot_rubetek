# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline, IdentityProcessorStep, RewardProcessorStep
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
    transition_to_batch
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from lerobot.robots.rc10.rc10 import RC10
from lerobot.robots.rc10.config_rc10 import RC10Config
from lerobot.teleoperators.haptic.config_haptic import HapticConfig
from lerobot.teleoperators.haptic.teleop_haptic import Haptic

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "TASK"
HF_REPO_ID = "local/test_demo"


def main():
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

    # Build pipeline to convert follower joints to EE observation
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build pipeline to convert leader joints to EE action
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            IdentityProcessorStep()
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert EE action to follower joints
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            IdentityProcessorStep()
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    
    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            # Run the feature contract of the pipelines
            # This tells you how the features would look like after the pipeline steps
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(action=teleop.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Connect the robot and teleoperator
    teleop.connect()
    robot.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="recording_phone")

    if not teleop.is_connected or not robot.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode
        dataset.save_episode()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    teleop.disconnect()
    robot.disconnect()
    listener.stop()

    dataset.finalize()
    # dataset.push_to_hub()


if __name__ == "__main__":
    main()
