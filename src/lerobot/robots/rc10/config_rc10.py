from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Optional
from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

from .robot_adapter import RobotAdapter
# from API.controller import TaskSpaceJogController
from API.API.controller import TaskSpaceJogController

@RobotConfig.register_subclass("rc10")
@dataclass
class RC10Config(RobotConfig):
    
    # adapter_factory: Callable[[], object]

    action_scale: float = 100.0

    cameras: dict[str, CameraConfig] = field(
        default_factory={
            "cam_1": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=480,
                height=640,
            ),
        }
    )

    def adapter_factory(self):
        robot = TaskSpaceJogController(ip="10.10.10.10",
                                        rate_hz=100,
                                        velocity=1,
                                        acceleration=1,
                                        treshold_position=0.001,
                                        treshold_angel=1)
        adapter = RobotAdapter(robot=robot)

        return adapter