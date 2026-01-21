# lerobot/robots/myrobot/robot.py
from __future__ import annotations
import numpy as np
from typing import Any, Dict

from lerobot.robots.robot import Robot
from lerobot.cameras import make_cameras_from_configs

from .config_rc10 import RC10Config
from .robot_adapter import RobotAdapter

class RC10(Robot):
    """
    Обёртка над robot_adapter под интерфейс LeRobot.
    Ожидается robot_adapter с методами:
      - reset()
      - observe() -> объект с полями q, dq, tcp_pos, tcp_vel, images[image_key]
      - apply_action(a: np.ndarray, a_gripper: float)
      - emergency_stop()
    """

    config_class = RC10Config
    name = "rc10"

    def __init__(self, config: RC10Config):
        super().__init__(config)
        self.config: RC10Config = config
        self._adapter: RobotAdapter = None
        self._connected = False

        self.cameras = make_cameras_from_configs(config.cameras)
        cam_features = {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras.keys()}

        self._obs_features = {
            # "state.joints_pos": (6,),
            # "state.joints_vel": (6,),
            "tcp.x":   float,
            "tcp.y":   float,
            "tcp.z":   float,
            "tcp.roll":   float,
            "tcp.pitch":   float,
            "tcp.yaw":   float,

            "tcp.dx":   float,
            "tcp.dy":   float,
            "tcp.dz":   float,
            "tcp.droll":   float,
            "tcp.dpitch":   float,
            "tcp.dyaw":   float,

            **cam_features
        }

        # Действия
        self._act_features = {
            "tcp.delta_x": float,
            "tcp.delta_y": float,
            "tcp.delta_z": float,

            "gripper.state": float,
        }

    # ---- Интерфейс LeRobot ----
    @property
    def observation_features(self) -> Dict[str, tuple]:
        return self._obs_features

    @property
    def action_features(self) -> Dict[str, tuple]:
        return self._act_features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def connect(self, calibrate: bool = True) -> None:
        # Создаём/поднимаем адаптер
        self._adapter = self.config.adapter_factory()

        for cam in self.cameras.values():
            cam.connect()

        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        
        try:
            for cam in self.cameras.values():
                cam.disconnect()

            self._adapter.emergency_stop()

        except Exception:
            pass

        self._adapter = None
        self._connected = False

    def reset(self) -> None:
        self._ensure()
        self._adapter.reset()

    def get_observation(self) -> Dict[str, Any]:
        """
        Приводим наблюдение к плоскому dict с ключами, заявленными в observation_features.
        """
        self._ensure()
        o = self._adapter.observe()

        obs = {
            # "state.joints_pos": np.asarray(o.q, dtype=np.float32),       # shape (6,)
            # "state.joints_vel": np.asarray(o.dq, dtype=np.float32),      # shape (6,)
            "tcp.x":   o.tcp_pos[0],
            "tcp.y":   o.tcp_pos[1],
            "tcp.z":   o.tcp_pos[2],
            "tcp.roll":   o.tcp_pos[3],
            "tcp.pitch":   o.tcp_pos[4],
            "tcp.yaw":   o.tcp_pos[5],

            "tcp.dx":   o.tcp_vel[0],
            "tcp.dy":   o.tcp_vel[1],
            "tcp.dz":   o.tcp_vel[2],
            "tcp.droll":   o.tcp_vel[3],
            "tcp.dpitch":   o.tcp_vel[4],
            "tcp.dyaw":   o.tcp_vel[5],
        }
        
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Политика выдаёт нормализованную 3-мерную команду в диапазоне ~[-1, 1].
        Как и в твоей env, мы делим на action_scale перед вызовом apply_action().
        """
        self._ensure()
        delta = np.asarray([action["tcp.delta_x"], action["tcp.delta_y"], action["tcp.delta_z"]], dtype=np.float32)
        a = delta / float(self.config.action_scale)
        a_gripper = action["gripper.state"] 

        self._adapter.apply_action(a, a_gripper)
        return {"tcp.delta": delta, "gripper.state": a_gripper}

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass

    # ---- Вспомогательное ----
    def _ensure(self):
        if not self._connected or self._adapter is None:
            raise RuntimeError("MyRobot is not connected")
