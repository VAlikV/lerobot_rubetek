import socket
import numpy as np
from queue import Queue
from typing import Any, Dict
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from .config_haptic import HapticConfig


class Haptic(Teleoperator):
    config_class = HapticConfig
    name = "haptic"

    def __init__(self, config: HapticConfig):
        super().__init__(config)
        self.cfg = config
        self.sock = None
        self.last_msg = None
        self.event_queue = Queue()
        self._connected = False

        self._act_features = {
            "tcp.delta_x": float,
            "tcp.delta_y": float,
            "tcp.delta_z": float,
            "gripper.state": float,
        }

        self._feedback_features = {
            "feedback.force_x": float,
            "feedback.force_y": float,
            "feedback.force_z": float,
        }

    @property
    def feedback_features(self) -> Dict[str, tuple]:
        return self._feedback_features

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
        pass

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.cfg.ip, self.cfg.port))
        self.sock.settimeout(0.005)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024) # 1024 размер в байтах (по стандарту там вроде 212992 (?))
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self.sock.close()
        self.sock = None
        self._connected = False

    def _read_udp_delta(self):
        try:
            data, _ = self.sock.recvfrom(self.cfg.recv_buf)
        except socket.timeout:
            return False, None

        arr = np.array(list(map(float, data.decode()[1:-1].split(","))), dtype=np.float32)

        if self.last_msg is None:
            self.last_msg = arr
            return False, None

        delta = arr.copy()
        delta[0:3] = delta[0:3] - self.last_msg[0:3]
        self.last_msg = arr

        return True, delta

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            return {"tcp.delta_x": 0.0,
                    "tcp.delta_y": 0.0,
                    "tcp.delta_z": 0.0,
                    "gripper.state": 0.0}

        has_data, delta = self._read_udp_delta()
        if not has_data or delta is None:
            return {"tcp.delta_x": 0.0,
                    "tcp.delta_y": 0.0,
                    "tcp.delta_z": 0.0,
                    "gripper.state": 0.0}

        # оставляем только XY, как в RealRobotEnv
        a = delta[:3] * self.cfg.scale
        g_a = delta[3]
        return {"tcp.delta_x": float(a[0]),
                "tcp.delta_y": float(a[1]),
                "tcp.delta_z": float(a[2]),
                "gripper.state": float(g_a)}

    def get_teleop_events(self) -> Dict[str, Any]:
        has_data, _ = self._read_udp_delta()

        return {
            TeleopEvents.IS_INTERVENTION: bool(has_data),
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        
        return {"feedback.force_x": 0.0,
                "feedback.force_y": 0.0,
                "feedback.force_z": 0.0}

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass
