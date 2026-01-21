from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("haptic")
@dataclass
class HapticConfig(TeleoperatorConfig):
    ip: str = "127.0.0.1"
    port: int = 8081
    recv_buf: int = 1024
    scale: float = 1.0
