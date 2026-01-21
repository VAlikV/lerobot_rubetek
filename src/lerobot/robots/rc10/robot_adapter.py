from dataclasses import dataclass
import numpy as np, time

@dataclass
class Obs:
    images: dict             # {"cam_front": np.ndarray(H,W,3), "cam_side": ...}
    q: np.ndarray
    dq: np.ndarray
    tcp_pos: np.ndarray
    tcp_vel: np.ndarray
    # torque: np.ndarray
    # gripper_pos: float
    timestamp: float

class RobotAdapter:
    def __init__(self, robot):
        self.ctrl = robot

        self.ctrl.start()

        q, dq, tcp, tcp_vel, torque = self.ctrl.get_current_info()

        self.pos = tcp[0:3].copy()
        self.orient = np.array([[1.0, 0.0, 0.0], 
                                [0.0, -1.0, 0.0], 
                                [0.0, 0.0, -1.0]])
        
        self.ctrl.set_target(self.pos.copy(), self.orient.copy())

        self.start_pos = tcp[0:3].copy()
        self.start_orient = self.orient.copy()

    # ====================================================================================================

    def observe(self) -> Obs:
        q, dq, tcp, tcp_vel, torque = self.ctrl.get_current_info()

        return Obs(q, dq, tcp, tcp_vel, torque, time.time())

    # ====================================================================================================

    def apply_action(self, delta, a_gripper):

        # if delta[0:3].any() >= 0.0001:
        self.pos[0:3] += delta[0:3]
        self.ctrl.set_target(self.pos.copy(), self.orient.copy())
        # self.ctrl.step()
        # self.ctrl.set_gripper_cmd(a_gripper)

    # ====================================================================================================

    def emergency_stop(self, reason=""):  # по желанию
        self.ctrl.stop()

    # ====================================================================================================

    def reset(self):
        self.ctrl.set_target(self.start_pos, self.start_orient)

        self.pos = self.start_pos.copy()

        while not self._check_reset():
            time.sleep(0.001)
    
    # ====================================================================================================

    def _check_reset(self):

        tcp = self.ctrl.get_current_tcp()

        if (np.abs(self.start_pos - tcp[0:3]) >= 0.002).any():
            print(np.abs(self.start_pos - tcp[0:3]))
            return False
        else:
            return True
