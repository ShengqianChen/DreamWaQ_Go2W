
import sys
sys.path.append("/home/xxx/go2w_rl_gym")


from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

def trans_r2s(qj):

        tmp = qj.copy()


        tmp[3] = qj[12]
        tmp[7] = qj[13]
        tmp[11] = qj[14]
        tmp[15] = qj[15]
        tmp[4] = qj[3]
        tmp[5] = qj[4]
        tmp[6] = qj[5]
        tmp[8] = qj[6]
        tmp[9] = qj[7]
        tmp[10] = qj[8]
        tmp[12] = qj[9]
        tmp[13] = qj[10]
        tmp[14] = qj[11]

        return tmp

def trans_s2r(qj):

        tmp = qj.copy()


        tmp[12] = qj[3]
        tmp[13] = qj[7]
        tmp[14] = qj[11]
        tmp[15] = qj[15]
        tmp[3] = qj[4]
        tmp[4] = qj[5]
        tmp[5] = qj[6]
        tmp[6] = qj[8]
        tmp[7] = qj[9]
        tmp[8] = qj[10]
        tmp[9] = qj[12]
        tmp[10] = qj[13]
        tmp[11] = qj[14]

        return tmp

class Controller:
    def __init__(self, config: Config) -> None:


        self.config = config
        self.remote_controller = RemoteController()


        self.policy = torch.jit.load(config.policy_path)


        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_real_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.counter = 0
        self.lin_vel = np.array([0.0, 0.0, 0.0])


        if config.msg_type == "hg":

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)


        self.wait_for_low_state()


        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd)


    def LowStateHgHandler(self, msg: LowStateHG):
        """Handle low-level state messages from H1 Gen2."""
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        """Handle low-level state messages from H1 Gen1."""
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        """Send a command with CRC attached."""
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        """Block until the first low-level state is received."""
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")


    def zero_torque_state(self):
        """Enter a safe zero-torque state until Start is pressed."""
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def wheel_move(self):


        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        for i in range(len(self.config.joint2motor_idx)):
            if i >= 12:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(0.1 + self.qj[i])
                self.low_cmd.motor_cmd[motor_idx].dq = float(1.0 + self.dqj[i])
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i] * 1.0)
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i] * 1.0)
                self.low_cmd.motor_cmd[motor_idx].tau = float(0)
                print(i)
                print(self.low_cmd.motor_cmd[i])
            else:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = float(self.config.default_real_angles[i])
                self.low_cmd.motor_cmd[motor_idx].dq = float(0)
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[i] * 1.0)
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[i] * 1.0)
                self.low_cmd.motor_cmd[motor_idx].tau = float(0)
                print(i)
                print(self.low_cmd.motor_cmd[i])
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        """Move to the default pose using linear interpolation."""
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.joint2motor_idx
        kps = self.config.kps
        kds = self.config.kds
        default_pos = self.config.default_real_angles
        dof_size = len(dof_idx)


        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        print(default_pos)
        print(init_dof_pos)


        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        """Hold the default pose until button A is pressed."""
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:

            for i in range(12):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        """Main control loop executed every control cycle."""
        self.counter += 1


        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq

        self.qj = trans_r2s(self.qj)
        self.dqj = trans_r2s(self.dqj)
        self.action = trans_r2s(self.action)


        # self.lin_vel = self.lin_vel + self.config.control_dt * np.array([self.low_state.imu_state.accelerometer], dtype=np.float32)
        # lin_vel_obs = self.lin_vel * self.config.lin_vel_scale


        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        ang_vel_obs = ang_vel * self.config.ang_vel_scale


        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)


        self.cmd[0] = 0.5
        self.cmd[1] = 0.0
        self.cmd[2] = 0.0


        err_obs = self.qj - self.config.default_sim_angles
        err_obs[self.config.wheel_sim_indices] = 0
        err_obs = err_obs * self.config.dof_err_scale


        dqj_obs = self.dqj * self.config.dof_vel_scale


        qj_obs = self.qj.copy()
        qj_obs[self.config.wheel_sim_indices] = 0


        num_actions = self.config.num_actions


        self.obs[:3] = ang_vel_obs
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale
        self.obs[9:9+num_actions] = err_obs
        self.obs[9+num_actions:9+num_actions*2] = dqj_obs
        self.obs[9+num_actions*2:9+num_actions*3] = qj_obs
        self.obs[9+num_actions*3:9+num_actions*4] = self.action


        file_path = "/home/mmj/sim2real_g2w/unitree_rl_gym/deploy/obs_real.log"
        with open(file_path, "a") as file:

            file.write(",".join(map(str, self.obs)) + "\n")


        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        file_path = "/home/mmj/sim2real_g2w/unitree_rl_gym/deploy/action_real.log"
        with open(file_path, "a") as file:

            file.write(",".join(map(str, self.action)) + "\n")

        self.qj = trans_s2r(self.qj)
        self.action = trans_s2r(self.action)
        self.dqj = trans_s2r(self.dqj)

        for i in range(len(self.config.joint2motor_idx)):
            if i >= 12:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.action[i] * self.config.action_scale + self.qj[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 1.0 + self.dqj[i]
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
                print(i)
                print(self.low_cmd.motor_cmd[i])
            else:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i] + self.action[i] * self.config.action_scale
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
                print(i)
                print(self.qj[i])
                print(self.low_cmd.motor_cmd[i].q)

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name", default="g1.yaml")
    args = parser.parse_args()


    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)


    ChannelFactoryInitialize(0, args.net)


    controller = Controller(config)


    controller.zero_torque_state()


    controller.move_to_default_pos()
    controller.move_to_default_pos()
    controller.move_to_default_pos()


    controller.default_pos_state()

    # while(1):

    print('RL Begin---------')

    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break


    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
