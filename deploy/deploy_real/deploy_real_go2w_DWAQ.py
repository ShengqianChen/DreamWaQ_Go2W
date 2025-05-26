from typing import Union
import numpy as np
import time
import os
import torch  


# region ########### Unitree通信相关导入 ###########
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config

def trans_r2s(qj):
        # 创建一个临时张量，初始化为 qj 的值
        tmp = qj.copy()
        
        # 按照规则进行赋值
        tmp[0] = qj[3]
        tmp[1] = qj[4]
        tmp[2] = qj[5]
        tmp[3] = qj[13]

        tmp[4] = qj[0]
        tmp[5] = qj[1]
        tmp[6] = qj[2]
        tmp[7] = qj[12]

        tmp[8] = qj[9]
        tmp[9] = qj[10]
        tmp[10] = qj[11]
        tmp[11] = qj[15]

        tmp[12] = qj[6]
        tmp[13] = qj[7]
        tmp[14] = qj[8]
        tmp[15] = qj[14]

        return tmp

def trans_s2r(qj):
        # 创建一个临时张量，初始化为 qj 的值
        tmp = qj.copy()
        
        # 按照规则进行赋值
        tmp[3] = qj[0]
        tmp[4] = qj[1]
        tmp[5] = qj[2]
        tmp[13] = qj[3]

        tmp[0] = qj[4]
        tmp[1] = qj[5]
        tmp[2] = qj[6]
        tmp[12] = qj[7]

        tmp[9] = qj[8]
        tmp[10] = qj[9]
        tmp[11] = qj[10]
        tmp[15] = qj[11]

        tmp[6] = qj[12]
        tmp[7] = qj[13]
        tmp[8] = qj[14]
        tmp[14] = qj[15]

        return tmp

class Controller:
    def __init__(self, config: Config) -> None:
        # region ########### 初始化阶段 ###########
        #配置加载
        self.config = config
        self.remote_controller = RemoteController()
        
        # 加载模型文件
        self.actor = torch.jit.load('/home/drl/deploy/pre_train/g2wDWAQ/actor_dwaq.pt')
        self.encoder = torch.jit.load('/home/drl/deploy/pre_train/g2wDWAQ/encoder_dwaq.pt')
        self.latent_mu = torch.jit.load('/home/drl/deploy/pre_train/g2wDWAQ/latent_mu_dwaq.pt')
        self.latent_var = torch.jit.load('/home/drl/deploy/pre_train/g2wDWAQ/latent_var_dwaq.pt')
        self.vel_mu = torch.jit.load('/home/drl/deploy/pre_train/g2wDWAQ/vel_mu_dwaq.pt')
        self.vel_var = torch.jit.load('/home/drl/deploy/pre_train/g2wDWAQ/vel_var_dwaq.pt')

        #过程变量初始化
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_real_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.obs_hist_buf = np.zeros(config.num_obs * 5, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.counter = 0
        self.lin_vel = np.array([0.0, 0.0, 0.0])

        #DDS通信初始化
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        # 等待底层状态连接
        self.wait_for_low_state()

        init_cmd_go(self.low_cmd)
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()  

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandUp()
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    # region ########### 状态回调处理 ###########
    def LowStateHgHandler(self, msg: LowStateHG):
        """处理H1 Gen2的低层状态消息"""
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        """处理H1 Gen1的低层状态消息""" 
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)
    # endregion

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        """发送指令（自动添加CRC校验）"""
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        """等待直到收到底层状态数据"""
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    # region ########### 状态机流程 ###########
    def zero_torque_state(self):
        """
        零力矩状态（安全准备阶段）
        1. 发送零力矩指令
        2. 等待Start按键触发
        """
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        total_time = 1
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.joint2motor_idx
        kps = 70.0
        kds = 5.0
        default_pos = self.config.default_real_angles
        dof_size = len(dof_idx)
        
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        for i in range(num_step):
            alpha = i / num_step
            for j in range(12):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 80
                self.low_cmd.motor_cmd[motor_idx].kd = 5
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            # 腿部关节控制
            for i in range(12):
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = 70.0
                self.low_cmd.motor_cmd[motor_idx].kd = 5.0
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    # endregion

    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code

    def run(self):
        """主控制循环（每个控制周期执行）"""
        self.counter += 1
          
        # 获得机器人电机数据
        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].q # 关机反馈位置信息：默认为弧度值
            self.dqj[i] = self.low_state.motor_state[self.config.joint2motor_idx[i]].dq # 关节反馈速度

        #  print(self.dqj)
        self.qj = trans_r2s(self.qj)
        self.dqj = trans_r2s(self.dqj)
        self.action = trans_r2s(self.action)

        # 机器人角速度
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        ang_vel_obs = ang_vel * self.config.ang_vel_scale

        # 重力向量
        quat = self.low_state.imu_state.quaternion  # 四元数格式: w, x, y, z
        gravity_orientation = get_gravity_orientation(quat) # 根据IMU返回的机身姿态四元数，将重力向量的方向调转过来
        
        # # 外部控制命令
        self.cmd[0] = self.remote_controller.ly  # 前后速度
        self.cmd[1] = self.remote_controller.lx * -1  # 横向速度
        self.cmd[2] = self.remote_controller.rx * -1  # 旋转速度
        # ly,lx分别指左摇杆的y轴和x轴坐标，同理rx指右摇杆的x轴坐标，范围都是[-1, 1] 
        # 乘command_scale后范围变为[-2 , 2] [-2, 2] [-0.25, 0.25] 
        # 这样设置是为了使得给机器狗发布的命令不超过训练时候的范围
        # 训练范围设置的比较小是因为719场地小害怕损坏机器狗

        # 关节误差
        err_obs = self.qj - self.config.default_sim_angles
        err_obs[self.config.wheel_sim_indices] = 0
        err_obs = err_obs * self.config.dof_err_scale
        
        # 关节速度
        dqj_obs = self.dqj * self.config.dof_vel_scale

        # 目前关节位置
        qj_obs = self.qj.copy()
        qj_obs[self.config.wheel_sim_indices] = 0

        # 观测向量构建
        num_actions = self.config.num_actions

        self.obs[:3] = ang_vel_obs # 角速度
        self.obs[3:6] = gravity_orientation # 重力向量
        self.obs[6:9] = self.cmd * self.config.cmd_scale # 外部控制指令
        self.obs[9:9+num_actions] = err_obs # 关节位置和默认位置误差
        self.obs[9+num_actions:9+num_actions*2] = dqj_obs # 关节速度
        self.obs[9+num_actions*2:9+num_actions*3] = qj_obs  # 关节位置
        self.obs[9+num_actions*3:9+num_actions*4] = self.action  # 历史动作
        
        # 将 self.obs 写入文件，测试用
        # file_path = "/home/DRL/sim2real_g2w/unitree_rl_gym/deploy/obs_real.log"  # 文件路径
        # with open(file_path, "a") as file:  # 以追加模式打开文件
        # # 将 self.obs 转换为字符串并写入文件
        #     file.write(",".join(map(str, self.obs)) + "\n")

        # 拼接观测数据
        self.obs_hist_buf = self.obs_hist_buf[73:]
        self.obs_hist_buf = np.concatenate((self.obs_hist_buf, self.obs), axis=-1)

        tmp = torch.from_numpy(self.obs_hist_buf)
        h = self.encoder(tmp)
        vel_mu = self.vel_mu(h)
        vel_var = self.vel_var(h)
        latent_mu = self.latent_mu(h)
        latent_var = self.latent_var(h)
        vel = self.reparameterise(vel_mu, vel_var)
        latent = self.reparameterise(latent_mu, latent_var)

        print(vel)
        print(latent)    
        code = torch.cat((vel, latent), dim = -1)
        print(code)
        tmpp = torch.from_numpy(self.obs)
        obs_all = torch.cat((code, tmpp), dim = -1)

        # 根据观测值传入Policy函数获得Action
        # obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.actor(obs_all).detach().numpy().squeeze()

        self.qj = trans_s2r(self.qj)
        self.action = trans_s2r(self.action)
        self.dqj = trans_s2r(self.dqj)

        for i in range(len(self.config.joint2motor_idx)):
            if i >= 12:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = 0.0
                self.low_cmd.motor_cmd[motor_idx].dq = self.action[i] * 10.0
                self.low_cmd.motor_cmd[motor_idx].kp = 0.0
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            else:
                motor_idx = self.config.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_real_angles[i] + self.action[i] * self.config.action_scale
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        self.send_cmd(self.low_cmd) # 发送指令
        time.sleep(self.config.control_dt) # 休眠控制间隔

      #  fuckssdf

if __name__ == "__main__":
    # region ########### 程序入口 ###########
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name", default="g1.yaml")
    args = parser.parse_args()

    # 加载配置文件
    config_path = f"/home/drl/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    

    # 初始化DDS通信
    ChannelFactoryInitialize(0, args.net)

    # 创建控制器实例
    controller = Controller(config)

    # region ########### 状态机执行流程 ###########
    # 阶段1: 零力矩安全状态
    controller.zero_torque_state()
    
    # 阶段2: 平滑归位运动
    controller.move_to_default_pos()
    
    # # 阶段3: 默认姿态保持
    controller.default_pos_state()
    
    print('RL Begin---------')
    # # 阶段4: 主控制循环
    while True:
        try:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    
    # # 阶段5: 退出时进入阻尼模式
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
    # endregion
    # endregion