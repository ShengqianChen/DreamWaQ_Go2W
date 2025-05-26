import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

#            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.joint2motor_idx = config["joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]

            self.default_sim_angles = np.array(config["default_sim_angles"], dtype=np.float32)
            self.default_real_angles = np.array(config["default_real_angles"], dtype=np.float32)

            # self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            # self.arm_waist_kps = config["arm_waist_kps"]
            # self.arm_waist_kds = config["arm_waist_kds"]
            # self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.lin_vel_scale = config["lin_vel_scale"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.dof_err_scale = config["dof_err_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            
            self.action_scale = config["action_scale"]

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.wheel_real_indices = config['wheel_real_indices']
            self.wheel_sim_indices = config['wheel_sim_indices']
            self.wheel_speed = config['wheel_speed']



'''
control_dt：控制周期，单位为秒，表示控制系统执行一次控制更新的时间间隔。
msg_type：消息类型，用于指定通信中使用的消息格式或协议类型。
imu_type：IMU类型，表示使用的惯性测量单元的型号或种类。
weak_motor：弱电机列表，包含一些电机的索引或标识符，可能用于标识性能较弱或需要特殊处理的电机。
lowcmd_topic：低级命令主题，用于指定在机器人控制系统中发布低级命令的通信主题名称。
lowstate_topic：低级状态主题，用于指定接收低级状态信息的通信主题名称。
policy_path：策略路径，表示存储机器人控制策略（如训练好的神经网络模型）的文件路径，其中{LEGGED_GYM_ROOT_DIR}会被实际的路径替换。
leg_joint2motor_idx：腿关节到电机索引的映射，用于将腿关节的位置或指令映射到对应的电机索引。
kps：位置控制的比例增益，用于PID控制器中的比例部分，影响关节位置控制的响应速度和精度。
kds：位置控制的微分增益，用于PID控制器中的微分部分，有助于减少系统的振荡和提高稳定性。
default_angles：默认关节角度，表示机器人在初始状态或某些特定状态下各关节的目标角度。
arm_waist_joint2motor_idx：手臂腰部关节到电机索引的映射，用于将手臂腰部关节的位置或指令映射到对应的电机索引。
arm_waist_kps：手臂腰部位置控制的比例增益，用于控制手臂腰部关节的位置响应。
arm_waist_kds：手臂腰部位置控制的微分增益，用于提高手臂腰部关节控制的稳定性。
arm_waist_target：手臂腰部目标角度，表示手臂腰部关节在某些控制模式下的目标位置。
ang_vel_scale：角速度缩放因子，用于对角速度数据进行缩放，可能用于归一化或调整数据范围。
dof_pos_scale：自由度位置缩放因子，用于对关节位置数据进行缩放。
dof_vel_scale：自由度速度缩放因子，用于对关节速度数据进行缩放。
action_scale：动作缩放因子，用于对控制动作进行缩放，可能用于调整控制输入的范围或幅度。
cmd_scale：命令缩放因子数组，用于对不同的控制命令进行分别缩放。
max_cmd：最大命令值数组，表示各控制命令允许的最大值。
num_actions：动作数量，表示控制系统中需要输出的控制动作的维度或数量。
num_obs：观测数量，表示控制系统中用于状态观测的输入数据的维度或数量。
'''