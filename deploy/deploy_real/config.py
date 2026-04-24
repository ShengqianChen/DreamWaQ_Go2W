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
Config fields loaded from YAML:
- control_dt: control loop period in seconds
- msg_type / imu_type: communication and IMU modes
- joint2motor_idx, kps, kds: actuator mapping and gains
- default_sim_angles / default_real_angles: nominal joint targets
- lin_vel_scale, ang_vel_scale, cmd_scale, dof_err_scale, dof_vel_scale: observation scaling
- action_scale, num_actions, num_obs: policy I/O dimensions
'''