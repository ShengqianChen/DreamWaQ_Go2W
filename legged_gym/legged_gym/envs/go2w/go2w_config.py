from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2WRoughCfg( LeggedRobotCfg ):
    # 训练环境类
    class env(LeggedRobotCfg.env):
        num_envs = 4096 # 强化学习同时训练智能体的数量
        num_actions = 16 # 可操控的动作数量
        num_observations = 73 # 强化学习观测值的数量  
        num_obs_hist = 5
        num_privileged_obs = 320
      
    # 机器人指令类
    class commands( LeggedRobotCfg ):
        curriculum = True # 是否使用课程学习
        max_curriculum = 1.5 # 课程难度最高级
        num_commands = 4 # 指令的个数：x轴方向线速度，y轴方向线速度，角速度以及航向
        resampling_time = 10. # 指令更改的时间
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s] x轴方向线速度
            lin_vel_y = [-1.0, 1.0]   # min max [m/s] y轴方向线速度
            ang_vel_yaw = [-1, 1]    # min max [r ad/s] 角速度
            heading = [-3.14, 3.14] # 航向 实际上没有使用这个维度

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True  
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.5
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # 机器人初始状态
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m] 初始位置 四元数表示
        # 初始关节位置
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.67,     # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'RR_thigh_joint': 0.67,   # [rad]

            'FL_calf_joint': -1.3,   # [rad]
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,    # [rad]
            
            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,

        }
        init_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad] 机身
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.67,     # [rad] 大腿
            'RL_thigh_joint': 0.67,   # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'RR_thigh_joint': 0.67,   # [rad]

            'FL_calf_joint': -1.3,   # [rad] 小腿
            'RL_calf_joint': -1.3,    # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,    # [rad] 

            'FL_foot_joint':0.0, # 轮足
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

    # 机器人关节电机控制模式、参数
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P' # 位置控制、速度控制、扭矩控制
        
        stiffness = {'hip_joint': 50.,'thigh_joint': 50.,'calf_joint': 50.,"foot_joint":0}  # [N*m/rad] 刚度系数k_p 
        damping = {'hip_joint': 1,'thigh_joint': 1,'calf_joint': 1,"foot_joint":0.5}     # [N*m*s/rad] 阻尼系数k_d
        # action scale: target angle = actionScale * action + defaultAngle
        # 乘一个缩放因子，目的是让动作值适应不同关节的运动范围
        action_scale = 0.25
        vel_scale = 10.0 # 轮子的速度缩放超参数
        # decimation: Number of control action updates @ sim DT per policy DT
        # 仿真环境的控制频率/decimation = 实际环境中的控制频率
        decimation = 4
        wheel_speed = 1

    # 与机器人urdf相关参数
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf' # 存放位置
        name = "go2w"
        foot_name = "foot"
        wheel_name =["foot"] 
        penalize_contacts_on = ["thigh", "calf", "base"] # 惩罚接触
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter "base","calf","hip","thigh"
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
    
    # 奖励函数
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.4 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.34
        max_contact_force = 100. # forces above this value are penalized
       
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.8
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -0.1
            ang_vel_xy = -0.05
            orientation = -2
            torques = -0.0002
            dof_vel = -1e-7
            dof_acc = -1e-7
            base_height = -0.5
            feet_air_time =  0
            collision = -0.1
            feet_stumble = -0.1
            action_rate = -0.0002
            stand_still = -0.01
            dof_pos_limits = -0.9
            arm_pos = -0.  
            hip_action_l2 = -0.1

class GO2WRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'blind_50_1_40_0.5_trmeish'
        experiment_name = 'rough_go2w'
        num_steps_per_env = 24 # per iteration
        max_iterations = 30000
        load_run = "/home/hu/csq/DreamWaQ/legged_gym/logs/rough_go2w/May20_22-55-02_blind_50_1_40_0.5_trmeish"
        checkpoint = -1
  
