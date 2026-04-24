from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2WRoughCfg( LeggedRobotCfg ):

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 16
        num_observations = 73
        num_obs_hist = 5
        num_privileged_obs = 320

    class commands( LeggedRobotCfg ):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 10.
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1, 1.0]
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.0
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
        terrain_proportions = [0.1, 0.1, 0.35, 0.2, 0.25]
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0 ,
            'RR_hip_joint': 0.0,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,

            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,

        }
        init_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0 ,
            'RR_hip_joint': 0.0,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,

            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'

        stiffness = {'hip_joint': 40.,'thigh_joint': 40.,'calf_joint': 40.,"foot_joint":0}
        damping = {'hip_joint': 1,'thigh_joint': 1,'calf_joint': 1,"foot_joint":0.5}
        # action scale: target angle = actionScale * action + defaultAngle

        action_scale = 0.25
        vel_scale = 10.0
        # decimation: Number of control action updates @ sim DT per policy DT

        decimation = 4
        urdf_limit_scale = 1.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = True
        payload_mass_range = [-1.0, 2.0]
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]
        randomize_friction = True
        friction_range = [0.25, 1.25]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_kp = True
        kp_range = [0.9, 1.1]
        randomize_kd = True
        kd_range = [0.9, 1.1]
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        delay = True

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w_30_unitree/urdf/go2w.urdf'
        name = "go2w"
        foot_name = "foot"
        wheel_name =["foot"]
        wheel_armature_add = 0.000
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter "base","calf","hip","thigh"
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.4
        max_contact_force = 100. # forces above this value are penalized

        class scales( LeggedRobotCfg.rewards.scales ):

            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -10.0
            torques = -5.0e-4
            dof_acc = -1.e-7
            dof_vel = -1.e-7
            wheel_acc = -1.e-7
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.01
            stand_still = -0.5
            hip_pos = -0.5
            dof_error = -0.05

class GO2WRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
        learning_rate = 1.e-3
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2w'
        num_steps_per_env = 48
        max_iterations = 20000
        load_run = "/root/gpufree-data/logs/rough_go2w/Apr23_22-44-48_"
        checkpoint = -1
        resume = False
        resume_path = "/root/gpufree-data/logs/rough_go2w/Apr22_21-26-30_/model_1000.pt"

