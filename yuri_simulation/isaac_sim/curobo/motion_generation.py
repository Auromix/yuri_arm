#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################################################
#
# Copyright 2023 Herman Ye @Auromix
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#####################################################################################
#
# Description: Nvidia Isaac Sim and CuRobo integration demo, arm will follow the target cube and avoid obstacles.
# Version: 0.1.0
# Date: 2023-12-23
# Author: Herman Ye @Auromix
#
#####################################################################################
#
# Revision History:
#
# Date       Version  Author       Description
# ---------- -------- ------------ -----------------------------------------------
# 2023-12-13 0.1.0    Herman Ye    Created.

import torch
import torchvision
import numpy as np


# Isaac Sim
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)


class MyCuroboRobot():
    def __init__(self, curobo_robot_config, curobo_world_config, visualize_spheres):
        # Sim and real flag
        self.use_sim = True
        self.use_real = False
        # Set curobo robot configuration variable
        self.curobo_robot_config_file_name = curobo_robot_config
        # Set curobo world configuration variable
        self.curobo_world_config_file_name = curobo_world_config
        # Set visualize spheres variable
        self.visualize_collision_spheres_enable = visualize_spheres
        # Init the visualize collision spheres list of curobo robot
        self.isaac_spheres_prim_list = None
        # Init curobo execution plan
        self.execution_plan = None
        self.execution_index = 0
        # Setup the scene of isaac sim
        self.isaac_setup_scene()

        return

    def isaac_setup_scene(self):
        # Create Isaac Sim environment
        from omni.isaac.core import World
        self.isaac_world = World(stage_units_in_meters=1.0)

        # Add a default ground plane to the scene
        self.isaac_world.scene.add_default_ground_plane()

        # Set isaac world as default prim
        isaac_world_prim = self.isaac_world.stage.GetPrimAtPath("/World")
        self.isaac_world.stage.SetDefaultPrim(isaac_world_prim)

        # Add prim "/curobo"(type = Xform) to the stage
        self.isaac_world.stage.DefinePrim("/curobo", "Xform")

        # Setup curobo
        self.curobo_setup()

        # Create a visual target cube to follow
        # The position and orientation of the cube will be used as the curobo goal pose
        # Only visual and not physical
        from omni.isaac.core.objects import cuboid
        self.target_cube = cuboid.VisualCuboid(
            # prim path in stage
            prim_path="/World/target",
            # name
            name="my_target",
            # x, y, z
            position=np.array([0.4, 0.3, 0.25]),
            # qw, qx, qy, qz
            orientation=np.array([0, 1, 0, 0]),
            # r, g, b
            color=np.array([1.0, 0, 0]),
            # mesh size scale of original cube(1, 1, 1)
            size=0.05,
        )

        # Create a obstacle cube to test collision avoidance (isaac sim method)
        from omni.isaac.core.objects import DynamicCuboid
        self.my_obstacle_cube = self.isaac_world.scene.add(
            DynamicCuboid(
                prim_path="/World/my_obstacle_cube",
                name="my_obstacle_cube",
                position=np.array([0.4, 0, 0.25]),
                scale=np.array([0.4, 0.05, 0.4]),
                color=np.array([0, 0, 1.0]),
            ))

        # Focus on the cubes and robot
        from omni.kit.viewport.utility import get_active_viewport, frame_viewport_selection
        import omni.usd
        viewport = get_active_viewport()
        prims_to_visual_focus = ["/World/target",
                                 "/World/my_obstacle_cube", self.robot_prim_path]
        prim_to_select = ["/World/target"]
        ctx = omni.usd.get_context()
        ctx.get_selection().set_selected_prim_paths(prims_to_visual_focus, True)
        frame_viewport_selection(viewport)
        ctx.get_selection().set_selected_prim_paths(prim_to_select, True)

    def isaac_close(self):
        # Close Isaac Sim
        simulation_app.close()

    def isaac_run(self):
        # For demo start waiting
        wait_for_play_time_count = 0
        # For user operation detection in isaac sim
        previous_cube_position, previous_cube_orientation = self.target_cube.get_world_pose()
        # For user is not moving target cube detection
        previous_target_position_diff = 0.0
        previous_target_orientation_diff = 0.0
        user_is_not_moving_target_cube_count = 0

        # Main loop
        while simulation_app.is_running():
            # Get the current time step index since PLAY was clicked
            self.current_step_index = self.isaac_world.current_time_step_index

            # Step the simulation
            self.isaac_world.step(render=True)

            # Demo start check(index=0)
            if not self.isaac_world.is_playing():
                wait_for_play_time_count += 1
                if wait_for_play_time_count % 100 == 0:
                    print(
                        f"Please click 'PLAY' to start curobo demo... [current sim step index: {self.current_step_index}]")
                continue

            # Init(index=1)
            if self.current_step_index < 2:
                # Reset the world after all assets are loaded and the 'Play' button is pressed
                self.isaac_world.reset()

                # Set robot initial joint angles in isaac world
                self.joint_indices = [self.robot.get_dof_index(x)
                                      for x in self.joint_names]
                print(
                    f"<Joint Indices for Robot Joint Position Setting>:\n{self.joint_indices}\n")
                self.robot.set_joint_positions(
                    self.initial_joint_angles, self.joint_indices)
                print(
                    f"Setting robot initial joint angles in isaac world: {self.initial_joint_angles}")

                # Set max efforts for articulation view
                # Info@HermanYe: why???
                effort_values = np.array(
                    [5000 for i in range(len(self.joint_indices))])
                self.robot._articulation_view.set_max_efforts(
                    values=effort_values, joint_indices=self.joint_indices)

            # Skip the first 50 steps to avoid potential errors(2<=index<50)
            elif self.current_step_index >= 2 and self.current_step_index < 50:
                continue

            # Normal case(index>=50)
            elif self.current_step_index == 50:
                # Update curobo collision world once at the beginning
                self.curobo_update()
                print(
                    f"Please move the target cube to the desired position and orientation")
            else:
                # Normal curobo collision world update per 1000 steps
                if self.current_step_index % 1000 == 0.0:
                    self.curobo_update()

                # Get goal pose(target cube pose) in this sim step
                cube_position, cube_orientation = self.target_cube.get_world_pose()

                # Get sim robot current joint state
                self.isaac_sim_joint_state = self.robot.get_joints_state()

                # Get sim robot joint names
                self.isaac_sim_joint_names = self.robot.dof_names

                # Convert isaac sim joint state to curobo joint state
                self.curobo_joint_state = self.sim_joint_state_to_curobo_joint_state(self.isaac_sim_joint_state,
                                                                                     self.isaac_sim_joint_names,
                                                                                     self.curobo_motion_gen.kinematics.joint_names)

                # Normal collision spheres visualization update per 10 steps
                if self.visualize_collision_spheres_enable and self.current_step_index % 10 == 0:
                    self.visualize_sphere_of_curobo_robot(
                        self.curobo_motion_gen, self.curobo_joint_state, self.isaac_spheres_prim_list)

                # Normal motion planning
                # Get target pose(position or orientation) difference
                target_position_diff = np.linalg.norm(
                    cube_position - previous_cube_position)
                target_orientation_diff = np.linalg.norm(
                    cube_orientation - previous_cube_orientation)

                # Make sure the user is not moving the target cube(at least 5 steps here)
                if previous_target_position_diff == target_position_diff and previous_target_orientation_diff == target_orientation_diff:
                    user_is_not_moving_target_cube_count += 1
                else:
                    user_is_not_moving_target_cube_count = 0
                    previous_target_position_diff = target_position_diff
                    previous_target_orientation_diff = target_orientation_diff

                # Plan
                if (
                    # Target pose(position or orientation) is changed
                    (target_position_diff > 0.001 or target_orientation_diff > 0.001)
                    # Arm is not executing last motion so that the arm can be controlled by user
                    and np.linalg.norm(self.isaac_sim_joint_state.velocities) < 0.2
                    # User is not moving target cube now(avoid frequent motion planning when user is moving the target cube)
                    and user_is_not_moving_target_cube_count > 5
                ):

                    print(
                        f"[USER OPERATION] Target pose update:\nposition: {cube_position}\norientation: {cube_orientation}\n")

                    # Update previous cube pose
                    previous_cube_position = cube_position
                    previous_cube_orientation = cube_orientation

                    # Set goal pose
                    goal_position = cube_position
                    goal_orientation = cube_orientation
                    from curobo.types.math import Pose
                    inverse_kinematics_goal = Pose(
                        position=self.tensor_args.to_device(goal_position),
                        quaternion=self.tensor_args.to_device(
                            goal_orientation),
                    )
                    print(
                        f"<Inverse Kinematics Goal>:\n{inverse_kinematics_goal}\n")
                    # Start motion planning
                    self.curobo_plan(inverse_kinematics_goal)

                # Execute
                if self.execution_plan is not None:
                    # Execute motion plan
                    self.curobo_execute()

    def sim_joint_state_to_curobo_joint_state(self, isaac_sim_joint_state, isaac_sim_joint_names, curobo_motion_gen_joint_names):
        from curobo.types.state import JointState
        tensor_args = self.tensor_args
        curobo_joint_state = JointState(
            # Joint names
            joint_names=isaac_sim_joint_names,
            # Position control mode
            position=tensor_args.to_device(
                isaac_sim_joint_state.positions),
            velocity=tensor_args.to_device(
                isaac_sim_joint_state.velocities)*0.0,
            acceleration=tensor_args.to_device(
                isaac_sim_joint_state.velocities)*0.0,
            jerk=tensor_args.to_device(
                isaac_sim_joint_state.velocities)*0.0,
        )
        # Get joint state with a ordered joint names
        curobo_joint_state = curobo_joint_state.get_ordered_joint_state(
            curobo_motion_gen_joint_names)
        return curobo_joint_state

    def visualize_sphere_of_curobo_robot(self, motion_gen, curobo_joint_state, isaac_spheres_prim_list):
        # Get curobo robot collision spheres list
        curobo_sphere_list = motion_gen.kinematics.get_robot_as_spheres(
            curobo_joint_state.position)

        # Initialize the spheres in isaac sim
        if isaac_spheres_prim_list is None:
            from omni.isaac.core.objects import sphere
            isaac_spheres_prim_list = []
            # Create visual spheres
            for curobo_sphere_index, curobo_sphere in enumerate(curobo_sphere_list[0]):
                spheres_prim = sphere.VisualSphere(
                    # Prim path in stage
                    prim_path="/curobo/robot_sphere_" + \
                    str(curobo_sphere_index),
                    # Position (convert to one dimensional array)
                    position=np.ravel(curobo_sphere.position),
                    # Radius
                    radius=float(curobo_sphere.radius),
                    # Color
                    color=np.array([0, 0.8, 0.2]),
                )
                isaac_spheres_prim_list.append(spheres_prim)

        # Update the spheres in isaac sim
        else:
            for curobo_sphere_index, curobo_sphere in enumerate(curobo_sphere_list[0]):
                isaac_spheres_prim_list[curobo_sphere_index].set_world_pose(
                    position=np.ravel(curobo_sphere.position))
                isaac_spheres_prim_list[curobo_sphere_index].set_radius(
                    float(curobo_sphere.radius))

    def curobo_plan(self, inverse_kinematics_goal):
        # Convert curobo joint state into one dimensional
        joint_state = self.curobo_joint_state.unsqueeze(0)

        # Plan a single motion
        plan_result = self.curobo_motion_gen.plan_single(
            joint_state, inverse_kinematics_goal)

        # Get plan success
        plan_success = plan_result.success.item()

        if plan_success:
            print("Plan success, start execution")
            # Interpolate the trajectory
            interpolated_plan = plan_result.get_interpolated_plan()
            # Get all joint state plan
            curobo_full_plan = self.curobo_motion_gen.get_full_js(
                interpolated_plan)

            # # Find the joints exist in both
            # # Info@HermanYe: Might passive joints be included? Should skip this step?
            # dof_index_list = []
            # common_joint_state_names = []
            # for x in curobo_full_plan.joint_names:

            #     if x in self.isaac_sim_joint_names:
            #         dof_index_list.append(self.robot.get_dof_index(x))
            #         common_joint_state_names.append(x)

            common_joint_state_names = self.joint_names
            self.execution_plan = curobo_full_plan.get_ordered_joint_state(
                common_joint_state_names)
            self.execution_index = 0
        else:
            print(
                "Plan failed, could not find a solution within the given attempts limit")

    def curobo_execute(self):
        # Get current execution state
        execution_state = self.execution_plan[self.execution_index]

        # Create Isaac ArticulationAction
        from omni.isaac.core.utils.types import ArticulationAction
        action_to_execute = ArticulationAction(
            execution_state.position.cpu().numpy(),
            execution_state.velocity.cpu().numpy(),
            joint_indices=self.joint_indices,
        )

        # Apply action
        self.robot_articulation_controller.apply_action(action_to_execute)

        # Render per 'physics_render_rate' physics steps
        physics_render_rate = 2
        for i in range(physics_render_rate):
            self.isaac_world.step(render=False)

        # Update execution index
        self.execution_index += 1

        # Check if the execution is finished
        if self.execution_index >= len(self.execution_plan.position):
            self.execution_plan = 0
            self.execution_plan = None
            print("Execution finished")

    def curobo_setup(self,
                     # world_model,
                     interpolation_dt=0.02,
                     collision_activation_distance=0.02):
        def load_curobo_robot_config(curobo_robot_config_file_name="franka.yml"):
            # Load curobo robot configuration
            from curobo.util_file import load_yaml, join_path, get_robot_configs_path
            curobo_robot_config_full_path = join_path(
                get_robot_configs_path(), curobo_robot_config_file_name)
            print(
                f"Loading curobo robot configuration from {curobo_robot_config_full_path}")
            curobo_robot_config = load_yaml(
                curobo_robot_config_full_path)["robot_cfg"]
            print(f"<Curobo Robot Configuration>:\n{curobo_robot_config}\n")
            return curobo_robot_config

        def load_curobo_world_config(curobo_world_config_file_name="collision_table.yml"):
            # Load curobo world configuration
            from curobo.util_file import load_yaml, join_path, get_world_configs_path
            from curobo.geom.types import WorldConfig
            print("Loading curobo world configuration...")
            curobo_world_config_full_path = join_path(
                get_world_configs_path(), curobo_world_config_file_name)
            print(
                f"<Curobo World Configuration File path>:\n{ curobo_world_config_full_path}\n")
            curobo_world_config = WorldConfig.from_dict(
                load_yaml(curobo_world_config_full_path))
            print(f"<Curobo World Configuration>:\n{curobo_world_config}\n")
            return curobo_world_config

        def add_curobo_robot_to_isaac_scene(
            curobo_robot_config,
            isaac_world,
            robot_position_in_world=np.array([0, 0, 0]),
            robot_orientation_in_world=np.array([1, 0, 0, 0]),
            robot_name="my_robot"
        ):
            # Create a URDF interface
            from omni.importer.urdf import _urdf as isaac_urdf
            urdf_interface = isaac_urdf.acquire_urdf_interface()

            # Create an URDF import configuration
            import_config = isaac_urdf.ImportConfig()
            import_config.merge_fixed_joints = False
            import_config.convex_decomp = False
            import_config.import_inertia_tensor = True
            import_config.fix_base = True
            import_config.make_default_prim = False
            import_config.self_collision = False
            import_config.create_physics_scene = True
            import_config.import_inertia_tensor = False
            import_config.default_drive_strength = 20000
            import_config.default_position_drive_damping = 500
            import_config.default_drive_type = isaac_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
            import_config.distance_scale = 1
            import_config.density = 0.0

            # Get URDF path
            from curobo.util_file import get_assets_path, join_path, get_filename, get_path_of_dir
            urdf_full_path = join_path(
                get_assets_path(), curobo_robot_config["kinematics"]["urdf_path"])
            print(f"<URDF Path>:\n{urdf_full_path}\n")

            # Get robot description directory path
            robot_description_path = get_path_of_dir(urdf_full_path)
            print(f"<Robot Description Path>:\n{robot_description_path}\n")

            # Get URDF file name
            urdf_file_name = get_filename(urdf_full_path)
            print(f"<URDF File Name>:\n{urdf_file_name}\n")

            # Get robot from URDF
            print(
                f"Importing curobo robot from URDF...[START]")

            # Set robot stage path
            # Warning@HermanYe: It seems that the variable "stage_path" is useless
            stage_path = ""
            print(f"<Robot Name in Isaac scene>:\n{robot_name}\n")

            # Parse URDF
            imported_robot = urdf_interface.parse_urdf(
                robot_description_path, urdf_file_name, import_config)

            # Import robot
            robot_stage_path = urdf_interface.import_robot(
                # Asset Root
                robot_description_path,
                # Asset Name
                urdf_file_name,
                # Parsed URDF
                imported_robot,
                # Import Config
                import_config,
                # Stage Path
                stage_path,
            )
            # Use omni.isaac.core.robots to create a Isaac Robot instance
            from omni.isaac.core.robots import Robot
            print(
                f"<Robot Position(x,y,z) in World>:\n{robot_position_in_world}\n")
            print(
                f"<Robot Orientation(w,x,y,z) in World>:\n{robot_orientation_in_world}\n")
            print(f"<Robot Prim Stage Path>:\n{robot_stage_path}\n")

            # https://docs.omniverse.nvidia.com/kit/docs/omniverse-urdf-importer/latest/source/extensions/omni.importer.urdf/docs/index.html#omni.importer.urdf._urdf.Urdf.import_robot
            # Encapsulate the prim root as a Isaac Robot instance
            print("Encapsulating the robot prim root as a Isaac Robot instance...")
            robot_in_world = Robot(
                prim_path=robot_stage_path,
                name=robot_name,
                position=robot_position_in_world,
                orientation=robot_orientation_in_world,
            )

            # Fix Isaac Sim solver iteration counts
            print("Setting Isaac Sim solver iteration counts...")
            velocity_solver_iteration_count = 4
            position_solver_iteration_count = 44
            print(
                f"<Isaac Sim Velocity Solver Iteration Count>:\n{velocity_solver_iteration_count}\n")
            print(
                f"<Isaac Sim Position Solver Iteration Count>:\n{position_solver_iteration_count}\n")
            robot_in_world.set_solver_velocity_iteration_count(
                velocity_solver_iteration_count)
            robot_in_world.set_solver_position_iteration_count(
                position_solver_iteration_count)

            # Set Isaac Sim solver type to Projected Gauss-Seidel (PGS)
            print("Setting Isaac Sim Physics solver type...")
            solver_type = "PGS"
            print(f"<Isaac Sim Solver Type>:\n{solver_type}\n")
            isaac_world._physics_context.set_solver_type(solver_type)

            # Add Isaac core Robot instance to the scene
            robot = isaac_world.scene.add(robot_in_world)
            print(f"<Robot>:\n{robot}\n")
            print("Importing curobo robot from URDF...[DONE]")
            return robot, robot_stage_path

        def add_curobo_world_obstacles_to_isaac_world(curobo_world_config, isacc_world):
            print(f"Adding curobo world obstacles to isaac world...")
            # Add curobo world config to isaac world
            from curobo.util.usd_helper import UsdHelper
            self.usd_helper = UsdHelper()
            self.usd_helper.load_stage(isacc_world.stage)
            self.usd_helper.add_world_to_stage(
                curobo_world_config, base_frame="/World")

        def create_motion_gen_and_warm_up(robot_config, world_model, tensor_args, interpolation_dt, collision_activation_distance, collision_cache=None, collision_checker_type=None):
            # Load motion generation configuration from robot and world configurations
            from curobo.geom.sdf.world import CollisionCheckerType
            if collision_checker_type == None:
                collision_checker_type = CollisionCheckerType.PRIMITIVE

            # Create motion generation configuration
            from curobo.wrap.reacher.motion_gen import MotionGenConfig
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                # Robot configuration to load
                robot_cfg=robot_config,
                # World configuration to load
                world_model=world_model,
                # Tensor device type
                tensor_args=tensor_args,
                # Interpolation dt to use for output trajectory
                interpolation_dt=interpolation_dt,
                # Distance(in meters) at which collision checking is activated
                collision_activation_distance=collision_activation_distance,
                # Maximum number of steps to interpolate
                interpolation_steps=5000,
                # The number of IK seeds to run per query problem
                num_ik_seeds=50,
                # The number of trajectory optimization seeds to use per query problem
                num_trajopt_seeds=6,
                # The number of iterations of the gradient descent trajectory optimizer
                grad_trajopt_iters=500,
                # Whether to evaluate the interpolated trajectory
                evaluate_interpolated_trajectory=True,
                # Trajectory optimization time step
                trajopt_dt=0.5,
                # The number of timesteps for the trajectory optimization
                trajopt_tsteps=34,
                # Joint-space trajectory optimization time step
                js_trajopt_dt=0.5,
                # The number of timesteps for the joint-space trajectory optimization
                js_trajopt_tsteps=34,
                # Whether or not to use a CUDA graph, a mechanism for optimizing the execution of CUDA programs
                use_cuda_graph=True,
                # Number of graph planning seeds to use per query problem
                num_graph_seeds=12,
                # Collision checker type for curobo
                collision_checker_type=collision_checker_type,
                # collision cache helps the collision checker pre-allocate memory and optimize its performance
                collision_cache=collision_cache,
                # Whether to check self-collision
                self_collision_check=True,
                # Find trajectory with fixed number of iterations to improve the efficiency of the motion planning algorithm
                fixed_iters_trajopt=True,
            )

            # Instantiate MotionGen with the motion generation configuration
            from curobo.wrap.reacher.motion_gen import MotionGen
            motion_gen = MotionGen(motion_gen_config)

            # Warm up the motion gen
            try:
                print("Warming up curobo motion generation... [START]")
                print("Please wait...")
                motion_gen.warmup()
            except Exception as e:
                print("Warming up curobo motion generation... [FAILED]")
                print(f"Error: {e}")
            else:
                print("Warming up curobo motion generation... [DONE]")

            return motion_gen

        def add_essential_isaac_extensions(simulation_app):
            from omni.isaac.core.utils.extensions import enable_extension
            ext_list = [
                "omni.kit.asset_converter",
                "omni.kit.tool.asset_importer",
                "omni.isaac.asset_browser",
            ]
            [enable_extension(x) for x in ext_list]
            simulation_app.update()
            return True

        # Set up curobo logger
        from curobo.util.logger import setup_curobo_logger
        setup_curobo_logger("warn")
        print("Curobo Setup... [START]")

        # Load curobo robot configuration
        self.curobo_robot_config = load_curobo_robot_config(
            self.curobo_robot_config_file_name)
        # This list specifies the names of individual joints in the arm
        self.joint_names = self.curobo_robot_config["kinematics"]["cspace"]["joint_names"]
        print(f"<Joint Names>:\n{self.joint_names}\n")
        # This list represents the desired positions for each joint when the arm is in a retracted or idle state
        self.initial_joint_angles = self.curobo_robot_config["kinematics"]["cspace"]["retract_config"]
        print(f"<Initial Joint Angles>:\n{self.initial_joint_angles}\n")

        # Load curobo world configuration
        self.curobo_world_config = load_curobo_world_config(
            curobo_world_config_file_name=self.curobo_world_config_file_name)

        # Instantiate TensorDeviceType with CUDA device 0
        from curobo.types.base import TensorDeviceType
        self.tensor_args = TensorDeviceType(
            device=torch.device("cuda:0"), dtype=torch.float32)

        # Info@HermanYe: Modify the following parameters to fit your needs
        # Set collision checker type
        from curobo.geom.sdf.world import CollisionCheckerType
        # Choose from PRIMITIVE, MESH, BLOX
        self.collision_checker_type = CollisionCheckerType.MESH

        # Set pre-defined obstacle number for collision cache
        number_of_obstacle_cuboids = 30
        number_of_obstacle_mesh = 10
        collision_cache = {"obb": number_of_obstacle_cuboids,
                           "mesh": number_of_obstacle_mesh}

        # Set interpolation dt
        interpolation_dt = 0.02

        # Set collision activation distance in meters
        collision_activation_distance = 0.02

        # Create curobo motion gen instance
        self.curobo_motion_gen = create_motion_gen_and_warm_up(
            robot_config=self.curobo_robot_config,
            world_model=self.curobo_world_config,
            tensor_args=self.tensor_args,
            interpolation_dt=interpolation_dt,
            collision_activation_distance=collision_activation_distance,
            collision_cache=collision_cache,
            collision_checker_type=self.collision_checker_type
        )

        # Isaac Sim related
        if self.use_sim:
            # Add the curobo robot to the Isaac Sim scene
            self.robot, self.robot_prim_path = add_curobo_robot_to_isaac_scene(
                self.curobo_robot_config,
                self.isaac_world,
                robot_position_in_world=np.array([0, 0, 0]),
                robot_orientation_in_world=np.array([1, 0, 0, 0]),
            )

            # Get the implicit PD controller of the robot, which can be used to set the parameters of the PD controller, apply actions, switch control modes, etc
            self.robot_articulation_controller = self.robot.get_articulation_controller()

            # Add essential isaac extensions
            add_essential_isaac_extensions(simulation_app)

            # Add curobo world obstacles to isaac world
            add_curobo_world_obstacles_to_isaac_world(
                self.curobo_world_config, self.isaac_world)

        print("Curobo Setup... [DONE]")

    def curobo_update(self):
        # Ignore these obstacles
        ignore_list = [
            self.robot_prim_path,
            "/World/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        # Get obstacles from isaac world
        obstacles = self.usd_helper.get_obstacles_from_stage(
            # the obstacles with respect to the robot
            reference_prim_path=self.robot_prim_path,
            # Ignore the robot, target cube, and ground plane
            ignore_substring=ignore_list,
        ).get_collision_check_world()
        # Update curobo world for motion generation
        self.curobo_motion_gen.update_world(obstacles)
        print(
            f"Updated curobo obstacles world with respect to the robot... [current sim step index: {self.current_step_index}]")


def cuda_pytorch_environment_check(print_info=True, print_test=False):
    try:
        # Check PyTorch version
        torch_version = torch.__version__

        # Check torchvision version
        torchvision_version = torchvision.__version__

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()

        # Check CUDA version
        cuda_version = torch.version.cuda if cuda_available else "N/A (CUDA not available)"

        # Check cuDNN version
        cudnn_version = str(torch.backends.cudnn.version())

        # Get the current CUDA device
        current_device = torch.cuda.current_device()

        # Get the number of CUDA devices
        num_devices = torch.cuda.device_count()

        # Get the name of the current CUDA device
        device_name = torch.cuda.get_device_name(current_device)

        # Print info
        if print_info:
            print("CUDA PYTORCH ENVIRONMENT CHECK")
            print("##############################")
            print(f"Torch Version: {torch_version}")
            print(f"Torchvision Version: {torchvision_version}")
            print(f"CUDA with torch available: {cuda_available}")
            print(f"CUDA Version: {cuda_version}")
            print(f"cuDNN Version: {cudnn_version}")
            print(f"Current CUDA device: {current_device}")
            print(f"Number of CUDA devices: {num_devices}")
            print(f"Name of current CUDA device: {device_name}")
            print("##############################")

        # Test CUDA tensors
        test_cuda_a = torch.cuda.FloatTensor(2).zero_()
        test_cuda_b = torch.randn(2).cuda()
        test_cuda_c = test_cuda_a + test_cuda_b
        test_cuda_d = torch.zeros(4, device="cuda:0")

        # Print test results
        if print_test:
            print("\nCUDA TENSOR TEST")
            print("##############################")
            print("Tensor operations:")
            print("Tensor a (float 2) = " + str(test_cuda_a))
            print("Tensor b (randn 2) = " + str(test_cuda_b))
            print("Tensor c ( a + b ) = " + str(test_cuda_c))
            print("Tensor d (zeros 4) = " + str(test_cuda_d))
            print("##############################")

    except Exception as e:
        print("CUDA PYTORCH CHECK FAILED")
        print(f"Error: {e}")
    else:
        print("\nCUDA PYTORCH CHECK DONE\n")


def parse_command_line_arguments():
    # Import the required modules
    import argparse
    # Instantiate the parser
    command_line_arguments_parser = argparse.ArgumentParser()

    # Curobo robot configuration file
    command_line_arguments_parser.add_argument(
        "--robot_config",
        type=str,
        default="franka.yml",
        help="curobo robot configuration(.yml) to load"
    )

    # Curobo world configuration file
    command_line_arguments_parser.add_argument(
        "--world_config",
        type=str,
        default="collision_table.yml",
        help="curobo world configuration(.yml) to load"
    )

    # Enable nvblox
    command_line_arguments_parser.add_argument(
        "--nvblox",
        action="store_true",
        help="When True, enables nvblox in Isaac Sim",
    )
    # Collision spheres visualization
    command_line_arguments_parser.add_argument(
        "--visualize_spheres",
        action="store_true",
        help="When True, visualizes robot collision spheres in Isaac Sim",
    )

    # Parse the arguments
    command_line_args = command_line_arguments_parser.parse_args()

    # Return the parsed arguments
    return command_line_args


def main():
    # Check CUDA and PyTorch environment
    cuda_pytorch_environment_check()

    # Parse command line arguments
    command_line_args = parse_command_line_arguments()

    # Create a Curobo robot
    my_curobo_robot = MyCuroboRobot(
        command_line_args.robot_config, command_line_args.world_config, command_line_args.visualize_spheres)

    # Run curobo isaac sim demo
    my_curobo_robot.isaac_run()

    # Close the demo
    my_curobo_robot.isaac_close()


if __name__ == "__main__":
    main()
