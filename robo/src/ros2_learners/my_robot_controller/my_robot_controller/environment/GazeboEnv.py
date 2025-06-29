from numpy.typing import NDArray
import random
from collections import namedtuple
from gymnasium import Env, spaces
import numpy as np
import time
from typing import List, Tuple
from rclpy.node import Node
from Utils import Utils
import math
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import (
    SpawnEntity, DeleteEntity, SetEntityState, GetEntityState,
    SetModelState, SetModelConfiguration, SetLinkState,
    SetLinkProperties, SetPhysicsProperties
)
from subscribers import OdomSubscriber, ScanSubscriber
from config import (
    GOAL_REACHED_THRESHOLD, OBSTACLE_COLLISION_THRESHOLD,
    LIDAR_SAMPLE_SIZE, SAVE_INTERVAL, AGENT_COUNT,
    MAX_LIDAR_RANGE, RTF
)

# (done, collision, target, min_laser_reading)
DoneTuple = namedtuple(
    'DoneTuple', ['done', 'collision', 'target', 'min_laser_reading']
)

# Define the bounds of the environment
bounds = ((-30, 30), (-30, 30))
x_grid_size = bounds[0][1] - bounds[0][0]  # 60 units
y_grid_size = bounds[1][1] - bounds[1][0]  # 60 units

# Maximum possible distance to any goal within the environment
max_distance_to_goal = math.floor(math.sqrt(x_grid_size**2 + y_grid_size**2)) * 1.0

agent_count = AGENT_COUNT  # Number of agents in the environment

class GazeboEnvMultiAgent(Env):
    def __init__(
        self,
        odom_subscribers: List[OdomSubscriber],
        scan_subscribers: List[ScanSubscriber],
        goal_position=(0., 0.),
        test=False,
        agent_count=AGENT_COUNT,
        nodename='GazeboEnv',
        alloted_bot=AGENT_COUNT + 2,
        special_agent_index: int = 1,  # Index of the agent with a different goal
        special_goal_position: Tuple[float, float] = (22.0, 14.0)  # Different goal position
    ):
        super(GazeboEnvMultiAgent, self).__init__()

        self.odom_subscribers = odom_subscribers
        self.scan_subscribers = scan_subscribers
        self.test = test
        self.special_agent_index = special_agent_index

        # Define action space: (linear_x velocity, angular_z velocity)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Define observation space: (lidar ranges, distance to goal, angle to goal, last action)
        self.output_shape = (184,)
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.zeros(LIDAR_SAMPLE_SIZE),
                np.array([0.0, -1.0]),
                np.array([-1.0, -1.0])
            )),
            high=np.concatenate((
                np.full(LIDAR_SAMPLE_SIZE, 1.0),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0])
            )),
            dtype=np.float32
        )

        self.agent_count = agent_count
        self.reward_range = (-200, 200)

        # Initialize ROS node
        self.node = Node(nodename)
        self.vel_pubs = {}

        for agent_index in range(agent_count):
            topic_name = f"/robot_{agent_index+1}/cmd_vel"
            self.vel_pubs[agent_index] = self.node.create_publisher(Twist, topic_name, 10)

        # Initialize service clients
        self.unpause = self.node.create_client(Empty, "/unpause_physics")
        self.pause = self.node.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.node.create_client(Empty, "/reset_world")
        self.spawn_goal_client = self.node.create_client(SpawnEntity, "/spawn_entity")
        self.delete_goal_client = self.node.create_client(DeleteEntity, "/delete_entity")
        self.set_entity_state = self.node.create_client(SetEntityState, "/gazebo/set_entity_state")

        self.req = Empty.Request

        # Define per-agent goal positions
        self.agent_goal_positions = [goal_position for _ in range(agent_count)]
        # Assign a different goal to the specified agent
        if 0 <= special_agent_index < agent_count:
            self.agent_goal_positions[special_agent_index] = special_goal_position
            self.special_agent_index = special_agent_index
            self.special_goal_position = special_goal_position
        else:
            self.node.get_logger().warn(
                f"Special agent index {special_agent_index} is out of range. No agent will have a different goal."
            )
            self.special_agent_index = None

        # Initialize last actions and previous distances to goals
        self.last_actions = {agent_index: (0.0, 0.0) for agent_index in range(agent_count)}
        self.prev_distances_to_goal = [
            Utils.get_distance_to_goal(
                self.get_robot_position(agent_index),
                self.agent_goal_positions[agent_index]
            ) for agent_index in range(agent_count)
        ]

    # Predefined list of potential goal positions
    goal_positions = [
        (19.0, 1.0), (16.968, -14.4), (24.0533, -14.316695),
        (8.0983, -9.77432), (-3.252, -2.312), (-2.57719, 7.596120),
        (18.689, 12.88605), (5.98, 11.43), (6.12, 15.416695)
    ]

    def raise_robot_model(self, robot_index: int):
        msg = SetLinkProperties.Request()
        msg.link_name = "my_robot_1::base_footprint"
        msg.gravity_mode = False

        model_name: str = f"my_robot_{robot_index+1}"
        print(f"Raising {model_name}")
        request = SetEntityState.Request()
        request.state.name = model_name

        request.state.pose.position.z = -2.0

        while not self.set_entity_state.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Waiting for set_entity_state service to be available")
        try:
            self.set_entity_state.call_async(request)
        except:
            self.node.get_logger().error("/gazebo/set_entity_state service call failed!")

    def change_goal_position(self, agent_index: int = None):
        """
        Change the goal position for a specific agent.
        If agent_index is None, change goal positions for all agents.
        """
        if agent_index is not None:
            if 0 <= agent_index < self.agent_count:
                # Assign a new goal position for the specified agent
                new_goal = random.choice(self.goal_positions)
                if(agent_index != self.special_agent_index):
                    self.agent_goal_positions[agent_index] = new_goal

                request = SetEntityState.Request()
                # Define unique goal entity names per agent
                goal_entity_name = f"bookshelf_agent_{agent_index+1}"

                request.state.name = goal_entity_name
                request.state.pose.position.x = new_goal[0]
                request.state.pose.position.y = new_goal[1]

                # Set a default orientation
                request.state.pose.orientation.x = 0.0
                request.state.pose.orientation.y = 0.0
                request.state.pose.orientation.z = 0.7089829505001405
                request.state.pose.orientation.w = 0.7052256205641677

                while not self.set_entity_state.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().info("Waiting for set_entity_state service to be available")
                try:
                    self.set_entity_state.call_async(request)
                    print(f"Agent {agent_index + 1} goal position changed to: {self.agent_goal_positions[agent_index]}")
                except:
                    self.node.get_logger().error("/gazebo/set_entity_state service call failed!")
            else:
                self.node.get_logger().warn(f"Agent index {agent_index} is out of range.")
        else:
            # Change goal positions for all agents
            for idx in range(self.agent_count):
                self.change_goal_position(agent_index=idx)

    def change_special_goal_position(self,pos):
        self.special_goal_position = pos
        self.agent_goal_positions[self.special_agent_index] = self.special_goal_position
        print(f"\nAgent {self.special_agent_index + 1} goal position changed to: {self.agent_goal_positions[self.special_agent_index]}\n")

    def change_special_agent_index(self,alloted_bot):
        self.special_agent_index = alloted_bot
        print(f"\nalloted bot is {self.special_agent_index}")
        

    def step(self, action_n: List):
        cmds = []

        for i in range(self.agent_count):
            linear_x = float(action_n[i][0])
            angular_z = float(action_n[i][1])
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            cmds.append(msg)
            self.last_actions[i] = (linear_x, angular_z)

        # Publish velocity commands for all agents
        for i in range(self.agent_count):
            cmd = cmds[i]
            self.vel_pubs[i].publish(cmd)

        # Unpause physics to let simulation proceed
        self.unpause_physics()

        # Wait for the simulation to update
        time.sleep(0.2 / RTF)

        # Pause physics to stabilize the environment state
        self.pause_physics()

        # Retrieve observations
        observations = self.get_obs()

        # Check termination conditions
        terminated, collision, target, min_laser_readings = self.check_done()

        # Calculate rewards
        rewards = self.get_reward(
            terminated, collision, target, action_n, min_laser_readings
        )

        # Compile additional information
        info = {
            "terminated": terminated,
            "collision": collision,
            "target": target
        }

        return observations, rewards, terminated, info

    def unpause_physics(self):
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Unpause physics service not available, waiting")

        try:
            self.unpause.call_async(Empty.Request())
        except:
            self.node.get_logger().error("Unpause physics service call failed!")

    def pause_physics(self):
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("Pause physics service not available, waiting")

        try:
            self.pause.call_async(Empty.Request())
        except:
            self.node.get_logger().error("Pause physics service call failed!")

    def reset(self):
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("/reset_world service not available, waiting")

        try:
            self.reset_proxy.call_async(Empty.Request())
        except:
            self.node.get_logger().error("/reset_world service call failed!")

        # Optionally, change goal positions after reset
        # Uncomment the following line to randomize goal positions on reset
        # self.change_goal_position()

        time.sleep(0.1 / RTF)

        # Retrieve and return initial observations
        observations = self.get_obs()

        return observations

    def get_obs(self):
        """
        Returns a list of observations for each agent.
        Each observation consists of:
        - Normalized and reduced LiDAR ranges
        - Normalized distance to the agent's goal
        - Normalized angle to the agent's goal
        - Last action taken by the agent
        """
        observations = []
        max_lidar_range = MAX_LIDAR_RANGE

        for i in range(self.agent_count):
            robot_position = Utils.get_position_from_odom_data(
                self.odom_subscribers[i].odom_data
            )
            orientation = self.odom_subscribers[i].odom_data.pose.pose.orientation
            robot_orientation = Utils.euler_from_quaternion(orientation)
            distance_to_goal = Utils.get_distance_to_goal(
                robot_position, self.agent_goal_positions[i]
            )
            angle_to_goal = Utils.get_angle_to_goal(
                robot_position, robot_orientation, self.agent_goal_positions[i]
            )

            # Reduce LiDAR samples for computational efficiency
            reduced_lidar_ranges = Utils.reduce_lidar_samples(
                self.scan_subscribers[i].laser_ranges, 20
            )

            # Normalize LiDAR ranges
            normalized_lidar_ranges = reduced_lidar_ranges / max_lidar_range
            normalized_lidar_ranges = np.clip(normalized_lidar_ranges, 0.0, 1.0)

            # Normalize distance and angle to goal
            normalized_dist_to_goal = distance_to_goal / max_distance_to_goal
            normalized_angle_to_goal = angle_to_goal / np.pi  # Range: [-1, 1]

            # Concatenate state parameters
            state_parameter_set = np.concatenate((
                [normalized_dist_to_goal, normalized_angle_to_goal],
                [self.last_actions.get(i)[0], self.last_actions.get(i)[1]]
            ))

            # Final observation vector
            observation = np.concatenate([normalized_lidar_ranges, state_parameter_set])

            observations.append(observation)

        return observations

    def get_robot_position(self, agent_index: int):
        odom_data = self.odom_subscribers[agent_index].odom_data
        return Utils.get_position_from_odom_data(odom_data)

    def check_done(self):
        """
        Checks termination conditions for each agent.
        Returns lists indicating whether each agent is done, has collided, reached the target, and their minimum LiDAR reading.
        """
        done_n = []
        collision_n = []
        target_n = []
        min_laser_reading_n = []

        for i in range(self.agent_count):
            done_tuple = self.check_done_id(i)
            done_n.append(done_tuple.done)
            collision_n.append(done_tuple.collision)
            target_n.append(done_tuple.target)
            min_laser_reading_n.append(done_tuple.min_laser_reading)

        return done_n, collision_n, target_n, min_laser_reading_n

    def check_done_id(self, agent_index: int) -> DoneTuple:
        """
        Determines if an agent has reached its goal or collided with an obstacle.
        """
        distance_to_goal = Utils.get_distance_to_goal(
            self.get_robot_position(agent_index),
            self.agent_goal_positions[agent_index]
        )

        min_laser_reading = min(self.scan_subscribers[agent_index].laser_ranges)

        if distance_to_goal < GOAL_REACHED_THRESHOLD:
            self.node.get_logger().info(
                f"Agent {agent_index + 1} reached the goal. Distance to goal: {distance_to_goal}"
            )
            return DoneTuple(done=True, collision=False, target=True, min_laser_reading=min_laser_reading)

        if min_laser_reading < OBSTACLE_COLLISION_THRESHOLD:
            self.node.get_logger().info(
                f"Agent {agent_index + 1} collided with an obstacle. minRange: {min_laser_reading}"
            )
            return DoneTuple(done=True, collision=True, target=False, min_laser_reading=min_laser_reading)

        return DoneTuple(done=False, collision=False, target=False, min_laser_reading=min_laser_reading)

    def get_reward(self, done_n, collision_n, target_n, action_n, min_laser_reading_n):
        """
        Calculates rewards for each agent based on their current state and actions.
        Returns a list of rewards corresponding to each agent.
        """
        rewards = []
        for i in range(self.agent_count):
            reward_i = self.get_reward_id(
                done_n[i],
                collision_n[i],
                target_n[i],
                action_n[i],
                min_laser_reading_n[i],
                i
            )
            rewards.append(reward_i)

        return rewards

    def get_reward_id(self, done: bool, collision: bool, target: bool, action: Tuple[float, float], min_laser_reading: float, agent_index: int):
        """
        Computes the reward for a single agent based on its state and action.
        """
        r_arrive = 100.0
        r_collision = -100.0

        if done:
            if target:
                return r_arrive
            if collision:
                return r_collision
            else:
                return 0.0
        else:
            # Reward structure:
            # Encourage forward movement (linear_x)
            # Penalize excessive turning (angular_z)
            # Penalize proximity to obstacles based on LiDAR readings
            linear_reward = action[0] / 2  # Normalize linear velocity
            angular_penalty = abs(action[1]) / 2  # Normalize angular velocity
            obstacle_penalty = (1 - min_laser_reading) / 2  # Normalize LiDAR reading

            total_reward = linear_reward - angular_penalty - obstacle_penalty
            return total_reward

    def calculate_approach_reward(self):
        """
        (Optional) Calculates a cumulative reward based on the progress each agent has made towards its goal.
        Currently not integrated into the main reward calculation.
        """
        k = 50
        total_approach_reward = 0.0
        for i in range(agent_count):
            current_distance_to_goal = Utils.get_distance_to_goal(
                self.get_robot_position(i),
                self.agent_goal_positions[i]
            )

            approach_dist = self.prev_distances_to_goal[i] - current_distance_to_goal
            approach_dist *= k

            total_approach_reward += approach_dist

        return total_approach_reward

    def close(self):
        """
        Cleans up the ROS node when the environment is closed.
        """
        self.node.destroy_node()