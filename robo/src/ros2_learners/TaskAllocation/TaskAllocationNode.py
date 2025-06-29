import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
import json
import os

# Global variables to store the robot's current positions (as a NumPy array)
positions = np.zeros((10, 2))  # Array to store positions of 10 robots, each with x, y

# Task position (same for all robots)
task_position = {'x': 0.0, 'y': 0.0}

# JSON file path where positions will be stored
json_file_path = "../logs/robot_positions.json"

class MultiRobotTaskNode(Node):
    def __init__(self):
        super().__init__('multi_robot_task_node')

        # Read task position from JSON file (if you still want to load and display it)
        self.read_task_position("../logs/list.json")

        # Check if the robot positions JSON file exists, if not create it
        if not os.path.exists(json_file_path):
            self.create_initial_json_file()

        # Subscribe to the /robot_{i}/odom topic for each robot (1 to 10)
        for i in range(1, 11):
            self.create_subscription(
                Odometry,
                f'/robot_{i}/odom',
                lambda msg, robot_id=i: self.odom_callback(msg, robot_id),
                10
            )

    def odom_callback(self, msg, robot_id):
        # Update the position of the specific robot in the positions array
        positions[robot_id - 1] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        
        # Store the position in the JSON file
        self.store_positions_to_json()

    def read_task_position(self, file_path="../logs/list.json"):
        global task_position
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                task_position = data.get("1", {}).get("position", {'x': 0.0, 'y': 0.0})
                self.get_logger().info(f"Task Position - x: {task_position['x']}, y: {task_position['y']}")
        except Exception as e:
            self.get_logger().error(f"Failed to read task position file: {e}")

    def create_initial_json_file(self):
        # Create the initial structure for the JSON file
        initial_data = {}

        for i in range(10):
            initial_data[str(i)] = {
                "name": f"robot_{i+1}",
                "position": {
                    "x": 0.0,
                    "y": 0.0
                }
            }

        # Write the initial data to the JSON file
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(initial_data, json_file, indent=4)
            self.get_logger().info(f"Initial positions file created: {json_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to create initial positions file: {e}")

    def store_positions_to_json(self):
        # Prepare the dictionary to store in JSON format
        robot_positions = {}

        # Populate the dictionary with robot data
        for i in range(10):
            robot_positions[str(i)] = {
                "name": f"robot_{i+1}",
                "position": {
                    "x": positions[i][0],
                    "y": positions[i][1]
                }
            }

        # Write the dictionary to a JSON file
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(robot_positions, json_file, indent=4)
            self.get_logger().info(f"Updated positions saved to {json_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save positions to JSON file: {e}")

def main():
    rclpy.init()

    # Create an instance of the MultiRobotTaskNode
    node = MultiRobotTaskNode()

    # Keep the node alive to listen to odometry data
    rclpy.spin(node)

    # Shutdown and clean up after spinning
    rclpy.shutdown()

if __name__ == '__main__':
    main()
