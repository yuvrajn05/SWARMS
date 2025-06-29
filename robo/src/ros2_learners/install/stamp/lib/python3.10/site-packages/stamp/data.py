import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json
import os

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')

        # Subscribe to the topics
        self.odom_subscription = self.create_subscription(
            Odometry,  # Correct message type for /odom topic
            '/robot_0/odom',
            self.odom_callback,
            10
        )
        self.classification_subscription = self.create_subscription(
            String,  # Correct message type for predicted class
            '/predicted_class',
            self.classification_callback,
            10
        )

        # Data storage
        self.data = []
        self.file_path = '/home/sid/ros2_ws/src/stamp/stamp/ros_data.json'

        # Store the latest classification value (predicted class)
        self.classification_data = None  # This will hold the latest valid predicted class value

    def odom_callback(self, msg):
        self.get_logger().info("Received odometry data")  # Debug log

        # Store data only if predicted_class is a valid integer
        if self.classification_data is not None:
            # Store the odometry data along with classification data
            odom_data = {
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "classification": self.classification_data  # Add the predicted class (as string)
                }
            }

            # Append the odometry data (with classification)
            self.data.append(odom_data)
            self.save_to_json()  # Save data to JSON
        else:
            self.get_logger().info("Predicted class is not a valid integer, not storing odometry data.")

    def classification_callback(self, msg):
        # Attempt to validate if the predicted_class is a valid integer
        try:
            # Attempt to convert the string to an integer
            predicted_class = int(msg.data)  # Convert string to integer
            self.classification_data = str(predicted_class)  # Store as string
            self.get_logger().info(f"Object detected with class {self.classification_data}")
        except ValueError:
            # If conversion fails, log the error and set classification_data to None
            self.get_logger().error(f"Failed to parse predicted class: '{msg.data}' is not a valid integer.")
            self.classification_data = None  # Invalidate classification if not a valid integer

    def save_to_json(self):
        try:
            self.get_logger().info(f"Saving data to: {self.file_path}")  # Debug log
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

            # Check if the file exists, and decide whether to append or overwrite
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    existing_data = json.load(f)
                existing_data["odom"].extend(self.data)  # Append new odometry data
                with open(self.file_path, 'w') as f:
                    json.dump(existing_data, f, indent=4)
            else:
                # If the file doesn't exist, create a new one
                with open(self.file_path, 'w') as f:
                    json.dump({"odom": self.data}, f, indent=4)

            self.get_logger().info(f"Data written to {self.file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to write to JSON: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

