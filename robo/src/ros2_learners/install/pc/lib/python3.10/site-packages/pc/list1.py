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
            '/robot_1/odom',
            self.odom_callback,
            10
        )
        self.classification_subscription = self.create_subscription(
            String,  # Correct message type for predicted class
            '/predicted_class1',
            self.classification_callback,
            10
        )

        # Data storage
        self.data = {}  # Using a dictionary instead of list
        self.file_path = '../../logs/list.json'

        # Store the latest classification value (predicted class)
        self.classification_data = None  # This will hold the latest valid predicted class value

    def odom_callback(self, msg):
        self.get_logger().info("Received odometry data")  # Debug log

        # Store data only if predicted_class is a valid integer and not 67
        if self.classification_data is not None and self.classification_data != "0":
            # Check if the class already exists in the data
            if self.classification_data in self.data:
                # If the class already exists, update the position
                self.data[self.classification_data]["x"] = msg.pose.pose.position.x
                self.data[self.classification_data]["y"] = msg.pose.pose.position.y
                self.get_logger().info(f"Updated position for class {self.classification_data}")
            else:
                # If the class doesn't exist, add a new entry
                self.data[self.classification_data] = {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y
                }
                self.get_logger().info(f"New class {self.classification_data} detected and added")

            # Save data to JSON
            self.save_to_json()  # Save data to JSON
        else:
            self.get_logger().info("Predicted class is 0, not storing odometry data.")

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
                existing_data.update(self.data)  # Merge new data into existing data
                with open(self.file_path, 'w') as f:
                    json.dump(existing_data, f, indent=4)
            else:
                # If the file doesn't exist, create a new one
                with open(self.file_path, 'w') as f:
                    json.dump(self.data, f, indent=4)

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
