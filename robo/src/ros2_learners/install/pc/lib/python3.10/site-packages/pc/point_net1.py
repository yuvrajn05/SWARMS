import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import torch
import numpy as np
from std_msgs.msg import String
from pointnet_model import PointNet  # Replace with actual model import

class PointNetInferenceNode(Node):
    def __init__(self):
        super().__init__('pointnet_inference_node')
        
        # Initialize PointNet model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pointnet = PointNet().to(self.device).eval()
        self.load_model()

        # Create subscription to point cloud topic
        self.create_subscription(PointCloud2, '/robot_1/camera/points', self.point_cloud_callback, 10)

        # Create publisher for predicted class
        self.class_pub = self.create_publisher(String, '/predicted_class1', 10)

    def load_model(self):
        try:
            # Load pre-trained model
            self.pointnet.load_state_dict(torch.load("save2.pth", map_location=self.device), strict=False)
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            raise

    def point_cloud_callback(self, msg):
        # Convert the PointCloud2 message to a numpy array with 'x', 'y', 'z' fields
        pc_data = list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # Check if pc_data is empty
        if len(pc_data) == 0:
            self.get_logger().warn('Received an empty point cloud.')
            return

        # Debug: Check the structure of the pc_data
        self.get_logger().info(f"First 5 points: {pc_data[:5]}")  # Print the first 5 points for inspection
        self.get_logger().info(f"Total points: {len(pc_data)}")
        
        # Ensure it's a list of tuples (or lists)
        if isinstance(pc_data, list) and len(pc_data) > 0:
            self.get_logger().info(f"First point data: {pc_data[0]}")
        
        try:
            # Extract the 'x', 'y', and 'z' coordinates from np.void objects
            pc_array = np.array([[point[0], point[1], point[2]] for point in pc_data], dtype=np.float32)
            self.get_logger().info(f"Converted pc_array shape: {pc_array.shape}")  
        except Exception as e:
            self.get_logger().error(f"Error converting point cloud data: {e}")
            return

        # Ensure the correct shape (1, 3, N) for the model
        pc_tensor = torch.tensor(pc_array, dtype=torch.float32).unsqueeze(0).transpose(2, 1).to(self.device)
        print(pc_tensor.shape)  # Shape: (1, 3, N)
	
        # Run inference
        with torch.no_grad():
            try:
                output, matrix3x3, matrix64x64 = self.pointnet(pc_tensor)
            except Exception as e:
                self.get_logger().error(f"Inference failed: {str(e)}")
                return

        # Get predicted class (assuming the output is of shape [batch_size, num_classes])
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()

        # Log the predicted class
        self.get_logger().info(f"Predicted Class: {predicted_class}")

        # Optionally, publish the predicted class or further processing
        self.publish_predicted_class(predicted_class)

    def publish_predicted_class(self, predicted_class):
        msg = String()
        msg.data = str(predicted_class)
        self.class_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    # Create the PointNet inference node
    pointnet_inference_node = PointNetInferenceNode()

    try:
        rclpy.spin(pointnet_inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        pointnet_inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
