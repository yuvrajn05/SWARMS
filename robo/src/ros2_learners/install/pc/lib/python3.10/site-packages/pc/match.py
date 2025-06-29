import json
import os
import time

class LabelPositionMatcher:
    def __init__(self, positions_file, labels_file, output_file, poll_interval=1.0):
        self.positions_file = positions_file
        self.labels_file = labels_file
        self.output_file = output_file
        self.poll_interval = poll_interval  # Poll every 1 second by default

        self.last_positions_mod_time = 0
        self.last_labels_mod_time = 0

    def load_data(self):
        """Load both position data and labels data."""
        try:
            # Load position data from the positions file
            with open(self.positions_file, 'r') as pos_file:
                self.positions_data = json.load(pos_file)

            # Load label data from the classification labels file
            with open(self.labels_file, 'r') as labels_file:
                self.labels_data = json.load(labels_file)

            print(f"Successfully loaded data from {self.positions_file} and {self.labels_file}")
        except Exception as e:
            print(f"Error loading files: {e}")

    def match_labels_with_positions(self):
        """Match classification IDs with their positions and print them."""
        # Check if position data exists
        if not self.positions_data:
            print(f"No position data found in {self.positions_file}")
            return
        
        # The dictionary where the matched results will be saved
        matched_results = {}

        # Iterate through the positions and match them with the labels
        for class_id, position in self.positions_data.items():
            if class_id in self.labels_data:
                label = self.labels_data[class_id]
                position_data = position
                # Print the matched results
                print(f"Class ID: {class_id} - {label}")
                print(f"Position - x: {position_data['x']}, y: {position_data['y']}")
                matched_results[class_id] = {
                    "label": label,
                    "position": position_data
                }
            else:
                print(f"Class ID: {class_id} does not have a label in {self.labels_file}")
        
        # Save the matched results to the output file
        self.save_results_to_json(matched_results)

    def save_results_to_json(self, matched_results):
        """Save the matched results to the output JSON file."""
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            with open(self.output_file, 'w') as output_file:
                json.dump(matched_results, output_file, indent=4)

            print(f"Matched results saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")

    def check_for_changes(self):
        """Check if the positions or labels files have been modified."""
        try:
            positions_mod_time = os.path.getmtime(self.positions_file)
            labels_mod_time = os.path.getmtime(self.labels_file)

            if positions_mod_time != self.last_positions_mod_time or labels_mod_time != self.last_labels_mod_time:
                self.last_positions_mod_time = positions_mod_time
                self.last_labels_mod_time = labels_mod_time

                # Reload data if files have changed
                self.load_data()

                # Match and save the updated results
                self.match_labels_with_positions()
        except Exception as e:
            print(f"Error checking for file changes: {e}")

    def run(self):
        """Continuously monitor and process the files."""
        while True:
            self.check_for_changes()  # Check if files have been updated
            time.sleep(self.poll_interval)  # Sleep for the polling interval (e.g., 1 second)

def main():
    # Paths to the files
    positions_file = '../../logs/list.json'  # Update with your positions file path
    labels_file = '../../logs/PromtObject.json'  # Update with your classification labels file path
    output_file = '../../logs/TaskObjectLocation.json'  # Path to save the output JSON file

    # Create an instance of the LabelPositionMatcher
    matcher = LabelPositionMatcher(positions_file, labels_file, output_file, poll_interval=1.0)

    # Start the continuous processing
    matcher.run()

if __name__ == '__main__':
    main()
