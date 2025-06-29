import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

# Load the model without compiling
model = load_model('dqn_9.keras')
model.summary()

def load_robot_positions(robot_file):
    """
    Load robot positions from a JSON file and store them in a NumPy array.
    
    Args:
    - robot_file (str): Path to the JSON file containing robot positions.
    
    Returns:
    - np.array: Array of shape (num_robots, 2) with robot positions (x, y).
    """
    try:
        with open(robot_file, 'r') as f:
            robot_data = json.load(f)

        positions = []
        for i in range(len(robot_data)):
            # Extract the x, y coordinates of each robot
            x = robot_data[str(i)]['position']['x']
            y = robot_data[str(i)]['position']['y']
            positions.append([x, y])
        
        # Convert the list of positions to a NumPy array
        return np.array(positions)

    except Exception as e:
        print(f"Error loading robot positions from file {robot_file}: {e}")
        return None

def load_task_position(task_file):
    """
    Load the task position and label from a JSON file and store the position in a NumPy array.
    
    Args:
    - task_file (str): Path to the JSON file containing the task position.
    
    Returns:
    - tuple: (task_position_array, task_label) 
      - task_position_array (np.array): Array with the task position [x_task, y_task].
      - task_label (str): The label of the task.
    """
    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        # Extract task data (x, y position and label)
        task_position = task_data["0"]["position"]
        task_label = task_data["0"]["label"]
        
        # Convert to a NumPy array for position
        task_position_array = np.array([task_position["x"], task_position["y"]])

        return task_position_array, task_label

    except Exception as e:
        print(f"Error loading task position from file {task_file}: {e}")
        return None, None


ACCUMULATION_STEPS = 4 
###########################################################################
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name):
        self.model.save(name + '.keras')

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # Accumulate gradients over `ACCUMULATION_STEPS` mini-batches
        accumulated_grads = None
        for step, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            # Perform forward pass and compute gradients
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                model_output = self.model(state)
                loss = tf.keras.losses.mean_squared_error(target, model_output)

            # Accumulate gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = [accumulated_grad + grad for accumulated_grad, grad in zip(accumulated_grads, grads)]

            # After `ACCUMULATION_STEPS` mini-batches, apply gradients
            if (step + 1) % ACCUMULATION_STEPS == 0:
                self.model.optimizer.apply_gradients(zip(accumulated_grads, self.model.trainable_variables))
                accumulated_grads = None  # Reset accumulated gradients

        # Update epsilon after every replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


###########################################################################

# Example input: positions (number_of_bots x 2), t_values as an array of zeros, and task_position as a 2D coordinate

robot_file = '../../logs/robot_positions.json'  # Change to the actual path of the robot positions file
task_file = '../../logs/TaskObjectLocation.json'    # Change to the actual path of the task position file

NUM_BOTS = 10  # Number of bots
positions = load_robot_positions(robot_file)
t_values = np.zeros(NUM_BOTS)  # Array of zeros with length equal to number of bots
task_position, task_label = load_task_position(task_file)
state_size = (NUM_BOTS * 2) + NUM_BOTS + 2
action_size = NUM_BOTS

# Concatenate the arrays
state = np.concatenate((
    positions.flatten(),  # Flatten positions to a 1D array
    t_values,             # Add t_values (array of zeros)
    task_position         # Add task position (2D)
))

state = np.reshape(state, [1, state_size])


# Create a DQN agent using the loaded model
test_agent = DQNAgent(state_size, action_size)
test_agent.model = model  # Set the model of the agent to the loaded model

# Get the best action from the trained model
action = test_agent.act(state)
print(f"The action chosen by the model: {action}")

# Now update the task JSON with the selected "alloted_bot"
task_data = {
    "0": {
        "label": task_label,
        "position": {
            "x": task_position[0],
            "y": task_position[1]
        },
        "alloted_bot": action  # Allotted bot is the action taken by the model
    }
}

# Save the updated task JSON to a file
output_file = '../logs/TaskObjectWithAllocation.json'  # You can change the output file path
with open(output_file, 'w') as f:
    json.dump(task_data, f, indent=4)

print(f"Task data updated and saved to {output_file}")
