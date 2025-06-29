import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import Input
from keras.models import Model

# Load the model without compiling
model = load_model('dqn_9.keras')

# Get the input shape from the first layer
#input_shape = model.layers[0].input_shape[1:]

# Create a new model with an Input layer
i#nputs = Input(shape=input_shape)
#outputs = model(inputs)  # Connect the loaded model to the Input layer
#new_model = Model(inputs=inputs, outputs=outputs)

#from keras.models import load_model
#new_model = load_model('dqn_9.keras')
# Now, you can use the model for inference or fhurther testing

# Example input: positions (number_of_bots x 2), t_values as an array of zeros, and task_position as a 2D coordinate
NUM_BOTS = 3  # Number of bots
positions = np.array([[1, 2], [3, 4], [5, 6]])  # Example positions (shape: [3, 2])
t_values = np.zeros(NUM_BOTS)  # Array of zeros with length equal to number of bots
task_position = np.array([7, 8])  # Task position (2D: [x, y])
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