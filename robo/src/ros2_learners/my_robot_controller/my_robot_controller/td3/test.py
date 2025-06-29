from torch.utils.tensorboard import SummaryWriter
from numpy import inf
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
from rclpy.executors import MultiThreadedExecutor
import rclpy
import threading
import time

sys.path.append("../environment")  # nopep8

from config import *
from GazeboEnv import GazeboEnvMultiAgent
from subscribers import OdomSubscriber, ScanSubscriber

# TODO:
#   - dynamic target spawn (place the goal close first then far)
#   - change optimizer Adam to RAdam


# td3 code
# ===============================================

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # cuda or cpu


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2

# td3 network


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(
            log_dir="./runs")
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0

        print(f"train function iteration count: {iterations}")
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(
                0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (
                next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + \
                F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        env.node.get_logger().info(f"writing new results for a tensorboard")
        env.node.get_logger().info(
            f"loss, Av.Q, Max.Q, iterations : {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

        # TODO (optional): add hyperparameters to tensorboard
        # self.writer.add_hparams()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" %
                   (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" %
                   (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# td3 code end
# ===============================================

def my_function(env):
    env.change_goal_position()
    print(f"Function called at {time.time()}")
    # Schedule the next call of the function after 10 seconds
    threading.Timer(200, my_function, [env]).start()


if __name__ == "__main__":
    rclpy.init()

    seed = 0  # Random seed number
    max_ep = 500  # maximum number of steps per episode
    file_name = "td3_policy"  # load policy from this file

    # Create the training environment
    # laser sample count
    environment_dim = LIDAR_SAMPLE_SIZE
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    # Create the network
    network = TD3(state_dim, action_dim, max_action)

    try:
        print("loading policy...")
        network.load(file_name, "./pytorch_models")
    except:
        raise ValueError('Could not load the stored model parameters')

    episode_timesteps = 0
    done = True

    goal_position = (0.43, 1.58)

    namespaces = [f"robot_{i+1}" for i in range(AGENT_COUNT)]

    # set up subscribers and environment
    executor = MultiThreadedExecutor()
    odom_subscribers = []
    scan_subscribers = []
    for i, namespace in enumerate(namespaces):
        robot_index = i
        odom_subscriber = OdomSubscriber(namespace, robot_index)
        scan_subscriber = ScanSubscriber(namespace, robot_index)
        odom_subscribers.append(odom_subscriber)
        scan_subscribers.append(scan_subscriber)
        executor.add_node(odom_subscriber)
        executor.add_node(scan_subscriber)

    env = GazeboEnvMultiAgent(odom_subscribers=odom_subscribers, scan_subscribers=scan_subscribers,
                              goal_position=goal_position, test=True)
    executor.add_node(env.node)

    # TODO: ros sleep node

    executor_thread = threading.Thread(target=executor.spin, daemon=False)
    executor_thread.start()

    # get observation shapes []
    obs_shape_n = [
        env.observation_space.shape for i in range(env.agent_count)]

    action_space_n = [env.action_space for i in range(env.agent_count)]



    print('Starting test...')

    # Initialize the environment (no reset needed)
    prev_observation_n = env.reset()  # Initial state at t=0

    episode_timesteps = 0  # Keep track of episode time steps
    done = False  # Keep track of whether the episode is done

    ij = 0

    while rclpy.ok():
        # Get actions from the previous observations
        action_n = []
        for i in range(AGENT_COUNT):
            action = network.get_action(np.array(prev_observation_n[i]))
            action_n.append(action)

        # Scale actions to fit proper ranges (assuming normalized actions)
        a_in_n = []
        for i, action in enumerate(action_n):
            a_in = [(action[0] + 1) / 2, action[1]]  # Assuming action[0] is normalized
            a_in_n.append(a_in)

        # Take the step in the environment with the actions
        observation, reward, done_n, info = env.step(a_in_n)

        # Handle the logic when a reset condition is met, instead of using env.reset():
        # Check if any reset conditions are met (e.g., max episode time, collisions, target reached)
        if episode_timesteps + 1 == max_ep:
            print("Max episode steps reached. Continue with next episode.")
            done = True  # End the episode
        elif all(info.get('collision')):
            print("All agents collided. Continue with the next episode.")
            done = True  # End the episode due to collision
        elif all(info.get('target')):
            print("All agents reached the target. Continue with next episode.")
            done = True  # End the episode because agents reached the target

        if(ij%10 == 0):
            #ij = 0
            print(ij)

            my_function(env)
            # time.sleep(30)
        ij = ij + 1

        # If the episode is done, you can:
        if done:
            # Optionally, you can set the environment to a new state (without calling env.reset())
            # If the environment needs to be reset (e.g., change goals, reinitialize states), handle it here
            print(f"Episode {episode_timesteps + 1} finished. Restarting from the same state.")
            
            # If you need to change the environment state without a full reset:
            # For example, you could reinitialize the goal position for the agents:
            # env.set_new_goals()  # Hypothetical function if needed for new goal
            
            done = False  # Set done to False to continue
            episode_timesteps = 0  # Reset the episode time steps
            
        else:
            # If not done, continue to the next time step
            prev_observation_n = observation  # Update previous observation
            episode_timesteps += 1  # Increment the episode time steps

        # Debugging print statements
        print(f"Episode Timesteps: {episode_timesteps}")
        if info:
            collision = info.get('collision')
            target = info.get('target')
            print(f"Collision: {collision}, Target reached: {target}")
        
        # Optional: Print every certain number of steps to check progress
        if episode_timesteps % 50 == 0:
            print(f"At timestep {episode_timesteps}, agents' actions and rewards being processed.")

    rclpy.shutdown()  # Shut down the ROS node when done
