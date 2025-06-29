from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from numpy import inf
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import sys
import os
from rclpy.executors import MultiThreadedExecutor
import rclpy
import threading
import matplotlib.pyplot as plt

sys.path.append("../environment")  # nopep8

from config import *
from config import _MAX_EP, _EVAL_EP, _MAX_TIMESTEPS, _EXPL_NOISE, _EXPL_DECAY_STEPS,  _EXPL_MIN, _BATCH_SIZE, _DISCOUNT, _TAU, _POLICY_NOISE, _NOISE_CLIP, _POLICY_FREQ, _BUFFER_SIZE, _EVAL_FREQ

from GazeboEnvTrain import GazeboEnvMultiAgent
from subscribers import OdomSubscriber, ScanSubscriber

# TODO:
#   - dynamic target spawn (place the goal close first then far)
#   - change optimizer Adam to RAdam


# td3 code
# ===============================================

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # cuda or cpu


def plot_performance(evaluations):
    # Extract the average rewards from the evaluations
    avg_rewards = [eval[0] for eval in evaluations]  # assuming evaluate() returns avg_reward as the first value
    avg_col = [eval[1] for eval in evaluations]  # average collision count or any other metric
    
    plt.figure(figsize=(10, 5))
    
    # Plot the average reward per evaluation
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards, label='Average Reward')
    plt.title('Average Reward over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.legend()
    
    # Plot the collision rate (if necessary)
    plt.subplot(1, 2, 2)
    plt.plot(avg_col, label='Average Collision Rate', color='red')
    plt.title('Collision Rate over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Collision Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.show()




def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        env.node.get_logger().info(f"evaluating episode {_}")
        count = 0
        state_n = env.reset()
        done_n = [False]
        while not any(done_n) and count < 501:
            # action = network.get_action(np.array(state))
            # env.node.get_logger().info(f"action : {action}")
            # a_in = [(action[0] + 1) / 2, action[1]]
            # state, reward, done, _ = env.step(a_in)
            # avg_reward += reward
            # count += 1

            iter_reward = 0
            action_n = []
            for i in range(AGENT_COUNT):
                action = network.get_action(np.array(state_n[i]))
                env.node.get_logger().info(f"action : {action}")
                action_n.append(action)

            a_in_n = []
            for i, action in enumerate(action_n):
                a_in = [(action[0] + 1) / 2, action[1]]
                a_in_n.append(a_in)

            state_n, reward_n, done_n, _ = env.step(a_in_n)
            for r in reward_n:
                iter_reward += r
                avg_reward += r
            count += 1

            # if iter_reward < -90 * AGENT_COUNT:
            if iter_reward < -90:
                col += 1

    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    env.node.get_logger().info("..............................................")
    env.node.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    env.node.get_logger().info("..............................................")
    return avg_reward


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

if __name__ == "__main__":
    try:
        rclpy.init()

        seed = 0
        file_name = "td3_policy"
        save_model = True
        load_model = True

        # Directory setup
        if not os.path.exists("./results"):
            os.makedirs("./results")
        if save_model and not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")

        # Initialize environment and network
        environment_dim = LIDAR_SAMPLE_SIZE
        robot_dim = 4
        torch.manual_seed(seed)
        np.random.seed(seed)
        state_dim = environment_dim + robot_dim
        action_dim = 2
        max_action = 1

        network = TD3(state_dim, action_dim, max_action)
        replay_buffer = ReplayBuffer(_BUFFER_SIZE, seed)
        
        if load_model:
            try:
                print("Will load existing model.")
                network.load(file_name, "./pytorch_models")
            except:
                print("Could not load the stored model parameters, initializing training with random parameters")

        evaluations = []
        timestep = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True
        epoch = 1
        episode_timesteps = 0
        episode_reward = 0
        count_rand_actions = 0
        random_action = []
        goal_position = (0.43, 1.58)

        # Initialize the environment and subscribers
        namespaces = [f"robot_{i+1}" for i in range(AGENT_COUNT)]
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

        env = GazeboEnvMultiAgent(odom_subscribers=odom_subscribers, scan_subscribers=scan_subscribers, goal_position=goal_position)
        executor.add_node(env.node)

        executor_thread = threading.Thread(target=executor.spin, daemon=False)
        executor_thread.start()

        prev_observation_n = env.reset()
        just_reset = True

        print('Starting iterations...')

        while timestep < _MAX_TIMESTEPS:
            if done:
                env.node.get_logger().info(f"Done. Episode num: {episode_num} - Total episode rewards: {episode_reward}")
                if timestep != 0:
                    env.node.get_logger().info(f"Training network")
                    network.train(replay_buffer, episode_timesteps, _BATCH_SIZE, _DISCOUNT, _TAU, _POLICY_NOISE, _NOISE_CLIP, _POLICY_FREQ)
                    
                    if timesteps_since_eval >= _EVAL_FREQ:
                        env.node.get_logger().info("Validating")
                        timesteps_since_eval %= _EVAL_FREQ
                        evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=_EVAL_EP))
                        
                        if save_model:
                            network.save(file_name, directory="./pytorch_models")
                            np.save("./results/%s" % (file_name), evaluations)
                        epoch += 1

                prev_observation_n = env.reset()
                done = False
                just_reset = True
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Exploration noise update
            if _EXPL_NOISE > _EXPL_MIN:
                _EXPL_NOISE -= (1 - _EXPL_MIN) / _EXPL_DECAY_STEPS

            # Get actions and step the environment
            action_n = []
            for i in range(AGENT_COUNT):
                action = network.get_action(np.array(prev_observation_n[i]))
                action = (action + np.random.normal(0, _EXPL_NOISE, size=action_dim)).clip(-max_action, max_action)
                action_n.append(action)

            a_in_n = []
            for i, action in enumerate(action_n):
                a_in = [(action[0] + 1) / 2, action[1]]
                a_in_n.append(a_in)

            observation, reward, done_n, info = env.step(a_in_n)
            if just_reset:
                reward = [0.0 for _ in range(AGENT_COUNT)]
                just_reset = False

            done_bool = 0 if episode_timesteps + 1 == _MAX_EP else int(any(done_n))
            done = 1 if episode_timesteps + 1 == _MAX_EP else int(any(done_n))

            for i, r in enumerate(reward):
                episode_reward += r

            for i in range(AGENT_COUNT):
                replay_buffer.add(prev_observation_n[i], action_n[i], reward[i], done_bool, observation[i])

            prev_observation_n = observation
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1

    except KeyboardInterrupt:
        print("Training interrupted manually.")
    
    finally:
        # Plot performance after training ends
        plot_performance(evaluations)
        rclpy.shutdown()