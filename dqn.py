#!/usr/local/bin/python3
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

import torch
import random
import sys

from multiprocessing import Process, Manager
from matplotlib import pyplot as plt
from collections import namedtuple
from coach import Coach
import argparse

parser = argparse.ArgumentParser(prog="DQN", description="Deep Q Network")

parser.add_argument('environment')
parser.add_argument('-t', '--test')

args = parser.parse_args()

env_name = sys.argv[1]
torch.manual_seed(0)

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("CUDA not available - using CPU")
print("-" * 50)

# # # #  PyTorch DQN Model
class Agent(torch.nn.Module):
    def __init__(self, in_size=4, hidden=64, layers=1, out=2):
        super(Agent, self).__init__()

        modules = []
        modules.append(torch.nn.Linear(in_size, hidden))
        modules.append(torch.nn.Tanh())
        
        for _ in range(layers):
            modules.append(torch.nn.Linear(hidden, hidden))
            modules.append(torch.nn.Tanh()
        )
        
        modules.append(torch.nn.Linear(hidden, out))
        
        self.actor = torch.nn.Sequential(
            *modules
        )

    def act(self, state):
        action = self.actor(state.float())
        return action

def preprocess(state):
    # state extraction for Pong RAM
    # state = state[[0x31, 0x36, 0x38, 0x3A, 0x3C]]
    #              ball x  ball y bvx  bvxy  paddle y
    return torch.tensor(state).float().cuda()/255

# define an update function for learning
def learn():
    state, action, reward, next_state, _ = zip(*random.sample(memory, BATCH_SIZE))

    state = torch.stack(state)
    next_state = torch.stack(next_state)
    action = torch.tensor(action).cuda()
    reward = torch.tensor(reward).cuda()

    q_values = agent.act(state)
    q_values = torch.gather(q_values, index=action.unsqueeze(1), dim=1)

    next_q_values = target_agent.act(next_state).detach()
    next_q_values = next_q_values.max(1)[0]

    target = reward + GAMMA * next_q_values
    target = target.unsqueeze(1)

    loss = (target - q_values).pow(2).sum().div(2)

    optimizer.zero_grad()
    loss.backward()

    for p in agent.parameters():
        p.grad.data.clamp_(-1.0, 1.0)

    optimizer.step()

# loading/saving checkpoint for testing
def load_agent():
    file = "./agents/" + env_name
    print("Loading ",file)
    try:
        agent.load_state_dict(torch.load(file))
        target_agent.load_state_dict(agent.state_dict())
    except:
        print("Failed to load " + file)

def save_model():
    print(" ~!  ---- Saving model ---- !~")
    file = "./agents/" + env_name
    torch.save(agent.state_dict(), file)

if args.test:
    env = gym.make(env_name, render_mode="human", obs_type="ram")
    env._max_episode_steps = 99999
else:
    env = gym.make(env_name, obs_type="ram")

# define hyperparameters
step = 0
highest = -9999
lives = 0
init_frameskip = 0
epoch = 0
episode = 1
init_action = 1
GAMMA = 0.99
BATCH_SIZE = 128
UPDATE_INTERVAL = 1000
rate = 0.00025
K = 1
total_steps = 0

print(env.observation_space.shape)

in_features = env.observation_space.shape[-1]
hidden = 256
hidden_layers = 4
actions = env.action_space.n
max_episode_steps = 1000

if args.test:
    max_episode_steps = 999999

env._max_episode_steps = max_episode_steps

EPSILON_START = 1.0
EPSILON_MIN = 0.00
EPSILON_STEPS = 50000

memory = []


# create objects
transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def reward_shaping(reward, done):
    # Penalize if game hasn't started after N steps
    if coach.step < 30 and reward == 0:
        return -0.01  # Small penalty for not starting
    return reward

agent = Agent(in_size=in_features, hidden=hidden, layers=hidden_layers, out=actions).cuda()
target_agent = Agent(in_size=in_features, hidden=hidden, layers=hidden_layers, out=actions).cuda()

optimizer = torch.optim.RMSprop(params=agent.parameters(), lr=rate)

load_agent()

coach = Coach(reward_shaping=reward_shaping, transition=transition)
print(agent)

# def plot(total_score):
    # x, y = zip(*running_scores)
    # plt.plot(x, y)
    # plt.draw()
    # plt.pause(0.0000001)

workers = []

# main loop
while True:
    if args.test:
        load_agent()

    # decrement epsilon value
    epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * torch.exp(
        torch.tensor(-1. * episode / EPSILON_STEPS))

    epsilon = max(epsilon, EPSILON_MIN)
    memory, score, steps = coach.run_episode(agent, env, memory, episode, preprocess, epsilon, args.test, learn)
    episode += 1
    total_steps += steps

    if score > highest:
        highest = score

    print("Episode: {} Score: {} Highest: {} Steps: {}".format(episode, score, highest, total_steps))

    if not args.test:
        if episode % UPDATE_INTERVAL == 0:
            print("Update target agent")
            target_agent.load_state_dict(agent.state_dict())
            save_model()

env.close()
