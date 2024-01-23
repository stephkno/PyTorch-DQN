#!/usr/local/bin/python3
import gym
import torch
import random
import sys
from multiprocessing import Process, Manager
from matplotlib import pyplot as plt
from collections import namedtuple
from coach import Coach

test = False
new = False
resume = False

if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
    test = True
elif len(sys.argv) > 1 and sys.argv[1].lower() == "new":
    new = True
elif len(sys.argv) > 1 and sys.argv[1].lower() == "resume":
    resume = True

torch.manual_seed(0)

# # # #  PyTorch DQN Model
class Agent(torch.nn.Module):
    def __init__(self, in_size=4, hidden=64, out=2):
        super(Agent, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out),
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
    print("Loading agent")
    # file = "/run/user/1000/gvfs/sftp:host=6502.local/Users/stephen/Documents/code/pytorch/reinforement_learning/checkpoint.pth"
    file = "./agents/last_checkpoint.pth"
    agent.load_state_dict(torch.load(file))
def save_model():
    print(" ~!  ---- Saving model ---- !~")
    torch.save(agent.state_dict(), './agents/last_checkpoint.pth')

# define training environment
env_name = "Breakout-ram-v4"

if test:
    env = gym.make(env_name, render_mode="human")
    env._max_episode_steps = 99999
else:
    env = gym.make(env_name)

# define hyperparameters
step = 0
highest = -9999
lives = 0
init_frameskip = 0
epoch = 0
episode = 1
init_action = 1
GAMMA = 0.99
BATCH_SIZE = 32
BATCH_MIN = 10000
BUFFER_CAP = 100000
UPDATE_INTERVAL = 50
rate = 0.0001
plot = not test
K = 1
total_steps = 0

in_features = 128
hidden = 256
actions = env.action_space.n
max_episode_steps = 1000

if test:
    max_episode_steps = 999999

env._max_episode_steps = max_episode_steps

EPSILON_START = 0.05
EPSILON_MIN = 0.01
EPSILON_STEPS = 5000

memory = []

# create objects
transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def reward_shaping(reward, done):
    # reward = 0 if not done else -1
    return reward

agent = Agent(in_size=in_features, hidden=hidden, out=actions).cuda()
target_agent = Agent(in_size=in_features, hidden=hidden, out=actions).cuda()
optimizer = torch.optim.RMSprop(params=agent.parameters(), lr=rate)

if resume or test:
    load_agent()

target_agent.load_state_dict(agent.state_dict())

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
    if test:
        load_agent()
    # decrement epsilon value
    epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * torch.exp(
        torch.tensor(-1. * episode / EPSILON_STEPS))

    epsilon = max(epsilon, EPSILON_MIN)
    memory, score, steps = coach.run_episode(agent, env, memory, episode, preprocess, epsilon, test, learn)
    episode += 1
    total_steps += steps

    if score > highest:
        highest = score
        if not test:
            save_model()

    print("Episode{} Score{} Highest{} Steps{}".format(episode, score, highest, total_steps))

    if not test:
        if episode % UPDATE_INTERVAL:
            target_agent.load_state_dict(agent.state_dict())

env.close()
