import torch
import torchvision
import gym
import gym_ple
import numpy
import sys
import math
import random
import cv2
import skimage.measure
import copy
import pygame
from skimage.transform import resize
from collections import namedtuple
from optparse import OptionParser

pygame.init()

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'reward', 'qvalues', 'paction', 'action'))


def preprocess_frame(state):
    state = (numpy.sum(state, axis=2) / 3.0) / 255.0
    state = ((state > 0.3) * 1.0)
    state = cv2.resize(state, (64, 64))
    return torch.tensor(state).unsqueeze(0).unsqueeze(0).float()

env = gym.make('SpaceInvaders-v4')

init_episode_frameskip = 90
episode_steps = 1000
env._max_episode_steps = episode_steps

state = env.reset()

for i in range(init_episode_frameskip):
    state, _, _, _ = env.step(0)

p_states = 1

def render_env(state):
    state = numpy.transpose(numpy.array(state), (1, 0))
    state = cv2.resize(state, (10 * width, 10 * height), interpolation=cv2.INTER_AREA)
    cv2.imshow("state", state)
    cv2.waitKey(1)

# env.render()
BATCH_SIZE = 256
REPLAY_MEMORY_SIZE = 5000

rewards = []
actions = []
next_states = []
current_q_values = []

score = 0
highest = 1
highest_steps = 1
sum_loss = 0.0
update_counter = 0
hidden = 512
n_layers = 1
frameskip = 1
rate = 0.00001
target_agent_update_interval = 10
gamma = 0.9
n_states = 4
n_actions = env.action_space.n
lives = 1
action = 0
in_features = 4

render = False
first = True

replay_memory = []

def one_hot(x):
    a = torch.zeros(n_actions)
    a[x] = 1.0
    return a

class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 5, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, hidden, 5, 2),
            torch.nn.ReLU(),
        )
        self.output = torch.nn.Linear(hidden, n_actions)

    def forward(self, x, train):
        b = x.shape[0]
        x = self.net(x)
        return self.output(x.view(b, -1))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.0)
        torch.nn.fill(m.weight.bias, 0.0)
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.5)
        torch.nn.fill(m.weight.bias, 0.0)


agent = Agent().float()

weights_init(agent)
target_agent = copy.copy(agent)

optimizer = torch.optim.Adam(lr=rate, params=agent.parameters())

# try to load saved agent, if not found start new
try:
    checkpoint = torch.load("./agents/{}".format(env_name))
    agent.load_state_dict(checkpoint['model_state_dict'])
    train = False
    greedy = True
except:
    print("New agent.")


def discount(rewards):
    R = 0.0
    discount_rewards = []

    for r in reversed(rewards):
        R = r + (R * gamma)  # sum of discounted rewards
        discount_rewards.insert(0, R)

    discount_rewards = torch.tensor(discount_rewards)
    discount_rewards_mean, discount_rewards_std = discount_rewards.mean(), discount_rewards.std()
    discount_rewards = (discount_rewards - discount_rewards_mean) / (discount_rewards_std+1e-10)

    return discount_rewards


def learn(gamma):
    if len(replay_memory) < REPLAY_MEMORY_SIZE:
        return 0.0

    sys.stdout.flush()
    batch = random.sample(replay_memory, BATCH_SIZE)

    state_batch = torch.stack([b.state for b in batch])
    next_states_batch = torch.stack([b.next_state for b in batch]).squeeze(1)
    c_q = torch.stack([b.qvalues for b in batch])
    reward_batch = torch.tensor([b.reward for b in batch])
    paction = torch.tensor([b.paction for b in batch])
    l = 0.0

    target_agent.hidden = torch.zeros(n_layers, len(state_batch), hidden)
    next_state_values = target_agent.forward(next_states_batch.float(), True)
    next_state_values = next_state_values.squeeze(1).max(dim=1)[0]
    state_values = reward_batch.unsqueeze(0).float() + (next_state_values * gamma)

    c_q = c_q.squeeze(1).squeeze(1)
    loss = torch.nn.MSELoss()(c_q[:, paction], state_values)
    l += loss.item()

    print("Train")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    replay_memory.clear()

    del loss, batch, state_batch, reward_batch, state_values, next_states_batch, c_q, next_state_values
    return l


total_reward = 0.0

def savemodel():
    a = input("Save? <Y/N>")
    if a == "Y" or a == "y":
        print("Saving agent parameters...")
        modelname = "./agents/{}".format(env_name)

        torch.save({
            'model_state_dict': agent.state_dict(),
        }, modelname)
    env.close()

anim = ["|", "\\", "-", "/"]

def push_replay_memory(*args):
    replay_memory.append(Transition(*args))
    if len(replay_memory) > REPLAY_MEMORY_SIZE:
        del replay_memory[0]

train = True
maxsteps = 1
total_steps = 0
psteps = 0
epoch = 0
q = torch.zeros(n_actions)
new_q = q
n_epochs = 5
games = 0
steps = 0
reward = 0.0
anxiety = 1.0
episode_games = 0
old_state = state
n_reward_steps = 0
lines_cleared = 0
action = 0
paction = 0
confidence = 0.0
clock = torch.zeros(1, 1)
state = preprocess_frame(state)
states = [state for _ in range(n_states)]
episode = []
attitude = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN:
            # gets the key name
            key_name = pygame.key.name(event.key)
            if key_name == 'r':
                render = not render
                print("Render {}".format(render))
    if render:
        # render_env(state)
        env.render(mode='human')
        if steps % 10 == 0:
            print("Confidence:{}".format(confidence))

    q = new_q

    anxiety = torch.nn.Sigmoid()(torch.randn(1))

    inp = torch.cat(states, dim=1)
    new_q = agent.forward(inp, False).view(-1)

    if anxiety > confidence:
        action = env.action_space.sample()
    else:
        action = torch.argmax(new_q, dim=-1)

    steps += 1
    total_steps += 1
    update_counter += 1

    if first:
        action = 0
        first = False

    new_reward = 0.0
    old_state = inp
    old_q = new_q

    for i in range(frameskip):
        new_state, r, done, info = env.step(int(action))
        if r > 0:
            r = 1
        score += r
        new_reward += r
        if done:
            break

    confidence *= 0.99
    confidence += reward

    push_replay_memory(inp, old_state, reward, old_q.view(-1), int(paction), int(action))

    state = preprocess_frame(new_state)
    states.append(state)
    if len(states) > n_states:
        del states[0]

    state = new_state
    paction = action
    reward = new_reward

    if score > highest:
        highest = score
    if steps > maxsteps:
        maxsteps = steps

    confidence *= 0.9
    confidence += new_reward
    attitude.append(confidence)

    if done:
        confidence = 0.0
        print(
            "|Game/Epoch {}/{} | Steps ({} - {}) | Score {} | Highest: {} | Attitude:{}".format(
                games, epoch, steps, total_steps,
                score, highest, torch.prod(torch.tensor(attitude),0)))
        attitude = []

        # for p in agent.parameters():
        #    print(p.grad)

        if len(replay_memory) >= BATCH_SIZE:
            sum_loss = learn(gamma)
            epoch += 1
            replay_memory = []

            if epoch % target_agent_update_interval == 0:
                target_agent.load_state_dict(agent.state_dict())
                print("\nTarget net update.\n")

        score = 0

        psteps = steps
        n_reward_steps = 0

        games += 1
        episode_games += 1

        agent.hidden = torch.zeros(n_layers, 1, hidden)
        state = env.reset()

        for i in range(init_episode_frameskip):
            state, _, _, _ = env.step(0)

        action = torch.tensor([0]).float()
        p_action = torch.tensor([0]).float()

        steps = 0
        reward = 0.0
        sum_loss = 0.0
        update_counter = 0
        total_reward = 0.0
