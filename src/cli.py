#!/usr/bin/env python3
import sys
import argparse
import torch
import gymnasium as gym
import ale_py
from collections import namedtuple
from collections import deque
from coach import Coach  # make sure coach.py is importable
from dqn import Agent, preprocess  # your existing Agent class
import random
from visualize import visualize_model
from run import Run
from episode import Episode

# --------------------- ARGUMENT PARSER ---------------------
parser = argparse.ArgumentParser(prog="DQN Main", description="Run DQN Agent")
parser.add_argument("environment", help="Gym environment name (ALE)")
parser.add_argument("-t", "--test", action="store_true", help="Run in test mode (no training)")
args = parser.parse_args()

# --------------------- ENVIRONMENT SETUP ---------------------
gym.register_envs(ale_py)
env_name = args.environment
#env_kwargs = {"obs_type": "ram"}
env_kwargs = {}
if args.test:
    env_kwargs["render_mode"] = "rgb_array"

env = gym.make(env_name, **env_kwargs)
env._max_episode_steps = 99999 if args.test else 5000

params = {}
update_steps = 0

# --------------------- HYPERPARAMETERS ---------------------
params["N_PREV_OBVS"] = 1           # No stacking needed!

# model architecture
params["IN_FEATURES"] = env.observation_space.shape[-1] * params["N_PREV_OBVS"]
params["ACTIONS"] = env.action_space.n
params["UPDATE_INTERVAL"] = 100
# Minimal, proven hyperparameters
params["GAMMA"] = 0.99              # Higher for CartPole
params["BATCH_SIZE"] = 64           # Much smaller
params["LEARNING_RATE"] = 0.001     # Can be higher
params["MEMORY_CAP"] = 50000        # Smaller buffer
params["MEMORY_MIN"] = 1000         # Start learning sooner
params["LAYERS"] = 2                # Simpler network
params["HIDDEN"] = 128              # Much smaller
params["LEARN_RATE"] = 4            # Update every step
params["LEARN_ITERATIONS"] = 1      # Keep at 1
params["RENDER_SCALE"] = 1
params["FRAMESKIP"] = 0
params["PRINT_INTERVAL"] = 10000

# --------------------- AGENT SETUP ---------------------
agent = Agent(in_size=params["IN_FEATURES"], hidden=params["HIDDEN"], layers=params["LAYERS"], out=params["ACTIONS"]).cuda()
target_agent = Agent(in_size=params["IN_FEATURES"], hidden=params["HIDDEN"], layers=params["LAYERS"], out=params["ACTIONS"]).cuda()
optimizer = torch.optim.Adam(agent.parameters(), lr=params["LEARNING_RATE"])

agent.apply(agent.init_weights)
target_agent.apply(target_agent.init_weights)

transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Optional: Load checkpoint if exists
def load_agent(filename="./agents/"):
    try:
        agent.load_state_dict(torch.load(filename + env_name))
        target_agent.load_state_dict(agent.state_dict())
        print(f"Loaded agent from {filename + env_name}")
    except Exception as e:
        print(f"No saved agent found: {e}")

def save_agent(filename="./agents/"):
    torch.save(agent.state_dict(), filename + env_name)
    print(f"Saved agent to {filename + env_name}")

# --------------------- REWARD SHAPING ---------------------
def reward_shaping(reward, done):
    return reward


coach = Coach(reward_shaping=reward_shaping, transition=transition, render=args.test, N_PREV_OBVS=params["N_PREV_OBVS"])
load_agent()

# --------------------- TRAINING / TEST LOOP ---------------------


# ----------------- Learning -----------------
def learn(agent, target_agent, run, batch_size=128, gamma=0.99, optimizer=None):
    """Sample batch from memory and perform DQN update"""

    if len(run.memory) < batch_size:
        return

    batch = random.sample(run.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states).cuda()
    next_states = torch.stack(next_states).cuda()
    actions = torch.tensor(actions).cuda()
    rewards = torch.tensor(rewards).cuda()
    dones = torch.tensor(dones).cuda()
    
    #agent.reset()
    q_values = agent.act(states)
    q_values = torch.gather(q_values, dim=1, index=actions.unsqueeze(1))

    next_q_values = target_agent.act(next_states).detach()
    next_q_values = next_q_values.max(1)[0]
    next_q_values[dones] = 0.0

    target = rewards + gamma * next_q_values
    target = target.unsqueeze(1)

    #loss = (q_values - target).pow(2).mean()
    loss = torch.nn.functional.smooth_l1_loss(q_values, target)
    #print(loss)

    optimizer.zero_grad()
    loss.backward()

    # Optional gradient clipping
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)

    optimizer.step()


current_run = Run(params)
current_run.agent = agent
current_run.params = params
current_run.env = env

while True:

    episode = Episode(current_run)
    episode.test = args.test

    # Pass learn as a proper function, not a lambda with no args
    score, steps = coach.run_episode(
        episode=episode,
        preprocess=preprocess,
        learn=lambda a=agent, t=target_agent, r=current_run: learn(a, t, r, optimizer=optimizer)
    )

    current_run.total_steps += steps
    update_steps += steps
    new_high_score = False

    if score > current_run.highest_score:
        current_run.highest_score = score
        new_high_score = True
        print(f"Episode: {current_run.episode} | Steps: {steps} | Score: {score} | Highest: {current_run.highest_score} | Total Steps: {current_run.total_steps}")

    if current_run.total_steps % params["PRINT_INTERVAL"] == 0:
        print(f"Episode: {current_run.episode} | Steps: {steps} | Score: {score} | Highest: {current_run.highest_score} | Total Steps: {current_run.total_steps}")

    if args.test:
        load_agent()
    else:
        if new_high_score:
            save_agent()
        if update_steps > params["UPDATE_INTERVAL"]:
            #print("Update!")
            target_agent.load_state_dict(agent.state_dict())
            update_steps = 0

    current_run.episode += 1

env.close()
