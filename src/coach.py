import torch
import random
import gymnasium as gym
import numpy as np
import pygame

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

import matplotlib.pyplot as plt
import torch.nn as nn
from run import Run

fig, ax = plt.subplots(figsize=(10, 6))

class Coach:
    def __init__(self, reward_shaping=None, transition=None, render=False, N_PREV_OBVS=1, render_skip=1):
        super(Coach, self).__init__()

        # Training parameters
        self.reward_shaping = reward_shaping
        self.total_step = 0
        max_step = 0
        step = 0
        self.transition = transition

        self.highest_eps = 0.0
        self.lowest_eps = 1.0

        # Render settings
        self.render = render
        self.render_skip = render_skip
        self.render_counter = 0
        self.screen = None

        self.temp_start = 2.0
        self.temp_end = 0.1
        self.temp_decay_steps = 1_000_000

        self.N_PREV_OBVS = N_PREV_OBVS

        self.states = []

        if self.render:
            pygame.init()

        #if self.test:
        #    # Create animation (updates every 200ms)
        #    ani = animation.FuncAnimation(fig, update, interval=200, cache_frame_data=False)

        #    plt.tight_layout()
        #    plt.show()

    def get_epsilon(self):
        epsilon = self.EPSILON_START + (self.EPSILON_MAX - self.EPSILON_MIN) * torch.exp(
            torch.tensor(-1.0 * self.total_step / (self.EPSILON_STEPS+1))
        )
        epsilon = max(epsilon, self.EPSILON_MIN)
        return epsilon.item()

    # calculate epsilon value based on "confidence" of q-values
    def get_confidence(self, y, step=None, total_steps=1_000_000):
        """
        Calculate epsilon based on Q-value confidence with optional hybrid decay.
        
        Args:
            y: Q-values tensor for current state
            step: Current training step (optional, for hybrid mode)
            total_steps: Total steps for epsilon decay (default 1M)
        
        Returns:
            epsilon value between 0 and 1
        """
        # Calculate variance normalized by mean
        y_centered = y - y.mean()
        var_norm = torch.var(y_centered) / (y_centered.abs().mean() + 1e-8)
        
        # CRITICAL FIX: Clamp var_norm to [0, 1] range
        var_norm = torch.clamp(var_norm, 0.0, 1.0)
        
        # Confidence-based epsilon
        epsilon_max = 1.0
        alpha = 2.0  # exponent to make drop sharper
        confidence_eps = epsilon_max * (1 - var_norm) ** alpha
        
        # Option 1: Pure confidence-based (use if model already partially trained)
        # return confidence_eps.item()
        
        # Option 2: Hybrid with standard decay (RECOMMENDED for cold start)
        if step is not None:
            # Standard epsilon decay: 1.0 -> 0.1 over total_steps
            base_eps = max(0.1, 1.0 - (step / total_steps) * 0.9)
            # Use minimum of both (more willing to exploit)
            eps = min(base_eps, confidence_eps.item())
            return eps
        else:
            return confidence_eps.item()

    def get_reverse_epsilon(self):
        """
        Epsilon starts near 0 and ramps quickly to 1 as step approaches max_step.
        """
        # Safety to avoid division by zero
        max_step = max(max_step, 1)
        
        # Compute a sigmoid-like ramp
        x = torch.tensor((step - 0.9 * max_step) / (0.1 * max_step))
        epsilon = 1 / (1 + torch.exp(-x))
        
        # Clamp to [EPSILON_MIN, EPSILON_MAX]
        epsilon = torch.clamp(epsilon, self.EPSILON_MIN, self.EPSILON_MAX)
        
        return epsilon.item()


    def get_curiosity(self, q_values):
        """
        Returns non-linear curiosity epsilon in [0,1] based on Q-value variance.
        High epsilon when Q-values are uniform (uncertain),
        drops quickly to near 0 when variance increases (confident).
        """
        with torch.no_grad():
            q_values = q_values.squeeze(0)  # remove batch dim if needed
            q_var = torch.var(q_values).item()

            # Update running min/max variance
            self.q_var_min = min(self.q_var_min, q_var)
            self.q_var_max = max(self.q_var_max, q_var)

            # Avoid divide by zero
            var_range = self.q_var_max - self.q_var_min + 1e-6

            # Scale into [0,1] relative to min/max variance
            normalized = (q_var - self.q_var_min) / var_range  # 0=low, 1=high variance

            # Invert so low variance → high curiosity
            inverted = 1.0 - normalized

            # Apply non-linear mapping (exponential or sigmoid)
            # This makes epsilon drop quickly when confidence rises
            non_linear = inverted ** 3  # cube, biased towards 0 quickly
            # or: non_linear = 1 / (1 + np.exp(10*(normalized-0.1)))  # sigmoid alternative

            # Scale by curiosity scale
            epsilon = self.curiosity_scale * non_linear

            # Clamp to [EPSILON_MIN, EPSILON_MAX]
            epsilon = np.clip(epsilon, self.EPSILON_MIN, self.EPSILON_MAX)

        return float(1-epsilon)

    def select_action_boltzmann(self, y, temperature=1.0):
        """        Sample action from softmax distribution over Q-values.
        
        Args:
            temperature: Controls exploration
        
        Returns:
            action: Selected action index
            confidence: Probability of selected action (0 to 1)
        """

        # Convert Q-values to probabilities
        probs = torch.softmax(y / temperature, dim=-1)
        
        # Sample action from distribution
        action = torch.multinomial(probs, 1).item()
        
        # Get the probability of the selected action
        confidence = probs[action].item()
        
        return action, confidence
    
    def epsilon_sample(self, y, env):
        r = torch.rand(1)
        if(r > y): 
            return np.argmax(y)
        else:
            return env.action_space.sample()


    # Pass learn as a proper function, not a lambda with no args
    def run_episode(self, episode, preprocess, learn):

        action_meanings = []#env.unwrapped.get_action_meanings()
        
        done = False
        score = 0.0
        step = 0
        
        # Reset environment
        obs, info = episode.run.env.reset()
        obs = preprocess(obs)
        states = [obs for _ in range(episode.run.params["N_PREV_OBVS"])]

        # Initial frameskip to skip idle frames at start
        for _ in range(episode.run.params["FRAMESKIP"]):
            _, _, _, _, info = episode.run.env.step(episode.run.params["FRAMESKIP_ACTION"])

        lives = info.get("ale.lives", 1)

        # Initialize Pygame screen once
        if self.render and self.screen is None:
            frame = episode.run.env.render()
            h, w, _ = frame.shape
            self.screen = pygame.display.set_mode((w*episode.run.params["RENDER_SCALE"], h*episode.run.params["RENDER_SCALE"]))

        self.max_eps = 0.0
        self.min_eps = 1.0


        # Episode loop
        while not done:

            # Compute current epsilon
            #current_epsilon = 0.01 if test else self.get_epsilon()
         
            # Action selection
            y = episode.run.agent.act(torch.cat(states))

            #if test:
            #    confidence = 0.99
            #    action = self.epsilon_sample(confidence, env)
            #else:
           
            temperature = max(self.temp_end, 
                self.temp_start - (self.total_step / self.temp_decay_steps) * (self.temp_start - self.temp_end))

            action, confidence = self.select_action_boltzmann(y, temperature)

            if episode.test:
                z = torch.tensor(y)
                z = z.detach().cpu().numpy()
                print("Q-Values: " + str(z))
                print("Action Sampled: " + "(" + str(action) + ")")

            s = "Step " + str(step) + " Max: " + str(episode.run.max_steps) + \
                " Confidence: " + str(round(confidence,4)) + \
                " Score: " + str(score) + \
                " Hi: " + str(episode.run.highest_score) \
            
            pygame.display.set_caption(s)

            reward = 0
            for _ in range(1+episode.run.params["FRAMESKIP"]):
                obs, r, done, _, info = episode.run.env.step(action)
                reward += r

            obs = preprocess(obs)
            next_states = states
            next_states.append(obs)
            next_states.pop(0)

            # Track rewards
            if reward != 0:
                steps_since_last_reward = 0
            else:
                steps_since_last_reward += 1

            score += reward
            self.total_step += 1
            step += 1

            lost_life = False

            if 'ale.lives' in info and info["ale.lives"] < lives:
                lives = info["ale.lives"]
                done = True  # per-life reset
                lost_life = False
                reward = -1

            # Apply reward shaping
            if self.reward_shaping is not None:
                reward = self.reward_shaping(reward, done)

            # Store transition & learn
            if not episode.test:
                if(done):
                    reward = -1

                episode.run.AddTransition(self.transition(torch.cat(states), action, reward, torch.cat(next_states), not lost_life))
                if len(episode.run.memory) > episode.run.params["MEMORY_MIN"] and self.total_step % episode.run.params["LEARN_RATE"] == 0:
                    for _ in range(episode.run.params["LEARN_ITERATIONS"]):
                        learn()
    
                    #arams["UPDATE_INTERVAL"] += 1

            states = next_states

            # Render every `render_skip` steps
            if self.render and self.render_counter % self.render_skip == 0:
                
                pygame.time.delay(100)  # 50 milliseconds = 0.05 seconds
                frame = episode.run.env.render()  # RGB ndarray
                frame = np.flip(frame, axis=0)  # optional: flip vertically

                # Convert to Pygame surface
                frame_surface = pygame.surfarray.make_surface(frame)
                frame_surface = pygame.transform.rotate(frame_surface, 90) 

                # Scale it by 3x
                scaled_surface = pygame.transform.scale(frame_surface, (frame.shape[1]*episode.run.params["RENDER_SCALE"], frame.shape[0]*episode.run.params["RENDER_SCALE"]))

                # Draw to screen
                self.screen.blit(scaled_surface, (0, 0))
                pygame.display.flip()

            self.render_counter += 1

            # Handle Pygame events to avoid freezing
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True

        #print(f"Episode finished: Score={score:.1f}, Steps={step}, Max steps={max_step}")

        episode.run.agent.reset()

        if step > episode.run.max_steps:
            episode.run.max_steps = step

        return score, step
    
