import threading
import time
import gymnasium as gym
import numpy as np
import torch
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
from OpenGL import GL as gl
from PIL import Image
import ale_py
import random
import queue
import pyimplot  # your ImPlot wrapper

gym.register_envs(ale_py)
from dqn import Agent, preprocess, Coach, transition

import faulthandler
faulthandler.enable()  # automatically print stack trace on crash

class RLGui:
    def __init__(self, agent, target_agent, coach):
        self.agent = agent
        self.target_agent = target_agent
        self.coach = coach

        # Core training
        self.running = False
        self.paused = False
        self.training_thread = None

        # Environment selection
        self.env_name_list = sorted([env_spec.id for env_spec in gym.envs.registry.values()])
        self.env_index = 0
        self.env = None
        self.render_env = False

        # Observation mode
        self.obs_modes = ["ram", "rgb_array"]
        self.obs_mode_index = 0
        self.env_obs_type = self.obs_modes[self.obs_mode_index]

        # Stats & logs
        self.score_history = []
        self.epsilon_history = []
        self.episode_count = 0
        self.logs = []
        self.memory = []

        # Thread-safe queues for GUI
        self.frame_queue = queue.Queue(maxsize=1)
        self.score_queue = queue.Queue()
        self.epsilon_queue = queue.Queue()
        self.log_queue = queue.Queue()

    # ----------------- ENV CREATION -----------------
    def create_env(self):
        obs_type = self.env_obs_type
        render_mode = "rgb_array" if obs_type != "ram" else None
        self.env = gym.make(
            self.env_name_list[self.env_index],
            render_mode=render_mode,
            obs_type=obs_type
        )

        obs_size = int(np.prod(self.env.observation_space.shape))
        actions = self.env.action_space.n
        self.agent = Agent(in_size=obs_size, hidden=256, layers=4, out=actions).cuda()
        self.target_agent = Agent(in_size=obs_size, hidden=256, layers=4, out=actions).cuda()
        self.clear()
        self.add_log(f"Created environment {self.env_name_list[self.env_index]} with obs_mode={obs_type}")

    # ----------------- TRAINING LOOP -----------------
    def training_loop(self):
        if self.env is None:
            self.create_env()

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            # Run one episode
            self.memory, score, steps = self.coach.run_episode(
                agent=self.agent,
                env=self.env,
                memory=self.memory,
                episode=self.episode_count,
                preprocess=preprocess,
                epsilon=0,
                test=False,
                learn=self.learn
            )

            self.episode_count += 1
            self.score_queue.put(score)
            self.epsilon_queue.put(self.coach.get_epsilon())
            self.log_queue.put(f"Episode {self.episode_count} | Score {score:.1f} | Steps {steps} | Epsilon {self.coach.get_epsilon():.4f}")

            # Capture last frame (non-blocking)
            if self.render_env and self.env_obs_type != "ram":
                state, _ = self.env.reset()
                frame = self.env.render(mode="rgb_array")
                try:
                    # Keep only latest frame
                    if not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

    # ----------------- LEARN FUNCTION -----------------
    def learn(self):
        if len(self.memory) < 128:
            return

        batch_size = 128
        transitions = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        state = torch.stack(state).cuda()
        next_state = torch.stack(next_state).cuda()
        action = torch.tensor(action).cuda()
        reward = torch.tensor(reward).cuda()

        q_values = self.agent.act(state)
        q_values = torch.gather(q_values, index=action.unsqueeze(1), dim=1)

        next_q_values = self.target_agent.act(next_state).detach()
        next_q_values = next_q_values.max(1)[0]
        target = reward + 0.99 * next_q_values
        target = target.unsqueeze(1)

        optimizer = torch.optim.RMSprop(self.agent.parameters(), lr=0.00025)
        optimizer.zero_grad()
        loss = (target - q_values).pow(2).mean()
        loss.backward()
        for p in self.agent.parameters():
            p.grad.data.clamp_(-1.0, 1.0)
        optimizer.step()

    # ----------------- LOGGING -----------------
    def add_log(self, text):
        self.logs.append(text)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    # ----------------- CLEAR -----------------
    def clear(self):
        self.score_history = []
        self.epsilon_history = []
        self.episode_count = 0
        self.logs = []
        self.memory = []

    # ----------------- GUI DRAW -----------------
    def draw(self):
        imgui.new_frame()

        # Update from queues
        while not self.score_queue.empty():
            self.score_history.append(self.score_queue.get_nowait())
        while not self.epsilon_queue.empty():
            self.epsilon_history.append(self.epsilon_queue.get_nowait())
        while not self.log_queue.empty():
            self.add_log(self.log_queue.get_nowait())

        # Current status
        imgui.begin("Current Episode Status", True)
        imgui.text(f"Episodes: {self.episode_count}")
        imgui.text(f"Last score: {self.score_history[-1] if self.score_history else 0}")
        imgui.text(f"Epsilon: {self.epsilon_history[-1] if self.epsilon_history else 0:.4f}")
        imgui.end()

        # Controls
        imgui.begin("Controls", True)
        changed_env, self.env_index = imgui.combo("Environment", self.env_index, self.env_name_list)
        if changed_env:
            self.create_env()
        changed_obs, self.obs_mode_index = imgui.combo("Observation Mode", self.obs_mode_index, self.obs_modes)
        if changed_obs:
            self.env_obs_type = self.obs_modes[self.obs_mode_index]
            self.create_env()
        if imgui.button("Start"):
            if not self.running:
                self.running = True
                self.paused = False
                self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
                self.training_thread.start()
            else:
                self.paused = False
        imgui.same_line()
        if imgui.button("Pause"):
            self.paused = True
        imgui.same_line()
        if imgui.button("Stop"):
            self.running = False
        imgui.same_line()
        if imgui.button("Reset"):
            self.clear()
            self.paused = False
        changed, self.render_env = imgui.checkbox("Render Environment", self.render_env)
        imgui.end()

        # Score plot
        imgui.begin("Score Plot", True)
        if self.score_history and pyimplot.begin_plot("Score vs Episode"):
            xs = np.arange(len(self.score_history), dtype=np.float64)
            ys = np.array(self.score_history, dtype=np.float64)
            pyimplot.plot_line("Score", xs, ys)
            pyimplot.end_plot()
        imgui.end()

        # Epsilon plot
        imgui.begin("Epsilon Plot", True)
        if self.epsilon_history and pyimplot.begin_plot("Epsilon vs Episode"):
            xs = np.arange(len(self.epsilon_history), dtype=np.float64)
            ys = np.array(self.epsilon_history, dtype=np.float64)
            pyimplot.plot_line("Epsilon", xs, ys)
            pyimplot.end_plot()
        imgui.end()

        # Logs
        imgui.begin("Logs", True)
        for line in self.logs[-200:]:
            imgui.text(line)
        if self.logs:
            imgui.set_scroll_here_y(1.0)
        imgui.end()

        # Environment render (from queue)
        if self.render_env:
            try:
                frame = self.frame_queue.get_nowait()
                imgui.begin("Environment Render", True)
                img = Image.fromarray(frame).transpose(Image.FLIP_TOP_BOTTOM)
                img_data = np.array(img).astype(np.uint8)
                tex_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img_data.shape[1], img_data.shape[0],
                                0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_data)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                imgui.image(tex_id, img_data.shape[1], img_data.shape[0])
                gl.glDeleteTextures([tex_id])
                imgui.end()
            except queue.Empty:
                pass

        imgui.render()

    # ----------------- RUN -----------------
    def run(self):
        if not glfw.init():
            print("Could not init GLFW")
            return
        window = glfw.create_window(1280, 720, "RL Dashboard", None, None)
        glfw.make_context_current(window)
        imgui.create_context()
        impl = GlfwRenderer(window)

        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.draw()
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)

        impl.shutdown()
        glfw.terminate()


if __name__ == "__main__":
    agent = Agent().cuda()
    target_agent = Agent().cuda()
    coach = Coach(transition=transition)

    ui = RLGui(agent, target_agent, coach)
    ui.run()
