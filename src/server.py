#!/usr/bin/env python3
"""
DQN Training HTTP Server
Provides REST API and SSE for managing and monitoring DQN training runs
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import gymnasium as gym
import ale_py
from collections import namedtuple, deque
import random
import threading
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import queue

# Import your existing modules
from coach import Coach
from dqn import Agent, preprocess
from run import Run
from episode import Episode

app = Flask(__name__)
CORS(app)

# Global state
current_training = None
training_lock = threading.Lock()
training_thread = None
sse_queues = []  # List of queues for SSE clients

@dataclass
class ServerState:
    """Server state containing all runs and their data"""
    runs: List[Dict] = None
    current_run_id: Optional[str] = None
    is_training: bool = False
    
    def __post_init__(self):
        if self.runs is None:
            self.runs = []

server_state = ServerState()

class TrainingSession:
    """Manages a single training session"""
    
    def __init__(self, environment: str, params: Dict = None):
        self.environment = environment
        self.params = params or self._default_params()
        self.should_stop = False
        self.run = None
        self.agent = None
        self.target_agent = None
        self.optimizer = None
        self.env = None
        self.coach = None
        self.update_steps = 0
        self.run_id = f"run_{int(time.time())}"
        
    def _default_params(self):
        """Default hyperparameters"""
        return {
            "N_PREV_OBVS": 1,
            "UPDATE_INTERVAL": 100,
            "GAMMA": 0.99,
            "BATCH_SIZE": 64,
            "LEARNING_RATE": 0.001,
            "MEMORY_CAP": 50000,
            "MEMORY_MIN": 1000,
            "LAYERS": 2,
            "HIDDEN": 128,
            "LEARN_RATE": 4,
            "LEARN_ITERATIONS": 1,
            "FRAMESKIP": 0,
            "PRINT_INTERVAL": 10000
        }
    
    def setup(self):
        """Initialize environment and agent"""
        gym.register_envs(ale_py)

        self.env = gym.make(self.environment, self.params.env_kwargs)
        self.env._max_episode_steps = 5000
        
        # Update params with env info
        self.params["IN_FEATURES"] = self.env.observation_space.shape[-1] * self.params["N_PREV_OBVS"]
        self.params["ACTIONS"] = self.env.action_space.n
        
        # Create agents
        self.agent = Agent(
            in_size=self.params["IN_FEATURES"],
            hidden=self.params["HIDDEN"],
            layers=self.params["LAYERS"],
            out=self.params["ACTIONS"]
        ).cuda()
        
        self.target_agent = Agent(
            in_size=self.params["IN_FEATURES"],
            hidden=self.params["HIDDEN"],
            layers=self.params["LAYERS"],
            out=self.params["ACTIONS"]
        ).cuda()
        
        self.agent.apply(self.agent.init_weights)
        self.target_agent.apply(self.target_agent.init_weights)
        
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.params["LEARNING_RATE"])
        
        # Setup coach
        transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.coach = Coach(
            reward_shaping=lambda r, d: r,
            transition=transition,
            render=False,
            N_PREV_OBVS=self.params["N_PREV_OBVS"]
        )
        
        # Create run
        self.run = Run(self.params)
        self.run.agent = self.agent
        self.run.params = self.params
        self.run.env = self.env
        
        # Try to load existing agent
        self._load_agent()
    
    def _load_agent(self):
        """Load agent from disk if available"""
        try:
            filename = f"./agents/{self.environment}"
            self.agent.load_state_dict(torch.load(filename))
            self.target_agent.load_state_dict(self.agent.state_dict())
            print(f"Loaded agent from {filename}")
        except Exception as e:
            print(f"No saved agent found: {e}")
    
    def _save_agent(self):
        """Save agent to disk"""
        try:
            filename = f"./agents/{self.environment}"
            torch.save(self.agent.state_dict(), filename)
            print(f"Saved agent to {filename}")
        except Exception as e:
            print(f"Failed to save agent: {e}")
    
    def learn(self):
        """Perform one learning step"""
        if len(self.run.memory) < self.params["BATCH_SIZE"]:
            return
        
        batch = random.sample(self.run.memory, self.params["BATCH_SIZE"])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).cuda()
        next_states = torch.stack(next_states).cuda()
        actions = torch.tensor(actions).cuda()
        rewards = torch.tensor(rewards).cuda()
        dones = torch.tensor(dones).cuda()
        
        q_values = self.agent.act(states)
        q_values = torch.gather(q_values, dim=1, index=actions.unsqueeze(1))
        
        next_q_values = self.target_agent.act(next_states).detach()
        next_q_values = next_q_values.max(1)[0]
        next_q_values[dones] = 0.0
        
        target = rewards + self.params["GAMMA"] * next_q_values
        target = target.unsqueeze(1)
        
        loss = torch.nn.functional.smooth_l1_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)
        self.optimizer.step()
    
    def train_loop(self):
        """Main training loop"""
        try:
            self.setup()
            
            while not self.should_stop:
                episode = Episode(self.run)
                episode.test = False
                
                score, steps = self.coach.run_episode(
                    episode=episode,
                    preprocess=preprocess,
                    learn=lambda: self.learn()
                )
                
                self.run.total_steps += steps
                self.update_steps += steps
                new_high_score = False
                
                if score > self.run.highest_score:
                    self.run.highest_score = score
                    new_high_score = True
                    self._save_agent()
                
                # Update target network
                if self.update_steps > self.params["UPDATE_INTERVAL"]:
                    self.target_agent.load_state_dict(self.agent.state_dict())
                    self.update_steps = 0
                
                self.run.episode += 1
                
                # Send update to all SSE clients
                update_data = {
                    "type": "episode_complete",
                    "run_id": self.run_id,
                    "episode": self.run.episode,
                    "score": score,
                    "steps": steps,
                    "total_steps": self.run.total_steps,
                    "highest_score": self.run.highest_score,
                    "new_high_score": new_high_score,
                    "memory_size": len(self.run.memory)
                }
                broadcast_update(update_data)
            
            self.env.close()
            broadcast_update({"type": "training_stopped", "run_id": self.run_id})
        except Exception as e:
            import traceback
            error_data = {
                "type": "error",
                "run_id": self.run_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            broadcast_update(error_data)
            print(f"Training error: {e}")
            traceback.print_exc()
            if self.env:
                self.env.close()
    
    def get_state(self) -> Dict:
        """Get current state as dictionary"""
        if not self.run:
            return {}
        
        return {
            "run_id": self.run_id,
            "environment": self.environment,
            "episode": self.run.episode,
            "total_steps": self.run.total_steps,
            "highest_score": self.run.highest_score,
            "memory_size": len(self.run.memory),
            "params": self.params
        }

def broadcast_update(data: Dict):
    """Send update to all connected SSE clients"""
    message = f"data: {json.dumps(data)}\n\n"
    for q in sse_queues:
        try:
            q.put(message)
        except:
            pass

# ===================== API ENDPOINTS =====================

@app.route('/run', methods=['POST'])
def start_run():
    """Start a new training run"""
    global current_training, training_thread, server_state
    
    with training_lock:
        if current_training and not current_training.should_stop:
            return jsonify({"error": "Training already in progress"}), 400
        
        data = request.json
        environment = data.get('environment')
        params = data.get('params', None)
        
        if not environment:
            return jsonify({"error": "environment parameter required"}), 400
        
        try:
            current_training = TrainingSession(environment, params)
            server_state.is_training = True
            server_state.current_run_id = current_training.run_id
            
            # Start training in background thread
            training_thread = threading.Thread(target=current_training.train_loop, daemon=True)
            training_thread.start()
            
            return jsonify({
                "status": "started",
                "run_id": current_training.run_id,
                "environment": environment
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_run():
    """Stop the current training run"""
    global current_training, server_state
    
    with training_lock:
        if not current_training:
            return jsonify({"error": "No training in progress"}), 400
        
        current_training.should_stop = True
        server_state.is_training = False
        
        return jsonify({"status": "stopping", "run_id": current_training.run_id})

@app.route('/get', methods=['GET'])
def get_state():
    """Get complete server state"""
    global current_training, server_state
    
    with training_lock:
        current_run_data = None
        if current_training:
            current_run_data = current_training.get_state()
        
        return jsonify({
            "is_training": server_state.is_training,
            "current_run_id": server_state.current_run_id,
            "current_run": current_run_data,
            "runs": server_state.runs
        })

@app.route('/updates')
def stream_updates():
    """Server-Sent Events endpoint for real-time updates"""
    def generate():
        q = queue.Queue()
        sse_queues.append(q)
        
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            
            while True:
                message = q.get()
                yield message
        except GeneratorExit:
            sse_queues.remove(q)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/')
def index():
    """Serve the debug interface"""
    return app.send_static_file('index.html')

# Global error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions and send to client"""
    import traceback
    error_details = {
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc()
    }
    
    # Also broadcast to SSE clients
    broadcast_update({
        "type": "error",
        "error": str(e),
        "error_type": type(e).__name__,
        "traceback": traceback.format_exc()
    })
    
    return jsonify(error_details), 500

if __name__ == '__main__':
    print("Starting DQN Training Server...")
    print("Endpoints:")
    print("  POST /run - Start training (body: {environment: string, params?: object})")
    print("  POST /stop - Stop training")
    print("  GET /get - Get current state")
    print("  GET /updates - SSE stream for real-time updates")
    print("  GET /health - Health check")
    print("\nServer running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, threaded=True)