import os
from collections import deque, namedtuple

import torch
import random
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ScoreKeeper:
    def __init__(self, num_agents=2, target_score=0.5, window_len=100):
        self.NUM_AGENTS = num_agents
        self.TARGET_SCORE = target_score
        self.WINDOW_LEN = window_len

        self.scores = []
        self.scores_window = deque(maxlen=self.WINDOW_LEN)
    
    def reset(self):
        self.curr_score = np.zeros(self.NUM_AGENTS)
    
    def update_timestep(self, rewards):
        self.curr_score += rewards
    
    def update_episode(self, i_episode):
        score = np.max(self.curr_score)
        self.scores.append(score)
        self.scores_window.append(score)

        self.reset()

        return self._check_solved(i_episode)
        
    def _check_solved(self, i_episode):
        print(f'\rEpisode {i_episode}\t Score: {self.scores[-1]:.2f}', end='', flush=True)

        if i_episode >= self.WINDOW_LEN:
            if i_episode % 10 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score (over past 100 episodes): {np.mean(self.scores_window):.2f}')
            if np.mean(self.scores_window) >= self.TARGET_SCORE:
                print(f'Environment solved in {i_episode-self.WINDOW_LEN} episodes!\tAverage Score: {np.mean(self.scores_window):.2f}')
                return True

        return False
    

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=1e+6, seed=42):
        self.BUFFER_SIZE = int(buffer_size)

        self.memory = deque(maxlen=self.BUFFER_SIZE)
        self.seed = random.seed(seed)

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

