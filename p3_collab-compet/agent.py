import os
import numpy as np

import torch
from torch import nn, optim

from utils import device
from model import Actor, Critic



class DummyAgent:
    def __init__(self, state_size=24, action_size=2):
        self.STATE_SIZE = state_size
        self.ACTION_SIZE = action_size

    def act(self, state):
        actions = np.random.randn(self.ACTION_SIZE)
        actions = np.clip(actions, -1, 1)
        
        return actions
    
    def step(self, state, action, reward, next_state, done):
        pass

    def checkpoint(self):
        pass
    
    def load_networks(self):
        pass

    def save_networks(self):
        pass


class Agent:
    def __init__(self, state_size=24, action_size=2, replay_buffer=None):
        # Hyperparameters
        self.STATE_SIZE = state_size
        self.ACTION_SIZE = action_size

        self.TAU = 1e-2
        self.GAMMA = 1.0
        self.ACTOR_LR = 5e-4
        self.CRITIC_LR = 5e-4

        self.LEARN_EVERY = 1
        self.BATCH_SIZE = 1024

        # Actor
        self.local_actor = Actor(state_size, action_size).to(device)

        self.target_actor = Actor(state_size, action_size).to(device)
        self.target_actor.load_state_dict(self.local_actor.state_dict())
        self.target_actor.eval()

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=self.ACTOR_LR)

        # Critic
        self.local_critic = Critic(state_size, action_size).to(device)

        self.target_critic = Critic(state_size, action_size).to(device)
        self.target_critic.load_state_dict(self.local_critic.state_dict())
        self.target_critic.eval()

        self.critic_loss_fn = nn.MSELoss()

        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.CRITIC_LR)

        # Replay Memory
        self.replay_buffer = replay_buffer

        # State Variables
        self.t_step = 0

    def act(self, state, noise_scale=0.1):
        state = torch.from_numpy(state).float().to(device)

        self.local_actor.eval()
        with torch.no_grad():
            actions = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()

        noise = np.random.normal(loc=0.0, scale=noise_scale, size=actions.shape)
        actions += noise
        actions = np.clip(actions, -1, 1)
        
        return actions

    def step(self):
        self.t_step += 1

        if (len(self.replay_buffer) > self.BATCH_SIZE) and (self.t_step % self.LEARN_EVERY == 0):
            self._learn()
            self.t_step = 0

    def _learn(self):
        def _soft_update(local_model, target_model):
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

        # ------ Sample Experiences ------ #
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.BATCH_SIZE)

        # ------ Train Local Critic ------ #
        # Calculate Q-Targets
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_states_value_estimates = self.target_critic(next_states, next_actions)
            Q_targets = rewards + (self.GAMMA * (1 - dones) * next_states_value_estimates)
        # Compute Q-Estimates
        Q_estimates = self.local_critic(states, actions)
        # Compute Critic Loss (MSE between Q-Targets and Q-Estimates)
        msbe_loss = self.critic_loss_fn(Q_estimates, Q_targets)
        # Backpropagate and Optimize
        self.critic_optimizer.zero_grad()
        msbe_loss.backward()
        self.critic_optimizer.step()

        # ------ Train Local Actor ------ #
        # Get Actions from Local Actor
        actions_estimates = self.local_actor(states)
        # Get Value Estimates from Target Critic (without updating it)
        self.local_critic.eval()
        state_values = -self.local_critic(states, actions_estimates).mean()
        self.local_critic.train()
        # Backpropagate and Optimize (only local actor parameters)
        self.actor_optimizer.zero_grad()
        state_values.backward()
        self.actor_optimizer.step()

        # ------ Soft Update Target Networks ------ #
        _soft_update(self.local_critic, self.target_critic)
        _soft_update(self.local_actor, self.target_actor)

    def _reset(self):
        self.local_actor._reset_parameters()
        self.target_actor.load_state_dict(self.local_actor.state_dict())

        self.local_critic._reset_parameters()
        self.target_critic.load_state_dict(self.local_critic.state_dict())

    def save_checkpoint(self, i_episode, i_agent, save_dir='./checkpoints'):
        os.makedirs(save_dir, exist_ok=True)

        save_dict = {
            'local_actor_state_dict': self.local_actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),

            'local_critic_state_dict': self.local_critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }

        torch.save(save_dict, os.path.join(save_dir, f'agent-{i_agent}' + f'__episode-{i_episode}.' + '.pt'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.local_actor.load_state_dict(checkpoint['local_actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        self.local_critic.load_state_dict(checkpoint['local_critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    def save_networks(self, i_agent, checkpoint_actor='actor', checkpoint_critic='critic'):
        torch.save(self.local_actor.state_dict(), checkpoint_actor + f'__agent-{i_agent}' + '.pt')

    def load_networks(self, i_agent):
        self.local_actor.load_state_dict(torch.load('actor' + f'__agent-{i_agent}' + '.pt'))
