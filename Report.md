[//]: # (Image References)
[image1]: https://raw.githubusercontent.com/RishabhMalviya/udacity-drlnd-p3/refs/heads/master/p3_collab-compet/scores.png "Scores Plot"

# Learning Algorithm - MADDPG
MADDPG extends DDPG to multi-agent settings by using centralized critics with decentralized actors. Each agent learns its own policy while accounting for other agents’ behaviors through the critic. This design makes learning more stable and enables effective cooperation and competition in complex multi-agent environments.

## Hyperparameters
```python
TAU = 1e-2
GAMMA = 1.0
ACTOR_LR = 5e-4
CRITIC_LR = 5e-4

LEARN_EVERY = 1
BATCH_SIZE = 1024
BUFFER_SIZE = 1e+6
```

## Noise Schedule
Also, the noise schedule was engineered so that the frst 300 episodes work with `0.5` scale noise, and from there on, it decays to `0.1` with a decay rate of `0.995`:
```python
class NoiseScheduler:
    def __init__(self, start=0.5, end=0.1, decay=0.995, num_episodes_before_decay=250):
        self.start = start
        self.end = end
        self.decay = decay

        self.num_episodes_before_decay = num_episodes_before_decay
        self.curr_episode = 1

        self.current = start        

    def reset(self):
        self.current = self.start

    def step(self):
        self.curr_episode += 1

        if self.curr_episode > self.num_episodes_before_decay:
            self.current = max(self.end, self.current * self.decay)

    def get_noise_scale(self):
        return self.current if self.curr_episode > self.num_episodes_before_decay else self.start
```

## Reward Engineering
When the agents were taking a long time to begin learning (~1000 episodes), I started plotting the length of the episodes. This revealed that up until about episode ~1000, the episodes would run for only ~15 timesteps.

Seeing this, I modified the rewards (that went into the ReplayBuffer). I added a small reward of `0.001` for each timestep of the episode. This encouraged the agents to keep the play going for as long as possible.

But the agents took this too far. I started seeing episodes that were ~1500 timesteps long, with a score of still around `0.3`. So, then I modified the rewards again. The agents would get an extra bonus of `0.001` up until episode 300, but from there on out, they would each get a penalty of `-0.001`. This encouraged the agents to kplay for longer in initial stages of training, and then to win quicker (be more attacking) in later stages of training. 

## Network Architectures
Actor:
```python
class Actor(nn.Module):
    def __init__(self, state_size=24, action_size=2, seed=42, fc1_units=256, fc2_units=256):
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self._reset_parameters()
        
    def _reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return F.tanh(self.fc3(x))
```

Critic:
```python
class Critic(nn.Module):
    def __init__(self, state_size=24, action_size=2, seed=42, fcs1_units=192, fca1_units=64, fc2_units=128):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fca1 = nn.Linear(action_size, fca1_units)
        self.fc2 = nn.Linear(fcs1_units + fca1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fca1.weight.data.uniform_(*hidden_init(self.fca1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        xa = F.relu(self.fca1(action))

        x = F.relu(self.fc2(torch.cat((xs, xa), dim=1)))

        return self.fc3(x)
```

The initiailization of the layer weights was done with "fan-in" variance-scaling uniform initialization:
```python
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
```

# Plot of Rewards

![Score Plot][image1]

# Ideas for Future Work

