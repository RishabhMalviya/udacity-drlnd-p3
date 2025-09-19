[//]: # (Image References)
[image1]: https://raw.githubusercontent.com/RishabhMalviya/udacity-drlnd-p2/refs/heads/20agent/ddpg/p2_continuous-control/scores.png "Scores Plot"

# Learning Algorithm - VPG (Failed)
My first attempt was to implement VPG from first principles (see [this branch](https://github.com/RishabhMalviya/udacity-drlnd-p2/tree/1agent/REINFORCE)), but it failed spectacularly. Nonetheless, I learned a lot about how stochastic neural networks are implemented in PyTorch. Fo policy gradient methods, we need to get the log probabilities of our chosen actions. That means the neural network cannot be output deterministic values. The solution is to have the neural network output means and variances for each of the 4 action dimensions, and then use those to sample from a normal distribution.

## Things I Learned
- You don't actually need to backpropagate through the chosen actions for VPG, you need to backpropagate through the log probabilities. This means that you don't actually have to use the re-parametrization trick! In terms of PyTorch's distributional API, this means that you can sample the actions using `.sample()` instead of `.rsample()`.
- When this algorithm was not working on the actual task, I tried training the algorithm on a dummy probe environment, which rewarded the agent the closer it got to a fixed target action (look under the `Training a Distributional NN` in [this notebook](https://github.com/RishabhMalviya/udacity-drlnd-p2/blob/1agent/REINFORCE/p2_continuous-control/ScratchPad.ipynb)). The network was not learning until I amended the policy network to output only means (not variances). After some probing/debugging, I discovered the reason. The policy network would initially output bad actions; to reduce the probabilities of those bad actions, it would quickly learn to increase the variances it outputted. While great for encouraging exploration, it never learned to reduce the vraiances, and this prevented the network from ever learning to correct the means. 
    - *TL;DR* - Don't try to get the policy network to learn variances. It is better to think of it as an exploration/exploitation control variable, and vary it manually.


# Learning Algorithm - DDPG (Success)
Most policy-based methods were based on policy gradients, which required the calculation of log probabilities, and therefore, the use of distributional layers. I decided to switch to DDPG because it allowed the use of deterministic (as opposed to distributional) policy networks. Also, the fact that the benchmark implementation succeeded with DDPG was encouraging, especially since I had already spent several days failing to get VPG to work.

I think of DDPG is as a modified Q-Learning algorithm:
- It aims to learn the optimal Q-value function, and it does so using all of the tricks from DQNs like target-local networks and Replay Buffers. As the agent converges to the optimal policy, the Replay Buffer will get populated with experiences corresponding to that optimal policy, and the Q-Value network will learn the value functions for that optimal policy.
- The clever thing in DDPG is that the action-selection is not done from the Q-Value network. It is done using another (policy/actor) neural network. Since the Q-Value network requires states and actions as inputs, we can pass it an action that is calculated by this policy network. And since the policy network is deterministic and differentiable w.r.t the actions, we can backpropagate through it and optimize it!

The training steps for each neural network function as follows:
- *Q-Value (Critic) Network* - We use TD-estimates to teach the Q-Value network (just as in DQN). The clever thing is that the value for the `next_state` in the TD-target is calculated using the policy network. The Q-Value network gives us values for state-action pairs. But the TD-target requires the value for the state. So, we use the policy network to select the best action for `next_state`, and feed that into the Q-Value network to get the value of the `next_state`. Note that this TD-target is calculated using target networks (not local networks).
- *Policy (Actor) Network* - This is the most brilliant part of the algorithm in my opinion. We do not use policy gradients or anything to optimize the policy network. We've been passing the policy network's actions into the Q-Value network, right? So all we do is we maximize the output Q-Value. That's it. Simple, yet powerful. *NOTE* - This training step is done only with local (not target) networks.

Finally, the target networks are regularly soft-updated from the local networks.

## Exploration/Exploitation
Since the output of the policy network is deterministic, we add some noise to the outputs as a way to implement exploration. We can do this because the noisy actions are being used only to generate experience tuples. The only time we need actual outputs from our policy network (without noise) is during the learning steps; and during those, we re-calculate the actions for each state anyway. 

The original paper used an OU Noise process for encouraging exploration, but I read that stationary zero-mean Gaussian noise was providing similar results. After some experimentation, I set the Gaussian noise's scale (standard deviation) to `0.2`, and kept it fixed throughout training.

- I initially thought of using a schedule, where this scale would reduce as training progressed (similar to how we implement a schedule for the epsilon variable in epsilon-greedy action selection in Q-learning), but it turns out that wasn't necessary.
- It was actually quite important to choose a good value for this hyperparameter. When I had it set lower (at `0.1`), the algorithm was learning very slowly. When I had it set higher (at `0.5`), the agent wasn't learning at all. My intuition is that `0.2` is the sweet spot that facilitates some exploration, but also doesn't introduce so much randomness into the policy network's selected actions that it's outputs become meaningless.

## Network Architectures
I initially tried to keep these neural networks very small (with only one hidden layer). But that didn't work too well, so I added two layers. These were the final architectures used:

```python
class Actor(nn.Module):
    def __init__(self, state_size=33, action_size=4, seed=42, fc1_units=256, fc2_units=128):
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
^Notice how the output activation layer is `tanh`. This is because the environment documentation specified that the actions are supposed to be between `-1` and `1`.

```python
class Critic(nn.Module):
    def __init__(self, state_size=33, action_size=4, seed=42, fcs1_units=192, fca1_units=64, fc2_units=128):
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
^Notice how there are two input 'heads'. One for the state, and one for the action. This allows the network to reason about each independently before combining the information from both.

## Hyperparameters
These are all the hyperparameters used.
- `self.ACTOR_LR = 1e-4`: Learning rate for policy network (with Adam optimizer and zero weight decay)
- `self.CRITIC_LR = 1e-4`: Learning rate for Q-Value network (with Adam optimizer and zero weight decay)
- `self.NOISE_SCALE = 0.2`: Standard deviation for the zero-mean Gaussian noise that gets added to the actions during experience collection.
- `self.TAU = 1e-3`: This is the weight given to the local network during the soft update step that updates the target networks.
- `self.GAMMA = 0.95`: Discount factor. I think it was beneficial to keep it lower, because the effects of the agent's actions don't affect states so far out into the future.
- `self.LEARN_EVERY = 1`: After how many timesteps should learning be done.
- `self.BATCH_SIZE = 512`: How many experience-tuples to sample from the Replay Buffer during training.


## Things I Learned
- Suprisingly, I observed the biggest jumps in performace when I tuned the batch size and learning frequency for the algorithm. Before finding good values for these hyperparameters (I had the batch size set to `1000`), the algorithm was basically not learning at all.
- Switching to the 20 agent environment was also extrememly beneficial! This is obvious, since collecting more trajectories reduces variance.
- One important implementation detail in DDPG is that the local Q-value network does not get updated when we update the local policy network. This is important, because the gradients for the actor.
- Since DDPG is off-policy, it lends itself to a relatively clean implementations. In VPG, you have to carefully keep track of the log probability tensors during trajectory collection, because these log probabilities are what you use to train the networks during the learning step


## Plot of Rewards

![Score Plot][image1]

# References
1. [OpenAI's Spinning Up - VPG (REINFORCE)](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
2. [Probe Environments](https://andyljones.com/posts/rl-debugging.html)
3. [OpenAI's SpinningUp - DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
4. [DDPG Original Paper](https://arxiv.org/pdf/1509.02971)
