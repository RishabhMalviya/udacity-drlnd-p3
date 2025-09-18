import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent
from utils import ScoreKeeper, ReplayBuffer


def train_agent(
        n_episodes=500,
        checkpoint_every=100, 
        state_size=24,
        action_size=2,
        num_agents=2,
        unity_env_path='./Tennis_Linux/Tennis.x86_64'
):
    # ------ Hyperparameters ------ #
    N_EPISODES = n_episodes
    CHECKPOINT_EVERY = checkpoint_every

    # ------ Instantiations ------ #
    # Environment
    env = UnityEnvironment(file_name=unity_env_path, no_graphics=True)
    brain_name = env.brain_names[0]
    # Common Replay Buffer
    common_replay_buffer = ReplayBuffer(buffer_size=1e+5)
    # Agent
    agents = [
        Agent(state_size=state_size, action_size=action_size, replay_buffer=common_replay_buffer),
        Agent(state_size=state_size, action_size=action_size, replay_buffer=common_replay_buffer)
    ]
    # Scorekeeper
    scorekeeper = ScoreKeeper(num_agents=num_agents)
    # Solved State
    solved = False

    for i_episode in range(1, N_EPISODES+1):
        # ------ Resets ------ #
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        scorekeeper.reset()

        # ------ Collect Episode ------ #
        while True:
            # Take Action
            action = np.stack([agents[i].act(state[i]) for i in range(num_agents)])

            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
            

            # Perform Updates
            for i in range(num_agents): common_replay_buffer.add(state[i], action[i], reward[i], next_state[i], done[i])
            for i in range(num_agents): agents[i].step()

            state = next_state

            scorekeeper.update_timestep(reward)


            # Check Terminal Condition
            if np.any(done):
                break
        
        # ------ Monitoring and Checkpointing ------ #
        # Monitoring
        solved = scorekeeper.update_episode(i_episode)
        if solved:
            for i in range(num_agents):
                agents[i].save_networks(checkpoint_actor=f'trained_actor-{i}.pt', checkpoint_critic=f'trained_critic-{i}.pt')
            break
        # Checkpointing
        if i_episode % CHECKPOINT_EVERY== 0:
            for i in range(num_agents):
                agents[i].checkpoint(i_episode=i_episode, checkpoint_actor=f'trained_actor-{i}.pt', checkpoint_critic=f'trained_critic-{i}.pt')

    env.close()

    return solved, scorekeeper.scores


def plot_scores(scores, save_filename='scores.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.savefig(save_filename)


def watch_agent(
        n_episodes=3,
        state_size=24,
        action_size=2,
        num_agents=2,
        unity_env_path='./Tennis_Linux/Tennis.x86_64'
):
    N_EPISODES = n_episodes

    env = UnityEnvironment(file_name=unity_env_path, no_graphics=False)
    brain_name = env.brain_names[0]

    agent = Agent(num_agents=num_agents, state_size=state_size, action_size=action_size)
    agent.load_networks()

    try:
        for _ in tqdm(range(1, N_EPISODES+1)):
            env_info = env.reset(train_mode=False)[brain_name]
            state = env_info.vector_observations

            while True:
                env_info = env.step(agent.act(state))[brain_name]

                next_state = env_info.vector_observations
                state = next_state

                done = env_info.local_done
                if np.any(done):
                    break
    finally:
        env.close()


if __name__ == '__main__':
    solved, scores = train_agent(
        max_t=1000,
        n_episodes=2000,
        checkpoint_every=100
    )
    
    plot_scores(scores)
    
    if solved:
        watch_agent(
            max_t=250,
            n_episodes=1
        )
