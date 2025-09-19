import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent
from utils import ScoreKeeper, ReplayBuffer, NoiseScheduler


def train_agent(
        n_episodes=10_000,
        checkpoint_every=500,
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
    # Agents
    common_replay_buffer = ReplayBuffer(buffer_size=1e+6)
    agents = [
        Agent(state_size=state_size, action_size=action_size, replay_buffer=common_replay_buffer),
        Agent(state_size=state_size, action_size=action_size, replay_buffer=common_replay_buffer)
    ]
    # Utilities
    noise_scheduler = NoiseScheduler(start=0.5, end=0.15, decay=0.995)
    scorekeeper = ScoreKeeper(num_agents=num_agents)

    for i_episode in range(1, N_EPISODES+1):
        # ------ Resets ------ #
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        scorekeeper.reset()

        noise_scheduler.step()
        noise_scale = noise_scheduler.get_noise_scale()

        # ------ Collect Episode ------ #
        t_step = 0
        while True:
            # Take Action
            action = np.stack([
                agents[i].act(state[i], noise_scale=noise_scale) 
                for i in range(num_agents)
            ])

            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done

            # Perform Updates
            for i in range(num_agents):
                common_replay_buffer.add(
                    state[i], action[i], 
                    reward[i] + (0.001 if t_step < 300 else -0.001), # Encourage longer plays, but not too long
                    next_state[i], done[i]
                )  
            for i in range(num_agents):
                agents[i].step()

            state = next_state

            scorekeeper.update_timestep(reward)
            
            t_step += 1

            # Check Terminal Condition
            if np.any(done):
                break
        
        # ------ Monitoring and Checkpointing ------ #
        # Monitoring
        is_solved = scorekeeper.update_episode(i_episode)
        if is_solved:
            for i in range(num_agents):
                agents[i].save_checkpoint(i_episode=i_episode, i_agent=i)
                agents[i].save_networks(i_agent=i)
            break
        # Checkpointing
        if i_episode % CHECKPOINT_EVERY== 0:
            for i in range(num_agents): agents[i].save_checkpoint(i_episode=i_episode, i_agent=i)

    env.close()

    return is_solved, scorekeeper.scores


def plot_scores(scores, save_filename='scores.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Plot raw scores faintly
    x = np.arange(len(scores))
    ax.plot(x, scores, color='tab:blue', alpha=0.25, label='Score per episode')

    # Running average that uses available history for first episodes
    window = 100
    running_avg = np.array([np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))])
    ax.plot(x, running_avg, color='tab:orange', linewidth=2, label=f'Running average ({window})')

    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_filename)


def watch_agent(
        max_t=1000,
        n_episodes=3,
        state_size=24,
        action_size=2,
        num_agents=2,
        unity_env_path='./Tennis_Linux/Tennis.x86_64'
):
    N_EPISODES = n_episodes

    env = UnityEnvironment(file_name=unity_env_path, no_graphics=False)
    brain_name = env.brain_names[0]

    agents = [
        Agent(state_size=state_size, action_size=action_size),
        Agent(state_size=state_size, action_size=action_size)
    ]
    for i in range(num_agents):
        agents[i].load_networks(i_agent=i)

    try:
        for _ in tqdm(range(1, N_EPISODES+1)):
            env_info = env.reset(train_mode=False)[brain_name]
            state = env_info.vector_observations

            for _ in range(max_t):
                action = np.stack([
                    agents[i].act(state[i], noise_scale=0.0) 
                    for i in range(num_agents)
                ])

                env_info = env.step(action)[brain_name]
                next_state, done = env_info.vector_observations, env_info.local_done

                state = next_state

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
