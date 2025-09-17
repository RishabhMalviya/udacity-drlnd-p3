[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

This is my submission for Project 3 of Udacity's Deep Reinforcement Learning Nanodegree, Collaboration and Competition.

# Environment Details

The environment is the Unity ML Agents [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment:

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net

## Reward Structure
If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

## State and Action Spaces
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

## Completion Criteria
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Getting Started

This section will provide instructions on how to setup the repository code. It is tested in a Linux environment.

1. Run the following commands to download and extract the Unity ML Agents environment:
```bash
cd ./p3_collab-compet
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
unzip Tennis_Linux.zip
rm Tennis_Linux.zip
cd ..
```

2. Create (and activate) a new environment with Python 3.6.
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
	
3. Install the python dependencies into the actiavted `conda` environment:
```bash
cd ./python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


# Instructions

## Running the Training Code
To train the agent, make sure the `conda` environment is activated (if it isn't, run `source activate drlnd`), and that you are in the root of the repository. Then:

1. Navigate into the `p3_collab-compet` folder with: `cd ./p3_collab-compet` 
2. Run the training script: `python main.py`

If the environment gets solved, the model weights will get saved in `p3_collab-compet/checkpoint_actor.pth` and ``p3_collab-compet/checkpoint_critic.pth`, and you will see a simulation of the trained agent.

## Report
The details of the successfully trained agent and the learning algorithm can be found in `Report.ipynb`.

## Watch Trained Agent
To watch the trained agent:

1. Run `jupyter notebook`
2. Run the `p3_collab-compet/Watch Trained Agent.ipynb` Jupyter notebook.