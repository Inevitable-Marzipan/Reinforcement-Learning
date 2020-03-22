import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; 
sns.set_style("ticks")
      
from bandit import KArmedBandit
from bandit_policy import BanditPolicy


bandit_env = KArmedBandit()

episodes = 2000
steps = 1000
optimal_runs = {}
reward_runs = {}
epsilon_vals = [0, 0.01, 0.1]
for epsilon in epsilon_vals:
    rewards = np.array([0. for _ in range(steps)])
    optimal = np.array([0. for _ in range(steps)])
    for n in range(episodes):
        policy = BanditPolicy(bandit_env.action_space, epsilon)
        if n % 100 == 0:
            print('Epsilon Val:', epsilon, 'Episode Num:', n)
        episode_rewards = []
        episode_optimal = []
        for _ in range(steps):
            action = policy.act()
            reward = bandit_env.step(action)
            policy.update_policy(action, reward)
            episode_rewards.append(reward)
            episode_optimal.append(1 if action == bandit_env.optimal_action else 0)
        episode_rewards = np.array(episode_rewards)
        episode_optimal = np.array(episode_optimal)
        rewards += (1 / (n + 1)) * (episode_rewards - rewards)
        optimal += (1 / (n + 1)) * (episode_optimal - optimal)
    
    reward_runs[epsilon] = rewards
    optimal_runs[epsilon] = optimal


plt.subplots_adjust(hspace = 0.3)
plt.subplot(211)
plt.xlabel('Steps')
plt.ylabel('Average Reward')

for epsilon in epsilon_vals:
    if epsilon == 0:
        color = 'g'
    elif epsilon == 0.01:
        color = 'r'
    elif epsilon == 0.1:
        color = 'b'
    plt.plot(range(steps), reward_runs[epsilon], color=color, label=f"Epsilon: {epsilon}")

plt.legend(loc="lower right")
    
plt.subplot(212)
plt.xlabel('Steps')
plt.ylabel('Optimal Move Fraction')

for epsilon in epsilon_vals:
    if epsilon == 0:
        color = 'g'
    elif epsilon == 0.01:
        color = 'r'
    elif epsilon == 0.1:
        color = 'b'
    plt.plot(range(steps), optimal_runs[epsilon], color=color, label=f"Epsilon: {epsilon}")

plt.savefig('Epsilon_Greedy_Policy.png')



