import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; 
sns.set_style("ticks")
      
from bandit import KArmedBandit
from bandit_policy import BanditPolicy


bandit_env = KArmedBandit()

episodes = 2000
steps = 1000
ucb_policy = BanditPolicy(bandit_env.action_space, UCB_param=2)
epsilon_greedy_policy = BanditPolicy(bandit_env.action_space, epsilon=0.1)

reward_runs = {}
optimal_runs = {}
for n, policy in enumerate([ucb_policy, epsilon_greedy_policy]):
    rewards = np.array([0. for _ in range(steps)])
    optimal = np.array([0. for _ in range(steps)])
    for m in range(episodes):
        policy.reset()
        if m % 100 == 0:
            print('Policy Number', n, 'Episode Num:', m)
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
        rewards += (1 / (m + 1)) * (episode_rewards - rewards)
        optimal += (1 / (m + 1)) * (episode_optimal - optimal)
    
    reward_runs[n] = rewards
    optimal_runs[n] = optimal

plt.plot(range(steps), reward_runs[0], label='UCB, c = 2')
plt.plot(range(steps), reward_runs[1], label='Epsilon Greedy, epsilon= 0.1', color='grey')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend(loc='lower right')

plt.savefig('UCB_Policy.png')