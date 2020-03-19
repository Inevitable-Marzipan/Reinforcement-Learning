from statistics import NormalDist

class KArmedBandit:
    """
    Implements a K Armed Bandit

    """
    def __init__(self, arms=10, reward_mean=0, reward_std=1, sigma=1):
        self.arms = arms
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.sigma = sigma

        self.actual_reward_values = NormalDist(mu=reward_mean, sigma=reward_std).samples(arms)
        self.reward_distributions = \
            [NormalDist(mu=actual_reward_value, sigma=sigma) for actual_reward_value in self.actual_reward_values]
        
        self.action_space = [x for x in range(self.arms)]
        self.optimal_action = max(zip(self.actual_reward_values, range(len(self.actual_reward_values))))[1]
    
    def step(self, action):
        assert action in self.action_space,  f"invalid action {action}, type: {type(action)}"
        return self.reward_distributions[action].samples(1)

