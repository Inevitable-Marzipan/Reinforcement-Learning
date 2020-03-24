import numpy as np


class BanditPolicy():

    def __init__(self, action_space, epsilon=0., initial_value=0., step_size=None, UCB_param=None):
        self.action_space = action_space
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.step_size = step_size
        self.UCB_param = UCB_param
        self.q_estimation = {action: self.initial_value for action in self.action_space}
        self.n = {action: 0 for action in self.action_space}
        self.action_count = 0

    def act(self):

        if self.UCB_param is not None:
            ucb_action_vals = [(action, self.q_estimation[action] + \
                             self.UCB_param * np.sqrt(np.log(self.action_count + 1) / (self.n[action] + 1e-5))) 
                             for action in self.action_space]
            max_actions = self._allmax(ucb_action_vals, key=lambda x: x[1])
            if len(max_actions) == 1:
                return max_actions[0]
            else:
                return max_actions[np.random.randint(0, len(max_actions) - 1)]
            

        else:
            if np.random.random() > (1 - self.epsilon):
                return self.action_space[np.random.randint(0, len(self.action_space) - 1)]
            else:
                action_estimates = list((zip(self.q_estimation.keys(), self.q_estimation.values())))
                max_actions = self._allmax(action_estimates , key=lambda x: x[1])
                if len(max_actions) == 1:
                    return max_actions[0]
                else:
                    return max_actions[np.random.randint(0, len(max_actions) - 1)]
        
    def update_policy(self, action, reward):
        assert action in self.action_space,  f"invalid action {action}, type: {type(action)}"
        self.action_count += 1
        self.n[action] += 1
        current_estimation = self.q_estimation[action]
        if self.step_size is not None:
            self.q_estimation[action] = current_estimation + \
                                        self.step_size * (reward - current_estimation)
        else:
            self.q_estimation[action] = current_estimation + \
                                        (1 / self.n[action]) * (reward - current_estimation)
    
    @staticmethod
    def _allmax(a, key=lambda x: x):
        if len(a) == 0:
            return []
        all_ = [0]
        max_ = key(a[0])
        for i in range(1, len(a)):
            if key(a[i]) > max_:
                all_ = [i]
                max_ = key(a[i])
            elif key(a[i]) == max_:
                all_.append(i)
        return all_
    
    def reset(self):
        self.q_estimation = {action: self.initial_value for action in self.action_space}
        self.n = {action: 0 for action in self.action_space}
        self.action_count = 0
