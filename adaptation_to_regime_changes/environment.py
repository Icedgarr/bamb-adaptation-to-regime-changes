import numpy as np

from definitions import generate_regular_d, generate_irregular_d


class Environment:

    def __init__(self, start_regime=0, start_state=int(np.random.binomial(1, 0.5, 1)), probability_reward=0.75,
                 reward_value=0.8, total_number_observations=1000, change_regime_observation=800, trials=100, T=1):
        """
        :param start_regime: First regime of the environment
        :param start_state: First state of the environment
        :param total_number_observations: Total number of observations per trial
        :param change_regime_observation: Observation number on which the regime is changed (<total_number_observations)
        """
        self.total_number_observations = total_number_observations
        self.change_regime_observation = change_regime_observation
        self.state = start_state
        self.regime = start_regime
        self.duration = 0
        self.durations = np.zeros(total_number_observations)
        self.update_duration(-1)
        self.observation_number = 0
        
        self.states = np.zeros(total_number_observations)
        self.states[0] = start_state

        self.reward_value = reward_value
        self.probability_reward = probability_reward

    def generate_observations(self, action):
        if action == self.state:
            is_rewarded = int(np.random.binomial(1, self.probability_reward, 1))
            reward = is_rewarded #self.reward_value * is_rewarded + (1 - self.reward_value) * (1 - is_rewarded)
        else:
            is_rewarded = int(np.random.binomial(1, 1 - self.probability_reward, 1))
            reward = is_rewarded #self.reward_value * is_rewarded + (1 - self.reward_value) * (1 - is_rewarded)
        #self.update_environment()
        return reward

    def update_environment(self, tau):
        self.update_state(tau)
        self.update_duration(tau)
        self.observation_number += 1
        if (self.observation_number % self.change_regime_observation) == 0:
            self.change_regime()

    def update_state(self, tau):
        if self.duration == 0:
            self.state = 1 - self.state
        if tau < self.total_number_observations-1:
            self.states[tau+1] = self.state

    def update_duration(self, tau):
        if self.duration == 0:
            if self.regime == 0:
                self.duration = generate_regular_d()
            else:
                self.duration = generate_irregular_d()
        else:
            self.duration -= 1
        if tau < self.total_number_observations-1:
            self.durations[tau+1] = self.duration

    def change_regime(self):
        self.regime = 1 - self.regime
