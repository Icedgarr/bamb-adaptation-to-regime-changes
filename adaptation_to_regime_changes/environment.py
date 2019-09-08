import numpy as np

from adaptation_to_regime_changes.definitions import generate_regular_d, generate_irregular_d


class Environment:

    def __init__(self, start_regime=0, start_state=int(np.random.binomial(1, 0.5, 1)), probability_reward=0.75,
                 reward_value=0.8, total_number_observations=1000, change_regime_observation=800):
        """
        :param start_regime: First regime of the environment
        :param start_state: First state of the environment
        :param total_number_observations: Total number of observations per trial
        :param change_regime_observation: Observation number on which the regime is changed (<total_number_observations)
        """
        self.state = start_state
        self.regime = start_regime
        self.duration = 0
        self.update_duration()
        self.observation_number = 0

        self.reward_value = reward_value
        self.probability_reward = probability_reward
        self.total_number_observations = total_number_observations
        self.change_regime_observation = change_regime_observation

    def make_observation(self, action):
        if action == self.state:
            is_rewarded = int(np.random.binomial(1, self.probability_reward, 1))
            reward = self.reward_value * is_rewarded + (1 - self.reward_value) * (1 - is_rewarded)
        else:
            is_rewarded = int(np.random.binomial(1, 1 - self.probability_reward, 1))
            reward = self.reward_value * is_rewarded + (1 - self.reward_value) * (1 - is_rewarded)
        self.update_environment()
        return reward

    def update_environment(self):
        self.update_state()
        self.update_duration()
        self.observation_number += 1
        if (self.observation_number % self.change_regime_observation) == 0:
            self.change_regime()

    def update_state(self):
        if self.duration == 0:
            self.state = 1 - self.state

    def update_duration(self):
        if self.duration == 0:
            if self.regime == 0:
                self.duration = generate_regular_d()
            else:
                self.duration = generate_irregular_d()
        else:
            self.duration -= 1

    def change_regime(self):
        self.regime = 1 - self.regime
