"""This module contains the class that defines the interaction between
different modules that govern agent's behavior.
"""
import numpy as np
from misc import ln, softmax

        
class BayesianPlanner(object):
    
    def __init__(self, perception, action_selection, policies,
                 number_of_observations, number_of_durations,
                 prior_states = None, prior_policies = None, 
                 trials = 1, T = 2, number_of_states = 6,
                 number_of_policies = 10):
        
        #set the modules of the agent
        self.perception = perception
        self.action_selection = action_selection
        
        #set parameters of the agent
        self.ns = number_of_states #number of states
        self.na = number_of_policies #number of policies
        self.no = number_of_observations
        self.nd = number_of_durations
        
        if policies is not None:
            self.policies = policies
        else:
            #make action sequences for each policy
            self.policies = np.eye(self.npi, dtype = int)
        
        self.actions = np.unique(self.policies)
        
        if prior_states is not None:
            self.prior_states = prior_states
        else:
            self.prior_states = np.ones(self.nh)
            self.prior_states /= self.prior_states.sum()
            
        if prior_policies is not None:
            self.prior_policies = prior_policies
        else:
            self.prior_policies = np.ones(self.na)/self.na
        
        #set various data structures
        self.actions = np.zeros((trials), dtype = int)
        self.posterior_states = np.zeros((trials, T+1, self.ns))
        self.posterior_actions = np.zeros((trials, self.na))
        print(trials, self.no, self.na)
        self.posterior_observations = np.zeros((trials, self.no, self.na))
        self.posterior_durations = np.zeros((trials, T+1, self.nd))
        self.observations = np.zeros((trials), dtype = int)
        

    def reset_beliefs(self, actions):
        self.actions[:,:] = actions 
        self.posterior_states[:,:,:] = 0.
        self.posterior_policies[:,:,:] = 0.
        
        self.perception.reset_beliefs()
        self.planning.reset_beliefs()
        self.action_selection.reset_beliefs()
        
        
    def update_beliefs(self, tau, observation, response):
        if tau > 0:
            self.actions[tau-1] = response
            self.observations[tau-1] = observation
        

        self.perception.update_beliefs(tau, observation, response)

        self.posterior_states[tau] = self.perception.posterior_states.copy()
        self.posterior_durations[tau] = self.perception.posterior_durations.copy()
        self.posterior_observations[tau] = self.perception.posterior_observations.copy()
        self.posterior_actions[tau] = self.perception.posterior_policies.copy()

            
    
    def generate_response(self, tau):
        
        #get response probability
        posterior_policies = self.posterior_actions[tau]
        controls = self.policies

        self.actions[tau] = self.action_selection.select_desired_action(tau, posterior_policies, controls)
            
        
        return self.actions[tau]
    
    
    def estimate_response_probability(self, tau):
        
        posterior_policies = self.posterior_policies[1]
        controls = self.policies

        self.action_selection.estimate_action_probability(tau,
                                                 posterior_policies, controls)
    
    
    def __make_expected_predicted(self, controls, actions, trans_matrx, 
                                  state_beliefs, policy_beliefs):
        expct = np.zeros(self.nh)
        for pi, u in enumerate(controls):
            expct[:] += trans_matrx[:,:, u].dot(state_beliefs)*policy_beliefs[pi]
        
        prdct = np.zeros((self.nh, self.na))
        for a in actions:
            prdct[:, a] = trans_matrx[:,:, a].dot(state_beliefs)
            
        return expct, prdct

