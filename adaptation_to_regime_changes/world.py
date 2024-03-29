"""This module contains the World class that defines interactions between 
the environment and the agent. It also keeps track of all observations and 
actions generated during a single experiment. To initiate it one needs to 
provide the environment class and the agent class that will be used for the 
experiment.
"""
import numpy as np
from misc import ln
#from inference import Inference

class World(object):
    
    def __init__(self, environment, agent, trials = 1, T = 10):
        #set inital elements of the world to None        
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial
        
        self.free_parameters = {}
        
        #container for observations
        self.observations = np.zeros((self.trials), dtype = int)
                
        #container for agents actions
        self.actions = np.zeros((self.trials), dtype = int)
        
    def simulate_experiment(self):
        """This methods evolves all the states of the world by iterating 
        through all the trials and time steps of each trial.
        """
        
        for tau in range(self.trials):
            for t in range(self.T):
                self.__update_world(tau, t)
 
    
    def estimate_par_evidence(self, params, method='MLE'):

        
        val = np.zeros(params.shape[0])
        for i, par in enumerate(params):
            if method == 'MLE':
                val[i] = self.__get_log_likelihood(par)
            else:
                val[i] = self.__get_log_jointprobability(par)
        
        return val
    
    def fit_model(self, bounds, n_pars, method='MLE'):
        """This method uses the existing observation and response data to 
        determine the set of parameter values that are most likely to cause 
        the meassured behavior. 
        """
        
        inference = Inference(ftol = 1e-4, xtol = 1e-8, bounds = bounds, 
                           opts = {'np': n_pars})
        
        if method == 'MLE':
            return inference.infer_posterior(self.__get_log_likelihood)
        else:
            return inference.infer_posterior(self.__get_log_jointprobability)
        
        
    #this is a private method do not call it outside of the class
    def __get_log_likelihood(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()
        
        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)
        
        return ln(self.agent.asl.control_probability[p1, p2, p3]).sum()
    
    def __get_log_jointprobability(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()
        
        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)
        
        ll = ln(self.agent.asl.control_probability[p1, p2, p3]).sum()
        
        return  ll + self.agent.log_prior()
    
    #this is a private method do not call it outside of the class    
    def __update_model(self):
        """This private method updates the internal states of the behavioral 
        model given the avalible set of observations and actions.
        """

        for tau in range(self.trials):
            for t in range(self.T):
                if t == 0:
                    response = None
                    observation = None
                else:
                    response = self.actions[tau-1]
                    observation = self.observations[tau-1]
                
                self.agent.update_beliefs(tau, observation, response)
                self.agent.plan_behavior(tau)
                self.agent.estimate_response_probability(tau)
    
    #this is a private method do not call it outside of the class    
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the 
        whole world. Here we update the hidden state(s) of the environment, 
        the perceptual and planning states of the agent, and in parallel we 
        generate observations and actions.
        """
        
        if tau==0:
            response = None
            observation = None
        else:
            observation = self.observations[tau-1]
            response = self.actions[tau-1]
            self.environment.update_environment(tau)        
    
        self.agent.update_beliefs(tau, observation, response)
        
        self.actions[tau] = self.agent.generate_response(tau)
            
        self.observations[tau] = \
            self.environment.generate_observations(self.actions[tau])
            

        

class FakeWorld(object):
    
    def __init__(self, environment, agent, observations, actions, trials = 1, T = 10):
        #set inital elements of the world to None        
        self.environment = environment
        self.agent = agent

        self.trials = trials # number of trials in the experiment
        self.T = T # number of time steps in each trial
        
        self.free_parameters = {}
        
        #container for observations
        self.observations = np.zeros((self.trials, self.T), dtype = int)
        self.observations[:] = np.array([observations for i in range(self.trials)])
                
        #container for agents actions
        self.actions = np.zeros((self.trials, self.T), dtype = int)
        self.actions[:] = np.array([actions for i in range(self.trials)])
        
    def simulate_experiment(self):
        """This methods evolves all the states of the world by iterating 
        through all the trials and time steps of each trial.
        """
        
        for tau in range(self.trials):
            for t in range(self.T):
                self.__update_world(tau, t)
 
    
    def estimate_par_evidence(self, params, method='MLE'):

        
        val = np.zeros(params.shape[0])
        for i, par in enumerate(params):
            if method == 'MLE':
                val[i] = self.__get_log_likelihood(par)
            else:
                val[i] = self.__get_log_jointprobability(par)
        
        return val
    
    def fit_model(self, bounds, n_pars, method='MLE'):
        """This method uses the existing observation and response data to 
        determine the set of parameter values that are most likely to cause 
        the meassured behavior. 
        """
        
        inference = Inference(ftol = 1e-4, xtol = 1e-8, bounds = bounds, 
                           opts = {'np': n_pars})
        
        if method == 'MLE':
            return inference.infer_posterior(self.__get_log_likelihood)
        else:
            return inference.infer_posterior(self.__get_log_jointprobability)
        
        
    #this is a private method do not call it outside of the class
    def __get_log_likelihood(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()
        
        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)
        
        return ln(self.agent.asl.control_probability[p1, p2, p3]).sum()
    
    def __get_log_jointprobability(self, params):
        self.agent.set_free_parameters(params)
        self.agent.reset_beliefs(self.actions)
        self.__update_model()
        
        p1 = np.tile(np.arange(self.trials), (self.T, 1)).T
        p2 = np.tile(np.arange(self.T), (self.trials, 1))
        p3 = self.actions.astype(int)
        
        ll = ln(self.agent.asl.control_probability[p1, p2, p3]).sum()
        
        return  ll + self.agent.log_prior()
    
    #this is a private method do not call it outside of the class    
    def __update_model(self):
        """This private method updates the internal states of the behavioral 
        model given the avalible set of observations and actions.
        """

        for tau in range(self.trials):
            for t in range(self.T):
                if t == 0:
                    response = None
                    observation = None
                else:
                    response = self.actions[tau, t-1]
                    observation = self.observations[tau,t-1]
                
                self.agent.update_beliefs(tau, observation, response)
                self.agent.estimate_response_probability(tau)
    
    #this is a private method do not call it outside of the class    
    def __update_world(self, tau, t):
        """This private method performs a signel time step update of the 
        whole world. Here we update the hidden state(s) of the environment, 
        the perceptual and planning states of the agent, and in parallel we 
        generate observations and actions.
        """
        #print(tau, t)
        if t==0:
            self.environment.set_initial_states(tau)
            response = None
            observation = None
        else:
            response = self.actions[tau-1]
            observation = self.observations[tau-1]
        
    
        self.agent.update_beliefs(tau, observation, response)
        self.agent.estimate_response_probability(tau)
        