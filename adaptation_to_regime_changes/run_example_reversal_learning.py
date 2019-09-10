#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:56:24 2017

@author: sarah
"""

import numpy as np

#from plotting import *
from misc import *

import world 
import environment as env
import agent as agt
import perception as prc
import action_selection as asl
import itertools
import matplotlib.pylab as plt
from multiprocessing import Pool
from matplotlib.colors import LinearSegmentedColormap
import jsonpickle as pickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import json
import seaborn as sns
import os
from transition_probabilities import *
np.set_printoptions(threshold = 100000, precision = 5)



"""
set parameters
"""


save_data = False

trials = 200 #number of trials
T = 1 #number of time steps in each trial
no = p_o_t_s_t_a_t0.shape[0]
ns = p_s_t_d0.shape[0] #number of states
na = p_a_t.shape[0] #number of actions
nd = maximum_duration
npi = na
actions = np.array([0,1])
stable = True
    
"""
create matrices
"""
    
#generating probability of observations in each state
A = np.zeros((no,ns,na))

A[:] = p_o_t_s_t_a_t0
    

#state transition generative probability (matrix)
B = np.zeros((ns, ns, nd))

for d in range(nd):
    B[:,:,d] = p_st_st1_dt(d)
            
    
C = np.zeros((nd,nd))

distribution = np.zeros(nd) + 0.9/(nd-1)
distribution[20] = 0.1

C[:,0] = distribution
C[1:,1:] = np.eye(nd-1)

prior_durations = distribution
    
    
utility = np.array([0.99,0.01])
                        
"""
create environment (grid world)
"""

environment = env.Environment(start_regime=not stable,total_number_observations=trials)


"""
create policies
"""

pol = actions

prior_actions = np.ones(na) / na

"""
set state prior (where agent thinks it starts)
"""

state_prior = np.ones((ns)) / ns


"""
set action selection method
"""


ac_sel = asl.AveragedSelector(trials = trials, T = T, 
                              number_of_actions = na)



"""
set up agent
"""


# perception
bayes_prc = prc.BPMFPerception(A, B, C, utility, state_prior, prior_durations, prior_actions)


bayes_pln = agt.BayesianPlanner(bayes_prc, ac_sel, pol, no, nd,
                  trials = trials, T = T,
                  prior_states = state_prior,
                  number_of_states = ns, 
                  number_of_policies = npi)


"""
create world
"""

w = world.World(environment, bayes_pln, trials = trials, T = T)

"""
simulate experiment
"""

w.simulate_experiment()
    
    
"""
plot and evaluate results
"""

plt.figure(figsize=(10,5))
plt.plot(w.environment.states, label='state')
plt.plot(w.actions, ".", label='taken action')
plt.legend()
plt.title("actions taken")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(w.environment.states, label='state')
plt.plot(w.agent.posterior_actions[:,0], ".", label='posterior action 0')
plt.legend()
plt.title("posterior over actions")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(w.environment.states, label='state')
plt.plot(w.agent.posterior_states[:,0,0], "x", label='posterior state 0')
plt.legend()
plt.title("posterior over states")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(w.environment.states*10, label='state')
plt.plot(np.argmax(w.agent.posterior_durations[:,1,:], axis=1), "x", label='inferred durations')
plt.legend()
plt.title("most likely current duration")
plt.show()