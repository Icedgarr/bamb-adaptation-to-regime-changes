from misc import ln, softmax
import numpy as np
import matplotlib.pylab as plt
    
class BPMFPerception(object):
    def __init__(self,
                 generative_model_observations, 
                 generative_model_states,
                 generative_model_durations,
                 prior_observations,
                 prior_states,
                 prior_durations,
                 prior_actions,
                 T=1):

        self.generative_model_observations = generative_model_observations
        self.generative_model_states = generative_model_states
        self.generative_model_durations = generative_model_durations
        self.prior_observations = prior_observations
        self.prior_states = prior_states
        self.prior_durations = prior_durations
        self.prior_actions = prior_actions
        self.T = T
        self.na = prior_actions.shape[0]
        self.ns = prior_states.shape[0]
        self.no = prior_observations.shape[0]
        self.nd = prior_durations.shape[0]
        self.posterior_states = np.zeros((T+1,self.ns))
        self.posterior_durations = np.zeros((T+1,self.nd))
        self.posterior_observations = np.zeros((1,self.no,self.na))
        self.posterior_policies = np.zeros((1,self.na))

    def update_beliefs(self, tau, observation, response):
        
        if tau == 0:
            self.posterior_states[0] = 1./self.ns
            self.posterior_durations[0] = 1./self.nd
            self.posterior_states[1] = self.prior_states
            self.posterior_durations[1] = self.prior_durations
            self.posterior_observations[0] = np.exp(ln(self.prior_observations) + (self.posterior_states[1][np.newaxis,:,np.newaxis]*ln(self.generative_model_observations)).sum(axis=1))
            
            self.posterior_observations[0] /= self.posterior_observations[0].sum(axis=0)
            
            self.posterior_policies[0] = softmax(ln(self.prior_actions) - (self.posterior_observations[0] * ln(self.posterior_observations[0])).sum(axis=0)  + (self.posterior_observations[0][:,np.newaxis,:]*self.posterior_states[1,np.newaxis,:,np.newaxis]*ln(self.generative_model_observations)).sum(axis=(0,1)))
 
            
        else:
            old_post_s = self.posterior_states[1].copy()
            old_post_d = self.posterior_durations[1].copy()
            
            self.posterior_states[0] = softmax(ln(self.prior_states) + ln(self.generative_model_observations[observation,:,response]))
            self.posterior_durations[0] = old_post_d
            self.posterior_states[1] = softmax((old_post_d[np.newaxis,np.newaxis,:]*self.posterior_states[0][np.newaxis,:,np.newaxis]*ln(self.generative_model_states)).sum(axis=(1,2)))
            self.posterior_durations[1] = softmax((old_post_d[np.newaxis,:]*ln(self.generative_model_durations)).sum(axis=1)) \
                                            #+ (self.posterior_states[0][np.newaxis,:,np.newaxis]*self.posterior_states[1][:,np.newaxis,np.newaxis]*ln(self.generative_model_states)).sum(axis=(0,1)))
            self.posterior_observations[0] = np.exp(ln(self.prior_observations[:,np.newaxis]) + (self.posterior_states[1][np.newaxis,:,np.newaxis]*ln(self.generative_model_observations)).sum(axis=1))

            self.posterior_observations[0] /= self.posterior_observations[0].sum(axis=0)

            self.posterior_policies[0] = softmax(ln(self.prior_actions) - (self.posterior_observations[0] * ln(self.posterior_observations[0])).sum(axis=0) + (self.posterior_observations[0][:,np.newaxis,:]*self.posterior_states[1,np.newaxis,:,np.newaxis]*ln(self.generative_model_observations)).sum(axis=(0,1)))
            

