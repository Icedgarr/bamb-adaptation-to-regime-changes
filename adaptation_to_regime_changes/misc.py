#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as scs
import matplotlib.pylab as plt
import seaborn as sns

def evolve_environment(env):
    trials = env.hidden_states.shape[0]
    T = env.hidden_states.shape[1]
    
    for tau in range(trials):
        for t in range(T):
            if t == 0:
                env.set_initial_states(tau)
            else:
                if t < T/2:
                    env.update_hidden_states(tau, t, 0)
                else:
                    env.update_hidden_states(tau, t, 1)
                    
                    
def compute_performance(rewards):
    return rewards.mean(), rewards.var()


def ln(x):
    with np.errstate(divide='ignore'):
        return np.nan_to_num(np.log(x))
    
def logit(x):
    with np.errstate(divide = 'ignore'):
        return np.nan_to_num(np.log(x/(1-x)))
    
def logistic(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis = 0))
    #print("e_x", e_x)
    return np.nan_to_num(e_x / e_x.sum(axis = 0))

def sigmoid(x, a=1., b=1., c=0., d=0.):
    f = a/(1. + np.exp(-b*(x-c))) + d
    return f

def exponential(x, b=1., c=0., d=0.):
    f = np.exp(b*(x-c)) + d
    return f

def lognormal(x, mu, sigma):
    return -.5*(x-mu)*(x-mu)/(2*sigma) - .5*ln(2*np.pi*sigma)

def generate_bandit_timeseries_stable(Rho_0, nb, trials, changes):
    Rho = np.zeros((trials, Rho_0.shape[0], Rho_0.shape[1]))
    Rho[0] = Rho_0.copy()
    
    #set dummy state
    Rho[:,0,0] = 1

    for tau in range(1,trials):
        change = np.random.choice(changes, size=nb)
        Rho[tau,0,1:] = Rho[tau-1,0,1:] + change
        Rho[tau,1,1:] = Rho[tau-1,1,1:] - change
        Rho[tau][Rho[tau] > 1.] = 1.
        Rho[tau][Rho[tau] < 0.] = 0.
        
    return Rho


def generate_bandit_timeseries_change(Rho_0, nb, trials, changes):
    Rho = np.zeros((trials, Rho_0.shape[0], Rho_0.shape[1]))
    Rho[0] = Rho_0.copy()
    
    #set dummy state
    Rho[:,0,0] = 1
    
    means = np.zeros((trials,2, nb+1))
    means[:,1,1:] = 0.05
    means[0,1,1] = 0.95
    means[:,0,1:] = 0.95
    means[0,0,1] = 0.05
    
    for tau in range(0,nb-1):
        for i in range(1,trials//nb+1):
            means[tau*(trials//nb)+i,1,tau+1] =  means[tau*(trials//nb)+i-1,1,tau+1] - 0.9/(trials//nb)
            means[tau*(trials//nb)+i,1,tau+2] =  means[tau*(trials//nb)+i-1,1,tau+2] + 0.9/(trials//nb)
            
            means[tau*(trials//nb)+i,0,tau+1] =  1 - means[tau*(trials//nb)+i,1,tau+1]
            means[tau*(trials//nb)+i,0,tau+2] =  1 - means[tau*(trials//nb)+i,1,tau+2]

#    for tau in range(1,trials):
#        change = np.random.choice(changes, size=nb)
#        Rho[tau,0,1:] = Rho[tau-1,0,1:] + change
#        Rho[tau,1,1:] = Rho[tau-1,1,1:] - change
#        Rho[tau][Rho[tau] > 1.] = 1.
#        Rho[tau][Rho[tau] < 0.] = 0.
        
    return means

def generate_bandit_timeseries_slowchange(trials, nr, ns, nb):
    Rho = np.zeros((trials, nr, ns))
    Rho[:,0,0] = 1.
    Rho[:,0,1:] = 0.9
    for j in range(1,nb+1):
        Rho[:,j,j] = 0.1
    Rho[:,0,1] = 0.1
    Rho[:,1,1] = 0.9
    

    for i in range(1,trials):
        Rho[i,2,2] =  Rho[i-1,2,2] + 0.8/(trials)
        Rho[i,1,1] =  Rho[i-1,1,1] - 0.8/(trials)
        
        Rho[i,0,1] =  1 - Rho[i,1,1]
        Rho[i,0,2] =  1 - Rho[i,2,2]
        
    return Rho


def generate_bandit_timeseries_training(trials, nr, ns, nb, n_training, p=0.9, offset = 0):
    Rho = np.zeros((trials, nr, ns))
    Rho[:,0,0] = 1.
    Rho[:,0,1:] = p
    for j in range(1,nb+1):
        Rho[:,j,j] = 1.-p
    for i in range(0,trials+1,nb):
        for k in range(nb):
            for j in range(1,nb+1):
                Rho[(i+k)*trials//(nb*n_training):(i+k+1)*trials//(nb*n_training),j,j] = 1.-p
            Rho[(i+k)*trials//(nb*n_training):(i+k+1)*trials//(nb*n_training),0,1:] = p
            Rho[(i+k)*trials//(nb*n_training):(i+k+1)*trials//(nb*n_training),k+1,k+1+offset] = p
            Rho[(i+k)*trials//(nb*n_training):(i+k+1)*trials//(nb*n_training),0,k+1+offset] = 1.-p
#            Rho[(i+k)*trials//(nb*n_training):(i+k+1)*trials//(nb*n_training),1,k+2] = 0.1
#            Rho[(i+k)*trials//(nb*n_training):(i+k+1)*trials//(nb*n_training),0,k+2] = 0.9
#            Rho[(i+k+1)*trials//(nb*n_training):(i+k+2)*trials//(nb*n_training),1,k+2] = 0.9
#            Rho[(i+k+1)*trials//(nb*n_training):(i+k+2)*trials//(nb*n_training),0,k+2] = 0.1
#            Rho[(i+k+1)*trials//(nb*n_training):(i+k+2)*trials//(nb*n_training),1,k+1] = 0.1
#            Rho[(i+k+1)*trials//(nb*n_training):(i+k+2)*trials//(nb*n_training),0,k+1] = 0.9
    
    return Rho


def D_KL_nd_dirichlet(alpha, beta):
    D_KL = 0
    assert(len(alpha.shape) == 3)
    for j in range(alpha.shape[1]):
        D_KL += -scs.gammaln(alpha[:,j]).sum(axis=0) + scs.gammaln(alpha[:,j].sum(axis=0)) \
         +scs.gammaln(beta[:,j]).sum(axis=0) - scs.gammaln(beta[:,j].sum(axis=0)) \
         + ((alpha[:,j]-beta[:,j]) * (scs.digamma(alpha[:,j]) - scs.digamma(alpha[:,j].sum(axis=0))[np.newaxis,:])).sum(axis=0)
     
    return D_KL

def D_KL_dirichlet_categorical(alpha, beta):
    
    D_KL = -scs.gammaln(alpha).sum(axis=0) + scs.gammaln(alpha.sum(axis=0)) \
     +scs.gammaln(beta).sum(axis=0) - scs.gammaln(beta.sum(axis=0)) \
    
    for k in range(alpha.shape[1]):
        helper = np.zeros(alpha.shape[1])
        helper[k] = 1
        D_KL += alpha[k]/alpha.sum(axis=0)*((alpha-beta) * (scs.digamma(alpha) -\
                     scs.digamma((alpha+helper).sum(axis=0))[np.newaxis,:])).sum(axis=0)
     
    return D_KL


def plot_habit_learning(w, results, save_figs=False, fname=''):
    
    #plot Rho
#    plt.figure(figsize=(10,5))
    arm_cols = ['royalblue','blue']
#    for i in range(1,w.agent.nh):
#        plt.plot(w.environment.Rho[:,i,i], label="arm "+str(i), c=arm_cols[i-1], linewidth=3)
#    plt.ylim([-0.1,1.1])
#    plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
#    plt.yticks(fontsize=18)
#    plt.xticks(fontsize=18)
#    plt.xlabel("trial", fontsize=20)
#    plt.ylabel("reward probabilities", fontsize=20)
#    #plt.title("Reward probabilities for each state/bandid")
#    if save_figs:
#        plt.savefig(fname+"_Rho.svg")
#        plt.savefig(fname+"_Rho.png", bbox_inches = 'tight', dpi=300)
#    plt.show()
#    
#    plt.figure()
#    sns.barplot(data=results.T, ci=95)
#    plt.xticks([0,1],["won", "chosen", "context"])
#    plt.ylim([0,1])
#    #plt.title("Reward rate and rate of staying with choice with habit")
#    plt.yticks(fontsize=18)
#    plt.xticks(fontsize=18)
#    plt.xlabel("trial", fontsize=20)
#    plt.ylabel("rates", fontsize=20)
#    if False:
#        plt.savefig(fname+"_habit.svg")
#    plt.show()
    
    plt.figure(figsize=(10,5))
    for i in range(1,w.agent.nh):
        plt.plot(w.environment.Rho[:,i,i], label="arm "+str(i), c=arm_cols[i-1], linewidth=3)
    for t in range(1,w.agent.T):
        plt.plot(w.agent.posterior_context[:,t,1], ".", label="context", color='deeppink')
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("trial", fontsize=20)
    plt.ylabel("reward probabilities", fontsize=20)
    ax = plt.gca().twinx()
    ax.set_ylim([-0.1,1.1])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["$c_{1}$","$c_{2}$"],fontsize=18)
    ax.yaxis.set_ticks_position('right')
    #plt.title("Reward probabilities and context inference")
    if save_figs:
        plt.savefig(fname+"_Rho_c_nohabit.svg")
        plt.savefig(fname+"_Rho_c_nohabit.png", bbox_inches = 'tight', dpi=300)
    plt.show()
    
    plt.figure(figsize=(10,5))
    for i in range(1,w.agent.nh):
        plt.plot(w.environment.Rho[:,i,i], label="arm "+str(i), c=arm_cols[i-1], linewidth=3)
    for t in range(w.agent.T-1):
        plt.plot((w.actions[:,t]-1), ".", label="action", color='darkorange')
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("trial", fontsize=20)
    plt.ylabel("reward probabilities", fontsize=20)
    ax = plt.gca().twinx()
    ax.set_ylim([-0.1,1.1])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
    ax.yaxis.set_ticks_position('right')
    #plt.title("Reward probabilities and chosen actions")
    if save_figs:
        plt.savefig(fname+"_Rho_a_nohabit.svg")
        plt.savefig(fname+"_Rho_a_nohabit.png", bbox_inches = 'tight', dpi=300)
    plt.show()
    
    plt.figure(figsize=(10,5))
    for i in range(1,w.agent.nh):
        plt.plot(w.environment.Rho[:,i,i], label="arm "+str(i), c=arm_cols[i-1], linewidth=3)
    for t in range(w.agent.T-1):
        plt.plot((w.agent.posterior_policies[:,t,2]* w.agent.posterior_context[:,t]).sum(axis=1), ".", label="action", color='darkorange')
    plt.ylim([-0.1,1.1])
    plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("trial", fontsize=20)
    plt.ylabel("reward probabilities", fontsize=20)
    ax = plt.gca().twinx()
    ax.set_ylim([-0.1,1.1])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["$a_{1}$","$a_{2}$"],fontsize=18)
    ax.yaxis.set_ticks_position('right')
    #plt.title("Reward probabilities and chosen actions")
    if save_figs:
        plt.savefig(fname+"_Rho_a_nohabit.svg")
        plt.savefig(fname+"_Rho_a_nohabit.png", bbox_inches = 'tight', dpi=300)
    plt.show()
    
#    plt.figure(figsize=(10,5))
#    for i in range(1,w.agent.nh):
#        plt.plot(w.environment.Rho[:,i,i]*w.agent.perception.prior_rewards[i], label="arm "+str(i), c=arm_cols[i-1], linewidth=3)
#    for t in range(w.agent.T-1):
#        plt.plot((w.actions[:,t]-1), ".", label="action", color='g', alpha=0.5)
#    plt.ylim([-0.1,1.1])
#    plt.legend(fontsize=16, bbox_to_anchor=(1.04,1), loc="upper left", ncol=1) #bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand"
#    plt.yticks(fontsize=18)
#    plt.xticks(fontsize=18)
#    plt.xlabel("trial", fontsize=20)
#    plt.ylabel("reward probabilities", fontsize=20)
#    #plt.title("Expected utility and chosen actions")
#    if False:
#        plt.savefig(fname+"_utility_a_habit.svg")
#        plt.savefig(fname+"_utility_a_habit.png", bbox_inches = 'tight', dpi=300)
#    plt.show()