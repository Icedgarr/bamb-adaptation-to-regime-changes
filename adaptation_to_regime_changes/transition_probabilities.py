import numpy as np
from definitions import theta_good, maximum_duration


# action probabilities p(a_t)
p_a_t = np.array([0.5, 0.5]).reshape(2, 1)


# state transitions p(s_t|s_t_1, d_t)
def p_st_st1_dt(d_t):
    if d_t == 0:
        return np.array([[0, 1], [1, 0]])
    else:
        return np.array([[1, 0], [0, 1]])


p_s_t_d0 = np.array([[0, 1], [1, 0]])
p_s_t_dother = np.array([[1, 0], [0, 1]])


# observations llh p(o_t| s_t, a_t)
def p_ot_st_at(a_t):
    if a_t == 0:
        return np.array([[theta_good, 1 - theta_good], [theta_good, 1 - theta_good]])
    else:
        return np.array([[1 - theta_good, theta_good], [theta_good, 1 - theta_good]])


p_o_t_s_t_a_t0 = np.array([[[theta_good, 1 - theta_good], [1 - theta_good, theta_good]], 
                          [[1 - theta_good, theta_good], [theta_good, 1 - theta_good]]])


# duration transitions p(d_t| d_t_1)
def p_dt_d_t1(d_t_1, distribution):
    if d_t_1 == 0:
        return distribution
    else:
        return np.concatenate([np.array([0]*maximum_duration).reshape(maximum_duration, 1),
                               np.identity(maximum_duration)], axis=1)

