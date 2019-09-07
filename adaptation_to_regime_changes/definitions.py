import numpy as np

# parameters
maximum_duration = 100
theta_good = 0.75

# Space of the random variables
a_t = {0: 'left', 1: 'right'}
s_t = {0: 'left good', 1: 'right_good'}
o_t = {0: 'good choice', 1: 'bad choice'}
d_t = list(range(maximum_duration))
regime = {0: 'regular', 1: 'irregular'}


# Duration generators
def generate_regular_d():
    duration = maximum_duration + 1
    while duration > maximum_duration:
        duration = int(np.random.binomial(40, 0.5, 1))
    return duration


def generate_irregular_d():
    duration = maximum_duration + 1
    while duration > maximum_duration:
        duration = int(np.round(np.random.lognormal(2, 1, 1))+1)
    return duration
