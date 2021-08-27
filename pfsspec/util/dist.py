import numpy as np

def get_random_dist(dist, random_state):
    if dist is None:
        return None
    elif dist == 'normal':
        return random_state.normal
    elif dist == 'uniform':
        return random_state.uniform
    elif dist == 'lognormal':
        return random_state.lognormal
    elif dist == 'beta':
        return random_state.beta
    else:
        raise NotImplementedError()