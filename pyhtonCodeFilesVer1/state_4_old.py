import numpy as np

def initialize_state(Ybus):

    n = Ybus.shape[0]

    theta = np.zeros(n)
    V = np.ones(n)

    return np.concatenate([theta, V])