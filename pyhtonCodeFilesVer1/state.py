import numpy as np

def initialize_state(Ybus, slack_bus=0):
    """
    State vector:
    x = [theta (excluding slack), V (all buses)]
    """

    n = Ybus.shape[0]

    theta = np.zeros(n-1)   # remove slack angle
    V = np.ones(n)

    return np.concatenate([theta, V])


def expand_state(x, slack_bus, n):
    """
    Convert reduced state → full state
    """

    theta = np.zeros(n)
    theta_indices = [i for i in range(n) if i != slack_bus]

    theta[theta_indices] = x[:n-1]

    V = x[n-1:]

    return theta, V