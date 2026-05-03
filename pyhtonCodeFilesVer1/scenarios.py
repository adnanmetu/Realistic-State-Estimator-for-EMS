import numpy as np
import copy

def add_noise(measurements, sigma=0.01):
    noisy = copy.deepcopy(measurements)

    for m in noisy:
        m["value"] += np.random.normal(0, sigma)

    return noisy


def add_bad_data(measurements):
    corrupted = copy.deepcopy(measurements)

    idx = np.random.randint(len(corrupted))
    corrupted[idx]["value"] *= 5  # large error

    print("Injected bad data at index:", idx)

    return corrupted


def remove_pmu(measurements):
    return [m for m in measurements if m["type"] != "theta"]