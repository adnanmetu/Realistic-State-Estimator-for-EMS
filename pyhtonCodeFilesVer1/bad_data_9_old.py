import numpy as np

def bad_data_detection(r, H, R):

    G = H.T @ np.linalg.inv(R) @ H
    S = R - H @ np.linalg.inv(G) @ H.T

    r_norm = r / np.sqrt(np.diag(S))

    idx = np.argmax(abs(r_norm))

    if abs(r_norm[idx]) > 3:
        print("Bad data detected at index:", idx)

    return r_norm