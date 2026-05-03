import numpy as np
from state import initialize_state
from measurement_model import h_x
from jacobian import build_jacobian

def run_wls(Ybus, measurements, max_iter=10):

    x = initialize_state(Ybus)

    z = np.array([m["value"] for m in measurements])
    R = np.diag([m["variance"] for m in measurements])
    R_inv = np.linalg.inv(R)

    for it in range(max_iter):

        z_est = h_x(x, Ybus, measurements)
        r = z - z_est

        H = build_jacobian(x, Ybus, measurements)

        G = H.T @ R_inv @ H

        dx = np.linalg.solve(G, H.T @ R_inv @ r)

        x = x + dx

        print(f"Iteration {it}, max update = {np.max(abs(dx))}")

        if np.max(abs(dx)) < 1e-6:
            break

    return x, r, H, G