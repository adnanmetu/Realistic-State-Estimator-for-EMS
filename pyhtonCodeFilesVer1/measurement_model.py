import numpy as np
from state import expand_state

def h_x(x, Ybus, measurements, slack_bus):

    n = Ybus.shape[0]
    theta, V = expand_state(x, slack_bus, n)

    G = Ybus.real
    B = Ybus.imag

    z_est = []

    for m in measurements:
        i = m["bus"] - 1

        # ---------------- Voltage ----------------
        if m["type"] == "V":
            z_est.append(V[i])

        # ---------------- PMU Angle ----------------
        elif m["type"] == "theta":
            z_est.append(theta[i])

        # ---------------- P Injection ----------------
        elif m["type"] == "P_inj":
            Pi = 0
            for j in range(n):
                Pi += V[i]*V[j]*(
                    G[i,j]*np.cos(theta[i]-theta[j]) +
                    B[i,j]*np.sin(theta[i]-theta[j])
                )
            z_est.append(Pi)

        # ---------------- Q Injection ----------------
        elif m["type"] == "Q_inj":
            Qi = 0
            for j in range(n):
                Qi += V[i]*V[j]*(
                    G[i,j]*np.sin(theta[i]-theta[j]) -
                    B[i,j]*np.cos(theta[i]-theta[j])
                )
            z_est.append(Qi)

    return np.array(z_est)