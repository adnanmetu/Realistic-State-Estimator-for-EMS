import numpy as np
from state import expand_state

def build_jacobian(x, Ybus, measurements, slack_bus):

    n = Ybus.shape[0]
    theta, V = expand_state(x, slack_bus, n)

    G = Ybus.real
    B = Ybus.imag

    # mapping reduced theta indices
    theta_map = [i for i in range(n) if i != slack_bus]

    H = []

    for m in measurements:
        i = m["bus"] - 1

        row = np.zeros((len(x)))

        # ---------------- Voltage ----------------
        if m["type"] == "V":
            row[len(theta_map) + i] = 1

        # ---------------- PMU Angle ----------------
        elif m["type"] == "theta":
            if i != slack_bus:
                idx = theta_map.index(i)
                row[idx] = 1

        # ---------------- P Injection ----------------
        elif m["type"] == "P_inj":

            for j in range(n):

                if j != slack_bus:
                    col_theta = theta_map.index(j)

                    if i == j:
                        sum_term = 0
                        for k in range(n):
                            sum_term += V[i]*V[k]*(
                                -G[i,k]*np.sin(theta[i]-theta[k]) +
                                 B[i,k]*np.cos(theta[i]-theta[k])
                            )
                        row[col_theta] = sum_term
                    else:
                        row[col_theta] = V[i]*V[j]*(
                            G[i,j]*np.sin(theta[i]-theta[j]) -
                            B[i,j]*np.cos(theta[i]-theta[j])
                        )

            # dP/dV
            for j in range(n):
                col_V = len(theta_map) + j

                if i == j:
                    sum_term = 0
                    for k in range(n):
                        sum_term += V[k]*(
                            G[i,k]*np.cos(theta[i]-theta[k]) +
                            B[i,k]*np.sin(theta[i]-theta[k])
                        )
                    row[col_V] = sum_term
                else:
                    row[col_V] = V[i]*(
                        G[i,j]*np.cos(theta[i]-theta[j]) +
                        B[i,j]*np.sin(theta[i]-theta[j])
                    )

        # ---------------- Q Injection ----------------
        elif m["type"] == "Q_inj":

            for j in range(n):

                if j != slack_bus:
                    col_theta = theta_map.index(j)

                    if i == j:
                        sum_term = 0
                        for k in range(n):
                            sum_term += V[i]*V[k]*(
                                G[i,k]*np.cos(theta[i]-theta[k]) +
                                B[i,k]*np.sin(theta[i]-theta[k])
                            )
                        row[col_theta] = -sum_term
                    else:
                        row[col_theta] = -V[i]*V[j]*(
                            G[i,j]*np.cos(theta[i]-theta[j]) +
                            B[i,j]*np.sin(theta[i]-theta[j])
                        )

            # dQ/dV
            for j in range(n):
                col_V = len(theta_map) + j

                if i == j:
                    sum_term = 0
                    for k in range(n):
                        sum_term += V[k]*(
                            G[i,k]*np.sin(theta[i]-theta[k]) -
                            B[i,k]*np.cos(theta[i]-theta[k])
                        )
                    row[col_V] = sum_term
                else:
                    row[col_V] = V[i]*(
                        G[i,j]*np.sin(theta[i]-theta[j]) -
                        B[i,j]*np.cos(theta[i]-theta[j])
                    )

        H.append(row)

    return np.array(H)