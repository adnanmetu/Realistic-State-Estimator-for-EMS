import numpy as np

def h_x(x, Ybus, measurements):

    n = Ybus.shape[0]

    theta = x[:n]
    V = x[n:]

    G = Ybus.real
    B = Ybus.imag

    z_est = []

    for m in measurements:
        i = m["bus"] - 1

        if m["type"] == "V":
            z_est.append(V[i])

        elif m["type"] == "theta":
            z_est.append(theta[i])

        elif m["type"] == "P_inj":
            Pi = 0
            for j in range(n):
                Pi += V[i]*V[j]*(
                    G[i,j]*np.cos(theta[i]-theta[j]) +
                    B[i,j]*np.sin(theta[i]-theta[j])
                )
            z_est.append(Pi)

        elif m["type"] == "Q_inj":
            Qi = 0
            for j in range(n):
                Qi += V[i]*V[j]*(
                    G[i,j]*np.sin(theta[i]-theta[j]) -
                    B[i,j]*np.cos(theta[i]-theta[j])
                )
            z_est.append(Qi)

        else:
            z_est.append(0)  # placeholder for flows/PMU

    return np.array(z_est)