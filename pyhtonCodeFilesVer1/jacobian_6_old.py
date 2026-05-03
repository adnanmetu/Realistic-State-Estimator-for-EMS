import numpy as np

def build_jacobian(x, Ybus, measurements):

    n = Ybus.shape[0]

    theta = x[:n]
    V = x[n:]

    G = Ybus.real
    B = Ybus.imag

    H = []

    for m in measurements:
        row = np.zeros(2*n)
        i = m["bus"] - 1

        if m["type"] == "V":
            row[n+i] = 1

        elif m["type"] == "theta":
            row[i] = 1

        elif m["type"] == "P_inj":
            for j in range(n):

                if i == j:
                    for k in range(n):
                        row[i] += V[i]*V[k]*(
                            -G[i,k]*np.sin(theta[i]-theta[k]) +
                             B[i,k]*np.cos(theta[i]-theta[k])
                        )
                else:
                    row[j] = V[i]*V[j]*(
                        G[i,j]*np.sin(theta[i]-theta[j]) -
                        B[i,j]*np.cos(theta[i]-theta[j])
                    )

        H.append(row)

    return np.array(H)