import numpy as np

def parse_cdf(file_path):
    buses = []
    branches = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    mode = None

    for line in lines:

        if "BUS DATA" in line:
            mode = "BUS"
            continue

        elif "BRANCH DATA" in line:
            mode = "BRANCH"
            continue

        elif "-999" in line:
            mode = None
            continue

        # ---------------- BUS DATA ----------------
        if mode == "BUS":
            if not line.strip():
                continue

            parts = line.split()

            # skip malformed lines
            if len(parts) < 8:
                continue

            try:
                bus_id = int(parts[0])

                # voltage magnitude and angle are usually near columns 7 and 8
                V = float(parts[6])
                theta = float(parts[7])

                buses.append({
                    "id": bus_id,
                    "V": V,
                    "theta": np.deg2rad(theta)
                })

            except:
                continue

        # ---------------- BRANCH DATA ----------------
        elif mode == "BRANCH":
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 6:
                continue

            try:
                branches.append({
                    "from": int(parts[0]),
                    "to": int(parts[1]),
                    "R": float(parts[6]),
                    "X": float(parts[7]),
                    "B": float(parts[8])
                })

            except:
                continue

    return buses, branches
############################################
def parse_measurements(file_path):

    measurements = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    order = ["V", "theta", "P_inj", "Q_inj", "P_flow", "Q_flow", "PMU"]

    i = 0
    type_idx = 0

    while i < len(lines):

        if type_idx >= len(order):
            break

        try:
            count = int(lines[i].strip())
            current_type = order[type_idx]
            type_idx += 1
            i += 1

            for _ in range(count):

                parts = lines[i].split()

                if len(parts) < 3:
                    i += 1
                    continue

                measurements.append({
                    "type": current_type,
                    "bus": int(parts[0]),
                    "value": float(parts[1]),
                    "variance": float(parts[2])
                })

                i += 1

        except:
            i += 1

    return measurements
#################################################
cdf_file = r"D:\HigherStudies\metu\MS Electrical and Electronics Engineering\ee574\Project Material\ieee_cdf_sample.dat"
meas_file = r"D:\HigherStudies\metu\MS Electrical and Electronics Engineering\ee574\Project Material\measure.dat"
#########################
#measurements = parse_measurements(meas_file)

#valid_bus_ids = {bus["id"] for bus in buses}
#measurements = [m for m in measurements if m["bus"] in valid_bus_ids]

#print("Measurement types:", sorted(set(m["type"] for m in measurements)))
#print("Number of measurements:", len(measurements))


buses, branches = parse_cdf(cdf_file)

measurements = parse_measurements(meas_file)

############ work around ############
valid_bus_ids = {bus["id"] for bus in buses}
measurements = [
    m for m in measurements
    if m["bus"] in valid_bus_ids
    and ("to" not in m or m["to"] in valid_bus_ids)
]
#####################################

def build_ybus(buses, branches):

    n = len(buses)
    Ybus = np.zeros((n, n), dtype=complex)

    bus_map = {bus["id"]: i for i, bus in enumerate(buses)}

    for br in branches:

        i = bus_map[br["from"]]
        j = bus_map[br["to"]]

        z = complex(br["R"], br["X"])
        y = 1 / z if z != 0 else 0

        Ybus[i, j] -= y
        Ybus[j, i] -= y

        Ybus[i, i] += y + complex(0, br["B"]/2)
        Ybus[j, j] += y + complex(0, br["B"]/2)

    return Ybus
#########################################
def run_wls(Ybus, measurements, slack_bus=0, max_iter=15):

    x = initialize_state(Ybus, slack_bus)

    z = np.array([m["value"] for m in measurements])
    R = np.diag([m["variance"] for m in measurements])
    R_inv = np.linalg.inv(R)

    for it in range(max_iter):

        z_est = h_x(x, Ybus, measurements, slack_bus)
        r = z - z_est

        H = build_jacobian(x, Ybus, measurements, slack_bus)

        G = H.T @ R_inv @ H

        dx = np.linalg.solve(G, H.T @ R_inv @ r)

        x = x + dx

        print(f"Iteration {it}: max(dx) = {np.max(abs(dx))}")

        if np.max(abs(dx)) < 1e-6:
            print("✅ Converged")
            break

    return x, r, H, G
################################
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
################################
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
#############################

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
####################################################
Ybus = build_ybus(buses, branches)
######## work around #####
implemented_types = {"V", "theta", "P_inj", "Q_inj"}
measurements = [m for m in measurements if m["type"] in implemented_types]
#x, r, H, G = run_wls(Ybus, measurements)

#print("Number of states:", len(initialize_state(Ybus, 0)))
#print("Number of measurements:", len(measurements))
#print("Rank of H:", np.linalg.matrix_rank(H))
#print("H shape:", H.shape)