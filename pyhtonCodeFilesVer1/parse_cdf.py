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

        if mode == "BUS":
            try:
                bus_id = int(line[0:4])
                V = float(line[27:33])
                theta = float(line[33:40])

                buses.append({
                    "id": bus_id,
                    "V": V,
                    "theta": np.deg2rad(theta)
                })
            except:
                continue

        elif mode == "BRANCH":
            try:
                from_bus = int(line[0:4])
                to_bus = int(line[5:9])
                R = float(line[19:29])
                X = float(line[29:40])
                B = float(line[40:50])

                branches.append({
                    "from": from_bus,
                    "to": to_bus,
                    "R": R,
                    "X": X,
                    "B": B
                })
            except:
                continue

    return buses, branches