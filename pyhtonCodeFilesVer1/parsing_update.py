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