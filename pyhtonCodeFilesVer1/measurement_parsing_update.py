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