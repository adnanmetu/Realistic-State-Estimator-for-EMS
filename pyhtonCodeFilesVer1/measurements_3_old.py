def parse_measurements(file_path):

    measurements = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    current_type = None

    order = ["V", "theta", "P_inj", "Q_inj", "P_flow", "Q_flow", "PMU"]

    while i < len(lines):
        try:
            count = int(lines[i].strip())
            i += 1

            if current_type is None:
                current_type = order[0]
            else:
                idx = order.index(current_type)
                current_type = order[idx+1] if idx+1 < len(order) else None

            for _ in range(count):
                parts = lines[i].split()

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