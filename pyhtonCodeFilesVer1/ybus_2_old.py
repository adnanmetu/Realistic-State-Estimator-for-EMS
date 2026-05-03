import numpy as np

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