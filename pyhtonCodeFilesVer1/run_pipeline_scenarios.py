import numpy as np

from parse_cdf import parse_cdf
from ybus import build_ybus
from measurements import parse_measurements
from ac_wls import run_wls
from observability import check_observability
from bad_data import bad_data_detection

# 🔥 CHANGE PATHS HERE
cdf_file = r"D:\your_path\ieee_cdf_sample.dat"
meas_file = r"D:\your_path\measure.dat"

# 1. Load data
buses, branches = parse_cdf(cdf_file)

# 2. Build Ybus
Ybus = build_ybus(buses, branches)

# 3. Load measurements
measurements = parse_measurements(meas_file)

# 4. Run WLS
x, r, H, G = run_wls(Ybus, measurements)

# 5. Observability
check_observability(H)

# 6. Bad data detection
R = np.diag([m["variance"] for m in measurements])
bad_data_detection(r, H, R)

print("\nFinal State Vector:")
print(x)