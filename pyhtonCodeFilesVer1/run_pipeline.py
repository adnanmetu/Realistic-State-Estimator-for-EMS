from parse_cdf import parse_cdf
from ybus import build_ybus
from measurements import parse_measurements
from ac_wls import run_wls
from observability import check_observability
from bad_data import bad_data_detection
from scenarios import add_noise, add_bad_data, remove_pmu
from plots import plot_convergence, plot_residuals

#################################################
cdf_file = r"D:\HigherStudies\metu\MS Electrical and Electronics Engineering\ee574\Project Material\ieee_cdf_sample.dat"
meas_file = r"D:\HigherStudies\metu\MS Electrical and Electronics Engineering\ee574\Project Material\measure.dat"
#########################

# Load
buses, branches = parse_cdf(cdf_file)
Ybus = build_ybus(buses, branches)
measurements = parse_measurements(meas_file)

# -------- SCENARIO 1 --------
print("\nSCENARIO 1: Clean SCADA")
x, r, H, G, dx_hist = run_wls(Ybus, measurements)
plot_convergence(dx_hist)
plot_residuals(r)

# -------- SCENARIO 2 --------
print("\nSCENARIO 2: SCADA + Noise")
meas_noise = add_noise(measurements)

############ work around ############
valid_bus_ids = {bus["id"] for bus in buses}
measurements = [
    m for m in measurements
    if m["bus"] in valid_bus_ids
    and ("to" not in m or m["to"] in valid_bus_ids)
]

######## work around #####
implemented_types = {"V", "theta", "P_inj", "Q_inj"}
measurements = [m for m in measurements if m["type"] in implemented_types]
#####################################
x, r, H, G, dx_hist = run_wls(Ybus, meas_noise)

# -------- SCENARIO 3 --------
print("\nSCENARIO 3: Bad Data + Missing PMU")
meas_bad = add_bad_data(meas_noise)
meas_bad = remove_pmu(meas_bad)

x, r, H, G, dx_hist = run_wls(Ybus, meas_bad)

check_observability(H)

R = np.diag([m["variance"] for m in meas_bad])
bad_data_detection(r, H, R)