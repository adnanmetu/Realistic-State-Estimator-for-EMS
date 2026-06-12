
"""
==============================================================================
EE574 Term Project – Realistic AC Weighted Least Squares (WLS) State Estimator
==============================================================================
measure.dat block order (8 blocks, terminated by count=0):
  Block 1 : |V|      Voltage magnitude        (single bus,  pu)
  Block 2 : θ_V      Voltage angle            (single bus,  rad or deg)
  Block 3 : P_inj    Active power injection   (single bus,  pu)
  Block 4 : Q_inj    Reactive power injection (single bus,  pu)
  Block 5 : P_flow   Active power flow        (two buses,   pu)
  Block 6 : Q_flow   Reactive power flow      (two buses,   pu)
  Block 7 : |I|      Current magnitude        (two buses,   pu)
  Block 8 : θ_I      Current angle            (two buses,   rad)

Each data line format:
  Single-bus  →  BUS_I        value  sigma  status
  Two-bus     →  BUS_I BUS_J  value  sigma  status
  status = 0 → active measurement, anything else → excluded

Technical requirements implemented:
  1. AC-WLS State Estimation               – iterative Newton / Gauss-Newton
  2. PMU measurement integration           – V_ang buses from Block 2 treated as PMU;
                                             both |V| and θ included with PMU sigma
  3. Observability analysis                – numerical SVD rank check of Jacobian H
  4. Bad data detection and identification – Largest Normalised Residual (LNR) test

Dependencies: numpy, scipy, tabulate   →   pip install numpy scipy tabulate
Run         : python state_estimator.py
==============================================================================
"""

import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
import re

# ── optional pretty-print ─────────────────────────────────────────────────────
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("[WARN] tabulate not installed – tables will be plain text.")

def banner(title):
    """Print a section banner."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def ptable(rows, headers):
    """Unified table printer: tabulate if available, plain otherwise."""
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt='grid'))
    else:
        print("  " + " | ".join(str(h) for h in headers))
        print("  " + "-" * 90)
        for r in rows:
            print("  " + " | ".join(str(x) for x in r))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – IEEE CDF NETWORK FILE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_ieee_cdf(filename: str):
    """Parse an IEEE Common Data Format (CDF) file."""
    """
        Parse an IEEE Common Data Format (CDF) file.

        CDF file structure
        ──────────────────
        Header line : contains base MVA (first float token in range 10–10000)

        BUS DATA FOLLOWS … -999
          Fixed-width columns (1-indexed):
            1-4   Bus number
            6-17  Bus name
            25-26 Bus type   0/1=PQ, 2=PV, 3=slack
            27-33 |V| pu     (load-flow solution – used as ground truth)
            34-40 θ degrees
            41-49 P load MW
            50-58 Q load MVAR
            59-67 P gen  MW
            68-74 Q gen  MVAR
            76-83 Base kV
           107-114 Shunt G pu
           115-122 Shunt B pu

        BRANCH DATA FOLLOWS … -999
          Fixed-width columns:
            1-4   From bus
            5-9   To bus
            20-29 R pu
            30-40 X pu
            41-50 B pu (total line charging susceptance)
            77-82 Transformer turns ratio (0 → plain line, use 1.0)
            84-90 Phase-shift angle degrees

        Returns
        -------
        buses    : list[dict]
        branches : list[dict]
        base_mva : float
        """
    banner("SECTION 1 – PARSING IEEE CDF NETWORK FILE")
    print(f"[CDF] File: {filename}")

    buses = []
    branches = []
    base_mva = 100.0    #default value
    mode = None  # 'bus' | 'branch' | None
    with open(filename, 'r') as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip('\n')
            #print(f"line number {lineno}: {line}")  # debugging
            # ── block sentinels ───────────────────────────────────────────────
            if 'BUS DATA FOLLOWS' in line:
                mode = 'bus'
                print(f"[CDF] Line {lineno:3d}: → BUS DATA block")
                continue
            if 'BRANCH DATA FOLLOWS' in line:
                mode = 'branch'
                print(f"[CDF] Line {lineno:3d}: → BRANCH DATA block")
                continue
            if line.strip() == '-999':
                print(f"[CDF] Line {lineno:3d}: end-of-block (-999)")
                mode = None
                continue
            if line.strip() == '':
                continue

            # ── header line ───────────────────────────────────────────────────

            #if mode is None:

            if lineno == 1:
                #####################################
                print("mode is None")
                print(line)
                try:
                    mb = float(line[30:37])
                    #print("printing line[30:37]")  #debugging
                    print(line[30:37])
                    if 10.0 <= mb <= 10_000.0:
                        base_mva = mb
                        print(f"[CDF] Base MVA = {base_mva}")
                    else:
                        print("MVA base is not defined properly")
                        print(mb)
                        print("Default Base MVA 100.0 is selected")
                except ValueError:
                    print("error in reading first line[30:37]")
                    print("Default Base MVA 100.0 is selected")
                    pass

                #################################
                """
                for tok in line.split():
                    try:
                        val = float(tok)
                        if 10.0 <= val <= 10_000.0:
                            base_mva = val
                            print(f"[CDF] Base MVA = {base_mva}")
                            break
                    except ValueError:
                        print(base_mva)
                        print("MVA base is not defined properly")
                        pass
                continue
                """
                ##################
            # ── bus record ────────────────────────────────────────────────────
            if mode == 'bus':
                try:
                    # Primary: fixed-width parse
                    bus_num = int(line[0:4])
                    bus_name = line[5:17].strip()
                    bus_type = int(line[24:26])
                    v_mag = float(line[27:33])
                    v_ang = float(line[33:40])
                    p_load = float(line[40:49])
                    q_load = float(line[49:58])
                    p_gen = float(line[59:67])
                    q_gen = float(line[67:74])
                    base_kv = float(line[76:83])
                    g_sh = float(line[106:114]) if len(line) > 106 else 0.0
                    b_sh = float(line[114:122]) if len(line) > 114 else 0.0
                except (ValueError, IndexError):
                    # Primary: fixed-width parse
                    bus_num = int(line[0:4])
                    bus_name = ' ' #line[5:17].strip()
                    #print("debugging bus type")
                    #print(line[24:26])
                    #print(line[24:27])
                    #print(line[25:27])
                    bus_type = int(line[24:27])
                    v_mag = float(line[27:33])
                    v_ang = float(line[33:40])
                    p_load = 0.0   #float(line[40:49])
                    q_load = 0.0    #float(line[49:58])
                    p_gen = 0.0     #float(line[59:67])
                    q_gen = 0.0     #float(line[67:74])
                    base_kv = 1.0   #float(line[76:83])
                    g_sh = float(line[106:114]) if len(line) > 106 else 0.0
                    try:
                        b_sh = float(line[114:122]) if len(line) > 114 else 0.0
                    except (ValueError):
                        b_sh = float(line[115:123]) if len(line) > 114 else 0.0
                """    
                except (ValueError, IndexError):
                    # Fallback: whitespace-split
                    p = line.split()
                    bus_num = int(p[0]);
                    bus_name = p[1]
                    bus_type = int(p[4]);
                    v_mag = float(p[5])
                    v_ang = float(p[6]);
                    p_load = float(p[7])
                    q_load = float(p[8]);
                    p_gen = float(p[9])
                    q_gen = float(p[10]);
                    base_kv = float(p[11])
                    g_sh = float(p[15]) if len(p) > 106 else 0.0
                    b_sh = float(p[16]) if len(p) > 114 else 0.0
                """
                buses.append({
                    'num': bus_num,
                    'name': bus_name,
                    'type': bus_type,
                    'v_mag': v_mag,
                    'v_ang': np.deg2rad(v_ang),  # stored in radians
                    'p_load': p_load / base_mva,
                    'q_load': q_load / base_mva,
                    'p_gen': p_gen / base_mva,
                    'q_gen': q_gen / base_mva,
                    'base_kv': base_kv,
                    'g_sh': g_sh,
                    'b_sh': b_sh,
                })

            # ── branch record ─────────────────────────────────────────────────
            elif mode == 'branch':
                try:
                    fr = int(line[0:4]);
                    to = int(line[5:9])
                    R = float(line[19:29]);
                    X = float(line[29:40])
                    B = float(line[40:50])
                    tap = float(line[76:82]) if line[76:82].strip() else 0.0
                    phs = float(line[83:90]) if len(line) > 83 and line[83:90].strip() else 0.0
                except (ValueError, IndexError):
                    p = line.split()
                    fr = int(p[0]);
                    to = int(p[1]);
                    R = float(p[2]);
                    X = float(p[3]);
                    B = float(p[4])
                    """
                    try:
                        R = float(p[6]);
                        X = float(p[7]);
                        B = float(p[8])
                    except (ValueError):
                        R = float(p[2]);
                        X = float(p[3]);
                        B = float(p[4])
                    """
                    tap = float(p[14]) if len(p) > 14 and p[14] != '0.0' else 0.0
                    phs = float(p[15]) if len(p) > 15 else 0.0

                branches.append({
                    'fr': fr, 'to': to,
                    'R': R, 'X': X, 'B': B,
                    'tap': tap,
                    'phs': phs,
                })

    print(f"\n[CDF] ✓ Buses    parsed : {len(buses)}")
    print(f"[CDF] ✓ Branches parsed : {len(branches)}")

    TYPE_LABEL = {0: 'PQ', 1: 'PQ', 2: 'PV', 3: 'Slack'}
    headers = ['Bus', 'Name', 'Type', '|V|pu', 'θ°', 'PLpu', 'QLpu', 'PGpu', 'QGpu']
    rows = [[b['num'], b['name'], TYPE_LABEL.get(b['type'], '?'),
             f"{b['v_mag']:.4f}", f"{np.rad2deg(b['v_ang']):.2f}",
             f"{b['p_load']:.4f}", f"{b['q_load']:.4f}", f"{b['p_gen']:.4f}", f"{b['q_gen']:.4f}"]
            for b in buses]
    ptable(rows, headers)

    return buses, branches, base_mva
######## cdf parsing end here ##################

########### measurement parsing start from here ############
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – MEASUREMENT FILE PARSER  (measure.dat)
# ══════════════════════════════════════════════════════════════════════════════
def parse_measure_dat(filename: str):
    """
    Parse measure.dat – SCADA / PMU measurement input file.

    Block order
    ───────────
    Block 1 : V_mag   Voltage magnitude        1 bus index per line  (pu)
    Block 2 : V_ang   Voltage angle (PMU θ)    1 bus index per line  (rad)
    Block 3 : P_inj   Active power injection   1 bus index per line  (pu)
    Block 4 : Q_inj   Reactive power injection 1 bus index per line  (pu)
    Block 5 : P_flow  Active power flow        2 bus indices per line (pu)
    Block 6 : Q_flow  Reactive power flow      2 bus indices per line (pu)
    Block 7 : I_mag   Current magnitude        2 bus indices per line (pu)
    Block 8 : I_ang   Current angle            2 bus indices per line (rad)

    Data line format
    ────────────────
    Single-bus :  BUS_I         value  sigma  status
    Two-bus    :  BUS_I  BUS_J  value  sigma  status

    status = 0  → active (included in estimation)
    status ≠ 0  → excluded

    PMU identification
    ──────────────────
    Buses appearing in Block 2 (V_ang) are treated as PMU buses.
    Their sigma values are taken directly from the file.
    In the estimation, a PMU bus contributes BOTH a V_mag measurement
    (from Block 1 if present, else synthetic from the same sigma) AND
    a V_ang measurement (from Block 2) – exactly what a PMU provides.

    Returns
    -------
    measurements : list[dict]
        keys: type, i, j, value, sigma, status, unit
    """
    banner("SECTION 2 – PARSING MEASUREMENT FILE  (measure.dat)")
    print(f"[MEAS] File: {filename}")

    # ── Block descriptor table ───────────────────────────────────────
    # (mtype, two_bus_flag, unit_hint)
    BLOCK_DESCRIPTORS = [
        ('V_mag', False, 'pu'),  # Block 1: voltage magnitude
        ('V_ang', False, 'rad'),  # Block 2: voltage angle
        ('P_inj', False, 'pu'),  # Block 3: active power injection
        ('Q_inj', False, 'pu'),  # Block 4: reactive power injection
        ('P_flow', True, 'pu'),  # Block 5: active power flow
        ('Q_flow', True, 'pu'),  # Block 6: reactive power flow
        ('I_mag', True, 'pu'),  # Block 7: current magnitude
        ('I_ang', True, 'rad'),  # Block 8: current angle
    ]

    measurements = []
    # Read all non-empty token lines into a flat list for sequential processing
    with open(filename, 'r') as fh:
        all_lines = []
        for ln in fh:
            stripped = ln.strip()
            if stripped and not stripped.startswith('#'):
                all_lines.append(stripped.split())

    ptr = 0  # pointer into all_lines
    block_idx = 0  # which block (0-based)
    n_measurements = 0
    N_prev = 0

    while ptr < len(all_lines) and block_idx < len(BLOCK_DESCRIPTORS):
        #print(f"length of lines : {len(all_lines)}")
        # ── read the count N for this block ───────────────────────────────────
        #print(f"length of all line is {len(all_lines)}")  # debugging
        #print(f"length of line {ptr} is {len(all_lines[ptr])}")   #debugging
        #print(all_lines[ptr])
        #print(all_lines[ptr+1])
        pattern = r'-?\d+\.?\d*'
        #filter only integers or floats
        s_n = [
            match.group()
            for item in all_lines[ptr]
            if (match := re.match(pattern, item))
        ]
        all_lines[ptr] = s_n
        #print(all_lines[ptr])
        #for text in all_lines[ptr]:
        #    print(text)
        #    s_n.append(re.findall(pattern, text))
        ################
        #print(all_lines[ptr+1])
        #print((all_lines[ptr]).strip())
        try:
            check_length = len(all_lines[ptr])  #check for end of file or false data
            while(ptr < (len(all_lines)) and check_length != 1):    #check for end of file or false (short) data
                ptr += 1
                # filter only integers or floats
                s_n = [
                    match.group()
                    for item in all_lines[ptr]
                    if (match := re.match(pattern, item))
                ]
                all_lines[ptr] = s_n
                check_length = len(all_lines[ptr])
            N = int(all_lines[ptr][0])
            n_measurements += 1
            #print(f"number of block is {n_measurements}")  #for debugging purpose

            if(n_measurements == 2):
                if(N > N_prev or abs(float(all_lines[ptr+1][1])) > 0.35):     #check for missing voltage angle measurement
                    block_idx += 1
                    n_measurements += 1

            if (n_measurements == 4):   #check for missing section rective power injection
                try:
                    # filter only integers or floats
                    s_n = [
                        match.group()
                        for item in all_lines[ptr+1]
                        if (match := re.match(pattern, item))
                    ]
                    #all_lines[ptr] = s_n
                except (ValueError):
                    print("There is some error in the file or the file is corrupted")
                if (len(s_n) == 5):
                    block_idx += 1
                    n_measurements += 1

            #format_lenth = len(all_lines[ptr])
            ######### need to check for missing entire power injection block
            if (n_measurements == 3 and len(all_lines[ptr+1]) == 5):
                block_idx += 2
                n_measurements += 2
            #debugging
            #print("The bottom line is for debugging purpose.")
            #print(f"N_prev = {N_prev}, N = {N}, n_measurements = {n_measurements}, block_idx = {block_idx}")
            N_prev = N
           # print(f"N_prev updated = {N_prev}")
        except (ValueError, IndexError):
            print(f"[MEAS] Cannot read block count at token-line {ptr}, "
                  f"tokens={all_lines[ptr]} – stopping")
            break
        if(ptr < (len(all_lines) - 1)):
            ptr += 1
            # filter only integers or floats
            s_n = [
                match.group()
                for item in all_lines[ptr]
                if (match := re.match(pattern, item))
            ]
            all_lines[ptr] = s_n
        # N=0 is the end-of-file sentinel
        if N == 0 and ptr == (len(all_lines) - 1):
            print(f"[MEAS] Count = 0 at block {block_idx + 1} → EOF sentinel")
            break
        if N == 0 and ptr < (len(all_lines) - 1):
            while(len(ptr < len(all_lines) and all_lines[ptr]) != 1):
                ptr += 1
                # filter only integers or floats
                s_n = [
                    match.group()
                    for item in all_lines[ptr]
                    if (match := re.match(pattern, item))
                ]
                all_lines[ptr] = s_n
                #print(f"line is {all_lines[ptr]}")
        mtype, two_bus, unit = BLOCK_DESCRIPTORS[block_idx]
        """
        print(f"\n[MEAS] ── Block {block_idx + 1}/8 ──────────────────────────────")
        print(f"[MEAS]    Type    : {mtype}")
        print(f"[MEAS]    Two-bus : {two_bus}")
        print(f"[MEAS]    Unit    : {unit}")
        print(f"[MEAS]    Count N : {N}")
        """
        block_recs = []
        for rec_idx in range(N):
            if ptr >= len(all_lines):
                print(f"[MEAS] WARNING: file ended before all {N} records "
                      f"in block {block_idx + 1} were read")
                break
            ####################
            # filter only integers or floats
            s_n = [
                match.group()
                for item in all_lines[ptr]
                if (match := re.match(pattern, item))
            ]
            all_lines[ptr] = s_n
            #print(all_lines[ptr])
            ####################
            tokens = all_lines[ptr]
            #print(f"length of line is {len(all_lines[ptr])}")   #debugging

            try:
                if two_bus:
                    # Two bus numbers, then value / sigma / status
                    bus_i = int(tokens[0])
                    bus_j = int(tokens[1])
                    value = float(tokens[2])
                    sigma = float(tokens[3])
                    status = int(tokens[4])
                    loc_str = f"bus {bus_i:3d} → {bus_j:3d}"
                else:
                    # One bus number, then value / sigma / status
                    bus_i = int(tokens[0])
                    bus_j = None
                    value = float(tokens[1])
                    sigma = float(tokens[2])
                    status = int(tokens[3])
                    loc_str = f"bus {bus_i:3d}        "
            except (IndexError, ValueError) as err:
                print(f"   [MEAS] Parse error (rec {rec_idx + 1}): "
                      f"tokens={tokens}  →  {err}  – skipped")
                continue

            ptr += 1
            # filter only integers or floats
            s_n = [
                match.group()
                for item in all_lines[ptr]
                if (match := re.match(pattern, item))
            ]
            all_lines[ptr] = s_n

            rec = {
                'type': mtype,
                'i': bus_i,
                'j': bus_j,
                'value': value,
                'sigma': sigma,
                'status': status,
                'unit': unit,
            }
            block_recs.append(rec)
            measurements.append(rec)

            # Per-measurement verbose print
            status_str = "active  " if status == 0 else f"EXCLUDED(status={status})"
            """
            print(f"   Rec {rec_idx + 1:2d}: {loc_str}  "
                  f"value={value:+11.5f} {unit:3s}  "
                  f"σ={sigma:.4f}  [{status_str}]")

        print(f"[MEAS] Block {block_idx + 1} done: {len(block_recs)} record(s) read")
            """
        block_idx += 1

        # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[MEAS] ✓ Total records parsed : {len(measurements)}")
    type_counts = {}
    for m in measurements:
        type_counts[m['type']] = type_counts.get(m['type'], 0) + 1
    print("[MEAS] Breakdown by block type:")
    for btype, cnt in sorted(type_counts.items()):
        print(f"         {btype:8s} : {cnt:3d}")
    active = sum(1 for m in measurements if m['status'] == 0)
    excluded = len(measurements) - active
    print(f"[MEAS] Active (status=0) : {active}")
    print(f"[MEAS] Excluded          : {excluded}")

    # Full parsed table
    headers = ['#', 'Type', 'Unit', 'Bus_i', 'Bus_j',
               'Value', 'Sigma', 'Status']
    rows = []
    for k, m in enumerate(measurements):
        rows.append([
            k, m['type'], m['unit'], m['i'],
            m['j'] if m['j'] is not None else '—',
            f"{m['value']:+.5f}", f"{m['sigma']:.4f}", m['status'],
        ])
    ptable(rows, headers)

    return measurements
############# measurement parsing ends here #################################

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Y-BUS ADMITTANCE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_ybus(buses, branches):
    """
        Build the complex n×n bus admittance matrix Y_bus.

        π-model stamp for branch i→j with complex tap t = tap·e^(jφ):
            Y[i,i] +=  y_s/|t|²  +  j·b_c
            Y[j,j] +=  y_s        +  j·b_c
            Y[i,j] -= y_s/conj(t)
            Y[j,i] -= y_s/t
        where  y_s = 1/(R+jX),  b_c = B/2.

        For plain lines t = 1∠0 and stamps reduce to the standard form.

        Returns: Y (complex n×n), G=Re(Y), B=Im(Y),
                 bus_idx (bus_number → 0-based index)
        """
    banner("SECTION 3 – BUILDING Y-BUS ADMITTANCE MATRIX")

    n = len(buses)
    bus_idx = {b['num']: k for k, b in enumerate(buses)}
    Y = np.zeros((n, n), dtype=complex)

    for br in branches:
        i = bus_idx[br['fr']]
        j = bus_idx[br['to']]
        R, X, B_ch = br['R'], br['X'], br['B']

        if abs(X) < 1e-12:
            print(f"[YBUS] WARNING: |X|≈0 on branch "
                  f"{br['fr']}→{br['to']} – branch skipped")
            continue

        y_s = 1.0 / complex(R, X)  # series admittance
        b_c = B_ch / 2.0  # half line charging
        tap = br['tap'] if br['tap'] != 0.0 else 1.0
        phi = np.deg2rad(br['phs'])
        t = tap * np.exp(1j * phi)  # complex turns ratio

        Y[i, i] += y_s / abs(t) ** 2 + 1j * b_c
        Y[j, j] += y_s + 1j * b_c
        Y[i, j] -= y_s / np.conj(t)
        Y[j, i] -= y_s / t

    # Shunt admittances
    for b in buses:
        k = bus_idx[b['num']]
        Y[k, k] += complex(b['g_sh'], b['b_sh'])

    G = Y.real
    B = Y.imag

    print(f"[YBUS] Matrix size           : {n} × {n}")
    ############ below is for debugging purpose #########
    return Y, G, B, bus_idx
############ Ybus end here #################

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – POWER & CURRENT HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def calc_P_inj(i, n, V, T, G, B):
    """P_i = Σ_j V_i·V_j·(G_ij·cosθ_ij + B_ij·sinθ_ij)"""
    P = 0.0
    for j in range(n):
        dt = T[i] - T[j]
        P += V[i] * V[j] * (G[i,j]*np.cos(dt) + B[i,j]*np.sin(dt))
    return P

def calc_Q_inj(i, n, V, T, G, B):
    """Q_i = Σ_j V_i·V_j·(G_ij·sinθ_ij − B_ij·cosθ_ij)"""
    Q = 0.0
    for j in range(n):
        dt = T[i] - T[j]
        Q += V[i] * V[j] * (G[i,j]*np.sin(dt) - B[i,j]*np.cos(dt))
    return Q

def calc_branch_flow(i, j, V, T, G, B, br):
    #Active and reactive power flow from bus i to bus j (π-model, pu). Returns (P_ij, Q_ij).
    tap = br['tap'] if br['tap'] != 0.0 else 1.0
    R, X, B_ch = br['R'], br['X'], br['B']
    if abs(X) < 1e-12:
        return 0.0, 0.0
    y_s      = 1.0 / complex(R, X)
    g_s, b_s = y_s.real, y_s.imag
    b_c      = B_ch / 2.0
    Vi, Vj   = V[i], V[j]
    dt       = T[i] - T[j]
    P_ij = (g_s/tap**2)*Vi**2 \
           - (Vi*Vj/tap)*(g_s*np.cos(dt) + b_s*np.sin(dt))
    Q_ij = -(b_s+b_c)/tap**2*Vi**2 \
           + (Vi*Vj/tap)*(b_s*np.cos(dt) - g_s*np.sin(dt))
    return P_ij, Q_ij

def calc_current_ij(i, j, V, T, br):
    """
    Complex branch current phasor from bus i to bus j (π-model).

        t      = tap·e^(jφ)
        y_s    = 1/(R+jX)
        b_c    = B/2
        A      = y_s + j·b_c        (combined from-end admittance)
        I_ij   = (A/conj(t))·V_i·e^jθ_i  −  y_s·V_j·e^jθ_j

    Returns (I_complex, magnitude, angle_rad).
    """
    tap = br['tap'] if br['tap'] != 0.0 else 1.0
    R, X, B_ch = br['R'], br['X'], br['B']
    if abs(X) < 1e-12:
        return complex(0), 0.0, 0.0
    y_s  = 1.0 / complex(R, X)
    b_c  = B_ch / 2.0
    phi  = np.deg2rad(br['phs'])
    t    = tap * np.exp(1j * phi)
    A    = y_s + 1j * b_c

    Vi_ph = V[i] * np.exp(1j * T[i])
    Vj_ph = V[j] * np.exp(1j * T[j])
    I_ij  = (A / np.conj(t)) * Vi_ph - y_s * Vj_ph

    return I_ij, abs(I_ij), np.angle(I_ij)


def find_branch(branches, fr, to):
    """Return branch dict for fr↔to (either direction). None if absent."""
    for br in branches:
        if (br['fr']==fr and br['to']==to) or \
           (br['fr']==to and br['to']==fr):
            return br
    return None

############ helper functions end here ############

########### measurement vector starts here ##################
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – MEASUREMENT VECTOR ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════
def build_measurement_vector(file_measurements, buses, branches,
                             bus_idx, G, B, pmu_buses, pmu_sigma,
                             excluded_indices=None):
    """
    Assemble z (m,), W (m×m diagonal), meta_list from active file measurements
    plus synthetic PMU V_mag entries where needed.

    Parameters
    ----------
    file_measurements : list[dict]  – output of parse_measure_dat()
    pmu_buses         : list[int]   – bus numbers with PMU (from Block 2)
    pmu_sigma         : float       – PMU measurement standard deviation
    excluded_indices  : set[int]    – global indices of measurements to EXCLUDE
                                      (used during iterative bad-data removal).
                                      Index refers to position in the final
                                      meta_list from the FIRST assembly call.

    PMU integration
    ───────────────
    Buses in Block 2 (V_ang) are PMU buses.  A PMU provides BOTH |V| and θ.

    excluded_indices mechanism
    ──────────────────────────
    On iterative calls (bad data removal), excluded_indices contains the
    original position(s) of identified bad measurements.  The function
    rebuilds z/W/meta_list skipping those positions, so the index mapping
    to the original file measurements is preserved via a 'orig_idx' field
    in each meta_list entry.

    Returns z, W, meta_list.
    """
    banner("SECTION 5 – ASSEMBLING MEASUREMENT VECTOR")

    if excluded_indices is None:
        excluded_indices = set()
    if excluded_indices:
        print(f"[Z] Excluded measurement original-indices: "
              f"{sorted(excluded_indices)}")

    n = len(buses)

    z_list = []; sig_list = []; meta_list = []
    skipped = 0; orig_counter = 0
    vmag_in_file = set()

    # ── Step 1: Load active file measurements (respecting exclusions) ─────────
    for m in file_measurements:

        # Skip inactive (status != 0)
        if m['status'] != 0:
            print(f"[Z] SKIP (status={m['status']}): {m['type']} bus={m['i']}")
            skipped += 1; orig_counter += 1; continue

        # Skip buses not in network
        if m['i'] not in bus_idx:
            print(f"[Z] SKIP (bus {m['i']} not in network): {m['type']}")
            skipped += 1; orig_counter += 1; continue
        if m['j'] is not None and m['j'] not in bus_idx:
            print(f"[Z] SKIP (bus {m['j']} not in network): "
                  f"{m['type']} {m['i']}→{m['j']}")
            skipped += 1; orig_counter += 1; continue

        # Skip excluded (identified bad data)
        if orig_counter in excluded_indices:
            print(f"[Z] SKIP (bad-data-excluded, orig_idx={orig_counter}): "
                  f"{m['type']} bus_i={m['i']} bus_j={m.get('j','—')}")
            orig_counter += 1; continue

        z_list.append(m['value']); sig_list.append(m['sigma'])
        meta_list.append({'type':m['type'],'i':m['i'],'j':m['j'],
                          'sigma':m['sigma'],'source':'file',
                          'orig_idx':orig_counter})
        if m['type'] == 'V_mag':
            vmag_in_file.add(m['i'])
        orig_counter += 1

    print(f"[Z] File measurements loaded : {len(z_list)}")
    print(f"[Z] File measurements skipped: {skipped}")
    print(f"[Z] File measurements excluded (bad data): {len(excluded_indices)}")

    z   = np.array(z_list, dtype=float)
    sig = np.array(sig_list, dtype=float)
    W   = np.diag(1.0/sig**2)

    m_total  = len(z); n_states = 2*n-1
    print(f"\n[Z] ── Final measurement vector ──────────────────────────")
    print(f"[Z] Total measurements m  : {m_total}")
    print(f"[Z] State dimension 2n-1  : {n_states}")
    print(f"[Z] Redundancy m/(2n-1)   : {m_total/n_states:.3f}  (>1 required)")
    print(f"[Z] Weight 1/σ² range     : "
          f"{np.min(np.diag(W)):.1f} … {np.max(np.diag(W)):.1f}")
    tc = {}
    for mm in meta_list: tc[mm['type']] = tc.get(mm['type'],0)+1
    print("[Z] Type breakdown:")
    for t,c in sorted(tc.items()):
        ex = next(k for k,mm in enumerate(meta_list) if mm['type']==t)
        print(f"   {t:8s}: {c:3d}  σ={meta_list[ex]['sigma']:.4f}  "
              f"w=1/σ²={1/meta_list[ex]['sigma']**2:.1f}")
    return z, W, meta_list
############# end of measurement vector #######################

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – MEASUREMENT FUNCTION  h(x)
# ══════════════════════════════════════════════════════════════════════════════

def compute_h(V, T, G, B, n, meta_list, bus_idx, branches):
    """
    Evaluate measurement function vector h(x) at state (V, T).

      V_mag         → |V_i|
      V_ang         → θ_i                         (radians)
      P_inj         → Σ_j V_i V_j(G_ij cosθ_ij + B_ij sinθ_ij)
      Q_inj         → Σ_j V_i V_j(G_ij sinθ_ij − B_ij cosθ_ij)
      P_flow        → branch active power    (π-model)
      Q_flow        → branch reactive power  (π-model)
      I_mag         → |I_ij|                (complex current magnitude)
      I_ang         → ∠I_ij                 (complex current angle, rad)
    """
    h = np.zeros(len(meta_list))
    for row, meta in enumerate(meta_list):
        mtype = meta['type']
        i_bus = meta['i']
        i     = bus_idx.get(i_bus)
        if i is None:
            continue

        if mtype == 'V_mag':
            h[row] = V[i]
        elif mtype == 'V_ang':
            h[row] = T[i]
        elif mtype == 'P_inj':
            h[row] = calc_P_inj(i, n, V, T, G, B)
        elif mtype == 'Q_inj':
            h[row] = calc_Q_inj(i, n, V, T, G, B)
        elif mtype in ('P_flow', 'Q_flow', 'I_mag', 'I_ang'):
            j_bus = meta['j']
            j     = bus_idx.get(j_bus)
            if j is None:
                continue
            br = find_branch(branches, i_bus, j_bus)
            if br is None:
                continue
            if mtype == 'P_flow':
                P_ij, _   = calc_branch_flow(i, j, V, T, G, B, br)
                h[row]    = P_ij
            elif mtype == 'Q_flow':
                _, Q_ij   = calc_branch_flow(i, j, V, T, G, B, br)
                h[row]    = Q_ij
            elif mtype == 'I_mag':
                _, mag, _ = calc_current_ij(i, j, V, T, br)
                h[row]    = mag
            elif mtype == 'I_ang':
                _, _, ang = calc_current_ij(i, j, V, T, br)
                h[row]    = ang
    return h
############# computation of h function ends here ####

############# jacobian H starts here #################
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – JACOBIAN  H = ∂h/∂x
# ══════════════════════════════════════════════════════════════════════════════
def build_H(V, T, G, B, n, meta_list, bus_idx, branches):
    """
        Build the m × (2n-1) measurement Jacobian  H = ∂h/∂x.

        State vector
        ────────────
        x = [ θ₂ … θₙ | |V₁| … |Vₙ| ]
             └─(n-1)─┘   └────n────┘
        θ₁ (slack, index 0) is fixed reference → not a state variable.

        Column mapping
        ──────────────
        ∂/∂θ_k  (k > 0, 0-based) → column  k−1
        ∂/∂|V_k|                 → column  (n−1)+k

        Current Jacobian  (I_mag, I_ang)
        ─────────────────────────────────
        With  A = y_s + j·b_c,  a_ij = A/conj(t):
            I_ij = a_ij·V_i·e^jθ_i − y_s·V_j·e^jθ_j

        Complex partials:
            ∂I/∂θ_i  =  j·a_ij·V_i·e^jθ_i
            ∂I/∂|V_i|=    a_ij·e^jθ_i
            ∂I/∂θ_j  = −j·y_s·V_j·e^jθ_j
            ∂I/∂|V_j|= −y_s·e^jθ_j

        Chain rule:
            ∂|I|/∂x  = Re(conj(I)·∂I/∂x) / |I|
            ∂∠I/∂x   = Im(conj(I)·∂I/∂x) / |I|²
        """
    n_states = 2 * n - 1
    m = len(meta_list)
    H = np.zeros((m, n_states))

    ANG_OFF = 0
    MAG_OFF = n - 1

    def col_T(k):
        return ANG_OFF + k - 1  # k=1→col 0, k=2→col 1, …

    def col_V(k):
        return MAG_OFF + k

    for row, meta in enumerate(meta_list):
        mtype = meta['type']
        i_bus = meta['i']
        i = bus_idx.get(i_bus)
        if i is None:
            continue
        Vi = V[i];
        Ti = T[i]

        # ── V_mag ─────────────────────────────────────────────────────────────
        if mtype == 'V_mag':
            H[row, col_V(i)] = 1.0

        # ── V_ang (PMU angle) ─────────────────────────────────────────────────
        elif mtype == 'V_ang':
            if i > 0:
                H[row, col_T(i)] = 1.0

        # ── P_inj ─────────────────────────────────────────────────────────────
        elif mtype == 'P_inj':
            P_i = calc_P_inj(i, n, V, T, G, B)
            Q_i = calc_Q_inj(i, n, V, T, G, B)
            if i > 0:
                H[row, col_T(i)] = -Q_i - B[i, i] * Vi ** 2
            for j in range(n):
                if j == i: continue
                dt = Ti - T[j]
                if j > 0:
                    H[row, col_T(j)] = Vi * V[j] * (G[i, j] * np.sin(dt)
                                                    - B[i, j] * np.cos(dt))
            H[row, col_V(i)] = P_i / Vi + G[i, i] * Vi
            for j in range(n):
                if j == i: continue
                dt = Ti - T[j]
                H[row, col_V(j)] = Vi * (G[i, j] * np.cos(dt) + B[i, j] * np.sin(dt))

        # ── Q_inj ─────────────────────────────────────────────────────────────
        elif mtype == 'Q_inj':
            P_i = calc_P_inj(i, n, V, T, G, B)
            Q_i = calc_Q_inj(i, n, V, T, G, B)
            if i > 0:
                H[row, col_T(i)] = P_i - G[i, i] * Vi ** 2
            for j in range(n):
                if j == i: continue
                dt = Ti - T[j]
                if j > 0:
                    H[row, col_T(j)] = -Vi * V[j] * (G[i, j] * np.cos(dt)
                                                     + B[i, j] * np.sin(dt))
            H[row, col_V(i)] = Q_i / Vi - B[i, i] * Vi
            for j in range(n):
                if j == i: continue
                dt = Ti - T[j]
                H[row, col_V(j)] = Vi * (G[i, j] * np.sin(dt) - B[i, j] * np.cos(dt))

        # ── P_flow ────────────────────────────────────────────────────────────
        elif mtype == 'P_flow':
            j_bus = meta['j'];
            j = bus_idx.get(j_bus)
            if j is None: continue
            br = find_branch(branches, i_bus, j_bus)
            if br is None or abs(br['X']) < 1e-12: continue
            tap = br['tap'] if br['tap'] != 0.0 else 1.0
            y_s = 1.0 / complex(br['R'], br['X'])
            g_s, b_s = y_s.real, y_s.imag
            Vj = V[j];
            dt = Ti - T[j]
            dP_dTi = (Vi * Vj / tap) * (g_s * np.sin(dt) - b_s * np.cos(dt))
            dP_dTj = -(Vi * Vj / tap) * (g_s * np.sin(dt) - b_s * np.cos(dt))
            dP_dVi = 2 * (g_s / tap ** 2) * Vi - (Vj / tap) * (g_s * np.cos(dt) + b_s * np.sin(dt))
            dP_dVj = -(Vi / tap) * (g_s * np.cos(dt) + b_s * np.sin(dt))
            if i > 0: H[row, col_T(i)] = dP_dTi
            if j > 0: H[row, col_T(j)] = dP_dTj
            H[row, col_V(i)] = dP_dVi
            H[row, col_V(j)] = dP_dVj

        # ── Q_flow ────────────────────────────────────────────────────────────
        elif mtype == 'Q_flow':
            j_bus = meta['j'];
            j = bus_idx.get(j_bus)
            if j is None: continue
            br = find_branch(branches, i_bus, j_bus)
            if br is None or abs(br['X']) < 1e-12: continue
            tap = br['tap'] if br['tap'] != 0.0 else 1.0
            y_s = 1.0 / complex(br['R'], br['X'])
            g_s, b_s = y_s.real, y_s.imag
            b_c = br['B'] / 2.0
            Vj = V[j];
            dt = Ti - T[j]
            dQ_dTi = -(Vi * Vj / tap) * (g_s * np.cos(dt) + b_s * np.sin(dt))
            dQ_dTj = (Vi * Vj / tap) * (g_s * np.cos(dt) + b_s * np.sin(dt))
            dQ_dVi = -2 * (b_s + b_c) / tap ** 2 * Vi + (Vj / tap) * (b_s * np.cos(dt) - g_s * np.sin(dt))
            dQ_dVj = (Vi / tap) * (b_s * np.cos(dt) - g_s * np.sin(dt))
            if i > 0: H[row, col_T(i)] = dQ_dTi
            if j > 0: H[row, col_T(j)] = dQ_dTj
            H[row, col_V(i)] = dQ_dVi
            H[row, col_V(j)] = dQ_dVj

        # ── I_mag  /  I_ang ───────────────────────────────────────────────────
        elif mtype in ('I_mag', 'I_ang'):
            j_bus = meta['j'];
            j = bus_idx.get(j_bus)
            if j is None: continue
            br = find_branch(branches, i_bus, j_bus)
            if br is None or abs(br['X']) < 1e-12: continue

            tap = br['tap'] if br['tap'] != 0.0 else 1.0
            phi = np.deg2rad(br['phs'])
            t = tap * np.exp(1j * phi)
            y_s = 1.0 / complex(br['R'], br['X'])
            b_c = br['B'] / 2.0
            A = y_s + 1j * b_c
            a_ij = A / np.conj(t)

            I_c, mag_I, _ = calc_current_ij(i, j, V, T, br)
            if mag_I < 1e-9:
                continue

            Vi_ph = V[i] * np.exp(1j * T[i])
            Vj_ph = V[j] * np.exp(1j * T[j])

            dI_dTi = 1j * a_ij * Vi_ph
            dI_dVi = a_ij * np.exp(1j * T[i])
            dI_dTj = -1j * y_s * Vj_ph
            dI_dVj = -y_s * np.exp(1j * T[j])

            if mtype == 'I_mag':
                def dmag(dI):
                    return np.real(np.conj(I_c) * dI) / mag_I

                if i > 0: H[row, col_T(i)] = dmag(dI_dTi)
                if j > 0: H[row, col_T(j)] = dmag(dI_dTj)
                H[row, col_V(i)] = dmag(dI_dVi)
                H[row, col_V(j)] = dmag(dI_dVj)
            else:
                def dang(dI):
                    return np.imag(np.conj(I_c) * dI) / mag_I ** 2

                if i > 0: H[row, col_T(i)] = dang(dI_dTi)
                if j > 0: H[row, col_T(j)] = dang(dI_dTj)
                H[row, col_V(i)] = dang(dI_dVi)
                H[row, col_V(j)] = dang(dI_dVj)

    return H
################### jacobian ends here ################

################### observability starts here #########
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – OBSERVABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def observability_analysis(H, n_states, buses, branches, bus_idx):
    """
        Numerical observability check via SVD rank of the Jacobian H.

        Theory
        ──────
        A system is fully observable iff rank(H) = 2n−1 (= n_states).
        Rank is estimated from singular values of H:
            rank = number of σ_i  >  tol = max(m, ns)·ε_machine·σ_max

        A rank deficiency of k means k state variables cannot be determined
        from the current measurement set – the gain matrix G_m will be singular
        and the normal equations cannot be solved.

        Prints
        ──────
        • H matrix dimensions and required rank
        • Computed rank, deficiency, and observability verdict
        """
    banner("SECTION 8 – OBSERVABILITY ANALYSIS")

    m, ns = H.shape
    print(f"[OBS] H shape          : {m} rows × {ns} cols")
    print(f"[OBS] Required rank    : {ns}  (= 2n−1,  n = {(ns + 1) // 2})")
    print(f"[OBS] Measurements m   : {m}")
    if m - ns >= 0:
        print(f"[OBS] Redundancy       : {m - ns} extra measurements beyond minimum")
    else:
        print(f"[OBS] Redundancy       : {ns - m} measurements are required at minimum...")

    sv = np.linalg.svd(H, compute_uv=False)
    tol = max(m, ns) * np.finfo(float).eps * sv[0]  #add epsilon to avoid division by zero
    rank = int(np.sum(sv > tol))

    #print(f"\n[OBS] SVD tolerance    : {tol:.3e}")  #SVD is singular value decomposition
    #print(f"[OBS] σ_max (largest)  : {sv[0]:.6f}")
    #print(f"[OBS] σ_min (smallest) : {sv[-1]:.4e}")
    print(f"[OBS] Computed rank    : {rank}")
    print(f"[OBS] Rank deficiency  : {ns - rank}")

    if rank == ns:
        print(f"\n[OBS] ✓ System is FULLY OBSERVABLE  "
              f"(rank = {rank} = required {ns})")
        return rank, sv, True, [], []

    ############ detect unobservable branches and observale islands if system is unobservable #######
    print(f"\n[OBS] ✗ System is NOT FULLY OBSERVABLE  "
          f"(rank={rank}, deficiency={ns - rank})")
    print(f"\n[OBS] ── Identifying Observable Islands ─────────────────────────")

    all_bus_nums = {b['num'] for b in buses}

    # Collect all bus pairs that have at least one flow/current measurement
    # in the current measurement set.  These are the 'observed' connections.
    # We derive the measured-branch set directly from H's non-zero structure:
    # a branch (fr,to) is 'measured' if its two bus columns in H have any
    # non-zero entry in a row that corresponds to a P_flow/Q_flow/I_mag/I_ang.
    # For simplicity and correctness we use the branch list and check H rows.
    #
    # A cleaner approach: build the measured-edge set from the measurement
    # meta_list passed indirectly via H.  Since we don't pass meta_list here,
    # we use the H matrix structure: a branch i-j is 'observed' if there
    # exists at least one row of H that has non-zero entries in BOTH
    # col_V(i) and col_V(j) (or their angle columns), indicating a coupling.
    MAG_OFF = len(buses) - 1

    def col_V(k):
        return MAG_OFF + k

    measured_pairs = set()
    for br in branches:
        fr, to = br['fr'], br['to']
        if fr not in bus_idx or to not in bus_idx:
            continue
        ci = col_V(bus_idx[fr])
        cj = col_V(bus_idx[to])
        # Check if any row of H couples these two buses
        coupling = np.any((np.abs(H[:, ci]) > tol) & (np.abs(H[:, cj]) > tol))
        if coupling:
            measured_pairs.add((min(fr, to), max(fr, to)))
    """
    print(f"[OBS] Measured (coupled) branch pairs: {len(measured_pairs)}")
    for p in sorted(measured_pairs):
        print(f"   Branch {p[0]:3d} ↔ {p[1]:3d}  [measured]")
    """
    # Union-Find to build connected components from measured pairs
    parent = {bnum: bnum for bnum in all_bus_nums}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (fr, to) in measured_pairs:
        union(fr, to)

    # Group buses by their root → observable islands
    island_map = {}
    for bnum in all_bus_nums:
        root = find(bnum)
        island_map.setdefault(root, set()).add(bnum)

    islands = sorted(island_map.values(), key=lambda s: min(s))

    print(f"\n[OBS] ── Observable Islands ─────────────────────────────────────")
    print(f"[OBS] Number of islands found : {len(islands)}")
    island_rows = []
    for k, isle in enumerate(islands, 1):
        sorted_buses = sorted(isle)
        names = [next(b['name'] for b in buses if b['num'] == bn)
                 for bn in sorted_buses]
        #print(f"   Island {k}: buses {sorted_buses}")
        #print(f"             names {names}")
        island_rows.append([k, len(isle),
                            str(sorted_buses),
                            ", ".join(names)])
    ptable(island_rows,
           ['Island', 'Size', 'Bus numbers', 'Bus names'])

    # Identify unobservable branches: endpoints in different islands
    unobs_branches = []
    print(f"\n[OBS] ── Unobservable Branches ──────────────────────────────────")
    print(f"[OBS] (Branches whose endpoints lie in different observable islands)")
    for br in branches:
        fr, to = br['fr'], br['to']
        if fr not in bus_idx or to not in bus_idx:
            continue
        if find(fr) != find(to):
            # These two buses are in different islands → branch is unobservable
            unobs_branches.append(br)

    if unobs_branches:
        print(f"[OBS] {len(unobs_branches)} unobservable branch(es) found:")
        ub_rows = []
        for br in unobs_branches:
            # Find which islands the two endpoints belong to
            isl_fr = next(k + 1 for k, isle in enumerate(islands) if br['fr'] in isle)
            isl_to = next(k + 1 for k, isle in enumerate(islands) if br['to'] in isle)
            ub_rows.append([br['fr'], br['to'],
                            f"{br['R']:.4f}", f"{br['X']:.4f}",
                            f"Island {isl_fr}", f"Island {isl_to}"])
        ptable(ub_rows,
               ['From', 'To', 'R (pu)', 'X (pu)',
                'From-island', 'To-island'])
    else:
        print("[OBS] No unobservable branches identified "
              "(rank deficiency may be due to insufficient injection coverage)")

    return rank, sv, False, islands, unobs_branches
################ observability ends here ################

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – AC-WLS STATE ESTIMATOR  (iterative Newton)
# ══════════════════════════════════════════════════════════════════════════════

def ac_wls_estimator(buses, branches, bus_idx, G, B,
                     z, W, meta_list, max_iter=50, tol=1e-6):
    """
        AC Weighted Least Squares State Estimator – Newton / Gauss-Newton.

        Objective
        ─────────
        Minimise  J(x) = [z − h(x)]ᵀ W [z − h(x)]

        Newton iteration at step k
        ──────────────────────────
        1. h   = h(xᵏ)                      measurement function
        2. r   = z − h                       residual vector
        3. H   = ∂h/∂x |_{xᵏ}              Jacobian  (m × n_states)
        4. G_m = Hᵀ W H                      gain matrix (n_states × n_states)
        5. rhs = Hᵀ W r                      right-hand side
        6. Solve  G_m Δx = rhs               normal equations
        7. xᵏ⁺¹ = xᵏ + Δx
        8. Stop if  max|Δx| < tol

        At every iteration the following are printed and logged:
          ‖r‖₂       – residual 2-norm  (should decrease monotonically)
          rank(G_m)  – must equal n_states; singular → unobservable
          cond(G_m)  – condition number; large → numerical issues
          max|Δx|    – largest state correction  (convergence indicator)

        State vector layout
        ───────────────────
        x = [ θ₂ … θₙ | |V₁| … |Vₙ| ]    (θ₁ = 0 is slack reference)

        Returns
        -------
        V_est, T_est : estimated voltage magnitudes and angles
        r_final      : final residual vector
        converged    : bool
        """
    banner("SECTION 9 – AC-WLS STATE ESTIMATION  (Newton iteration)")

    n = len(buses)
    n_states = 2 * n - 1
    m = len(z)

    # Flat start: |V| = 1 pu, θ = 0 rad everywhere
    T = np.zeros(n)
    V = np.ones(n)

    print(f"[WLS] Flat-start initialisation: |V|=1.0 pu,  θ=0.0 rad")
    print(f"[WLS] n_buses          : {n}")
    print(f"[WLS] n_states  (2n−1) : {n_states}  "
          f"({n - 1} angle states + {n} voltage states)")
    print(f"[WLS] Measurements m   : {m}")
    print(f"[WLS] Redundancy       : {m}/{n_states} = {m / n_states:.3f}")
    print(f"[WLS] Convergence tol  : {tol}")
    print(f"[WLS] Max iterations   : {max_iter}\n")

    converged = False
    log_rows = []

    for it in range(1, max_iter + 1):

        # ── Step 1-2: h(x) and residual ──────────────────────────────────────
        h = compute_h(V, T, G, B, n, meta_list, bus_idx, branches)
        r = z - h
        r2 = np.linalg.norm(r)

        # ── Step 3: Jacobian ──────────────────────────────────────────────────
        H_mat = build_H(V, T, G, B, n, meta_list, bus_idx, branches)

        # ── Step 4: Gain matrix ───────────────────────────────────────────────
        Gm = H_mat.T @ W @ H_mat

        # Diagnose gain matrix before solving
        rank_Gm = np.linalg.matrix_rank(Gm)
        cond_Gm = np.linalg.cond(Gm)
        """
        print(f"  Iter {it:2d} | ‖r‖={r2:.5e} | "
              f"rank(Gm)={rank_Gm}/{n_states} | "
              f"cond(Gm)={cond_Gm:.3e}", end="")
        """
        # Singular gain matrix → unobservable → cannot continue
        if rank_Gm < n_states:
            print(f"\n[WLS] ✗ Gain matrix SINGULAR at iteration {it}  "
                  f"(rank {rank_Gm} < {n_states})")
            print(f"[WLS]   System is unobservable with this measurement set.")
            converged = False
            break

        # ── Step 5-6: solve normal equations ─────────────────────────────────
        rhs = H_mat.T @ W @ r
        try:
            dx = solve(Gm, rhs)
        except np.linalg.LinAlgError as e:
            print(f"\n[WLS] ✗ Linear solve failed: {e}")
            converged = False
            break

        # ── Step 7: state update ──────────────────────────────────────────────
        dT = np.zeros(n)
        dT[1:] = dx[:n - 1]  # θ₂…θₙ  (θ₁ = 0, slack, never updated)
        dV = dx[n - 1:]  # |V₁|…|Vₙ|
        T += dT
        V += dV

        max_dx = max(np.max(np.abs(dT)), np.max(np.abs(dV)))
        #print(f" | max|Δx|={max_dx:.5e}")
        log_rows.append([it, f"{r2:.5e}", rank_Gm,
                         f"{cond_Gm:.3e}", f"{max_dx:.5e}"])

        # ── Step 8: convergence check ─────────────────────────────────────────
        if max_dx < tol:
            print(f"\n[WLS] ✓ Converged in {it} iterations  "
                  f"(max|Δx| = {max_dx:.2e} < tol = {tol})")
            converged = True
            break
    else:
        print(f"\n[WLS] ✗ Did NOT converge within {max_iter} iterations")

    # ── Final objective and residuals ─────────────────────────────────────────
    h_f = compute_h(V, T, G, B, n, meta_list, bus_idx, branches)
    r_f = z - h_f
    J = float(r_f @ W @ r_f)
    print(f"[WLS] Final objective J(x̂) = {J:.6f}")
    print(f"[WLS] Final ‖r‖₂           = {np.linalg.norm(r_f):.6f}")

    # Iteration log
    print("\n  Iteration summary:")
    ptable(log_rows, ['Iter', '‖r‖', 'rank(Gm)', 'cond(Gm)', 'max|Δx|'])

    # Estimated state vs CDF truth
    print("\n  Estimated State vs CDF Load-Flow Ground Truth:")
    headers = ['Bus', 'Name', '|V|_est', '|V|_true', 'Δ|V|',
               'θ_est°', 'θ_true°', 'Δθ°']
    rows = []
    for k, b in enumerate(buses):
        dV = V[k] - b['v_mag']
        dT = np.rad2deg(T[k]) - np.rad2deg(b['v_ang'])
        rows.append([
            b['num'], b['name'],
            f"{V[k]:.5f}", f"{b['v_mag']:.5f}", f"{dV:+.5f}",
            f"{np.rad2deg(T[k]):.3f}", f"{np.rad2deg(b['v_ang']):.3f}",
            f"{dT:+.3f}",
        ])
    ptable(rows, headers)

    ################## plots ############
    #print(f"residual plot test: {log_rows}")   #for debugging purpose only
    """
    if(converged):
        residual = [row[4] for row in log_rows]

        plt.figure()
        plt.stem(residual)
        plt.grid(True)
        plt.title("Convergence Result Values")
        plt.xlabel("WLS iterations")
        plt.ylabel("|Δx| value decreases in upward direction")
        plt.show(block=False)
    """
    #######################
    return V, T, r_f, converged
############# WLS ends here ############

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – BAD DATA DETECTION & IDENTIFICATION  (single pass)
# ══════════════════════════════════════════════════════════════════════════════

def bad_data_single_pass(z, W, meta_list, V, T, G, B, n,
                         bus_idx, branches, iteration_label=""):
    """
    Single-pass bad data detection and identification by Largest Normalised Residual (LNR)
    ───────────────────────────────────────────────────
    Residual sensitivity: S = I − H(HᵀWH)⁻¹HᵀW
    Residual covariance : Ω = S · W⁻¹
    Normalised residual : r_N_i = |r_i| / √Ω_ii

    Critical measurements (Ω_ii ≈ 0) get r_N = 0 and cannot be detected.
    The measurement with the SINGLE largest r_N > 3.0 is returned for
    removal (LNR method: one measurement per pass, then re-estimate).

    Returns
    ───────
    bad_detected    : bool   – True if J > chi2_threshold
    worst_orig_idx  : int    – orig_idx of worst measurement, or None
    worst_rN        : float  – normalised residual of worst measurement
    r_N             : np.ndarray  – full normalised residual vector
    """
    banner(f"SECTION 10 – BAD DATA ANALYSIS  [{iteration_label}]")
    h  = compute_h(V,T,G,B,n,meta_list,bus_idx,branches)
    r  = z-h
    H  = build_H(V,T,G,B,n,meta_list,bus_idx,branches)
    m  = len(z); n_st = H.shape[1]; dof = m-n_st

    print(f"[BDD] m={m},  n_states={n_st},  dof={dof}")

    Gm     = H.T@W@H
    Gm_inv = np.linalg.pinv(Gm)
    W_inv  = np.diag(1.0/np.diag(W))
    S      = np.eye(m) - H@Gm_inv@H.T@W
    Omega  = S@W_inv

    r_N = np.zeros(m); critical_ct = 0
    for i in range(m):
        o = Omega[i,i]
        if o > 1e-12:
            r_N[i] = abs(r[i])/np.sqrt(o)
        else:
            critical_ct += 1

    print(f"[BDD] Critical measurements (Ω_ii≈0, r_N forced to 0): {critical_ct}")
    print(f"[BDD] r_N stats: min={np.min(r_N):.4f}  "
          f"max={np.max(r_N):.4f}  mean={np.mean(r_N):.4f}")

    THRESH = 3.0
    bad_detected = [(i, meta_list[i], r_N[i]) for i in range(m) if r_N[i] > THRESH]
    if bad_detected:
        print(f"[BDD] ✗ BAD DATA DETECTED")
    else:
        print(f"[BDD] ✓ No bad data detected")

    # LNR identification – top 10 for display
    top_n = min(10,m)
    top_idx = np.argsort(r_N)[::-1][:top_n]
    print(f"\n[BDD] ── LNR Test (threshold={THRESH}) ── Top {top_n} ─────────────")
    rows=[]
    for rk,idx in enumerate(top_idx,1):
        mm = meta_list[idx]
        flg = "⚠ BAD" if r_N[idx]>THRESH else "OK"
        rows.append([rk, idx, mm.get('orig_idx','—'), mm.get('source','?'),
                     mm['type'], mm['i'], mm.get('j') or '—',
                     f"{z[idx]:+.5f}", f"{h[idx]:+.5f}",
                     f"{r[idx]:+.5f}", f"{r_N[idx]:.3f}", flg])
    ptable(rows, ['Rank','Idx','OrigIdx','Src','Type','Bus_i','Bus_j',
                  'z','h(x̂)','r','r_N','Flag'])

    # Identify the single worst measurement (for removal in this pass)
    worst_local_idx = int(np.argmax(r_N))
    worst_rN        = r_N[worst_local_idx]
    worst_orig_idx  = None

    flagged = [(i, meta_list[i], r_N[i]) for i in range(m) if r_N[i]>THRESH]

    if flagged:
        # Sort by descending r_N; pick the worst one for removal
        flagged.sort(key=lambda x: -x[2])
        worst_local_idx = flagged[0][0]
        worst_orig_idx  = meta_list[worst_local_idx].get('orig_idx', None)
        worst_rN        = flagged[0][2]
        print(f"\n[BDD] ✗ {len(flagged)} measurement(s) exceed threshold {THRESH}:")
        for idx,mm,rn in flagged:
            print(f"   local_idx={idx:3d}  orig_idx={mm.get('orig_idx','—'):>4}  "
                  f"type={mm['type']:8s}  "
                  f"bus_i={mm['i']:3d}  bus_j={mm.get('j') or '—':>3}  "
                  f"r_N={rn:.4f}")
        print(f"\n[BDD] → REMOVING measurement with LARGEST r_N:")
        mm_w = meta_list[worst_local_idx]
        print(f"   local_idx={worst_local_idx}  orig_idx={worst_orig_idx}  "
              f"type={mm_w['type']:8s}  "
              f"bus_i={mm_w['i']:3d}  bus_j={mm_w.get('j') or '—':>3}  "
              f"r_N={worst_rN:.4f}")
    else:
        print(f"\n[BDD] ✓ All normalised residuals ≤ {THRESH} – no bad data")

    return bad_detected, worst_orig_idx, worst_rN, r_N


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – ITERATIVE BAD DATA REMOVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def iterative_bad_data_removal(file_meas, buses, branches, bus_idx, G, B,
                               pmu_buses, pmu_sigma, z_init, W_init,
                               meta_init):
    """
    Iterative WLS + bad data removal loop.

    Algorithm (per course notes LNR procedure)
    ──────────────────────────────────────────
    Given an initially observable and converged WLS solution:

    Loop:
      1. Run bad_data_single_pass() on current measurement set.
      2. If no bad data detected → STOP (clean solution).
      3. Identify the measurement with the largest r_N (worst).
      4. Tentatively remove it from the active set (add its orig_idx
         to the excluded_indices set).
      5. Re-check observability with the reduced measurement set.
           • If UNOBSERVABLE after removal:
               – Print warning: the removed measurement is critical for
                 observability – it cannot be safely removed.
               – Restore it (remove from excluded_indices) and STOP.
           • If still OBSERVABLE:
               – Rebuild z, W, meta_list with the measurement excluded.
               – Re-run WLS estimation.
               – If WLS converges → go to step 1 with new residuals.
               – If WLS fails to converge → STOP.

    The loop continues until either:
      (a) No bad data is found (clean).
      (b) Removing the worst measurement makes the system unobservable.
      (c) WLS fails to converge after a removal.

    Parameters
    ----------
    z_init, W_init, meta_init : initial (pre-bad-data-removal) measurement set
    All other params           : same as main pipeline

    Returns
    -------
    V_final, T_final : final estimated state
    r_N_final        : final normalised residual vector
    excluded_set     : set of orig_idx values that were removed
    stop_reason      : str describing why the loop stopped
    """
    banner("SECTION 10 – ITERATIVE BAD DATA REMOVAL LOOP")
    print("[LOOP] Starting iterative WLS + bad-data removal.")
    print("[LOOP] Rule: remove one measurement per pass (largest r_N),")
    print("[LOOP]       stop if removal makes system unobservable.\n")

    excluded_set = set()   # orig_idx values of removed measurements
    z   = z_init.copy()
    W   = W_init.copy()
    meta_list = list(meta_init)
    pass_no   = 0
    V_final   = None; T_final = None; r_N_final = np.array([])
    stop_reason = "unknown"
    plotted = False
    n = len(buses)

    while True:
        pass_no += 1
        print(f"\n[LOOP] ════ Pass {pass_no} ═══════════════════════════════════")
        print(f"[LOOP] Active measurements: {len(z)}")
        print(f"[LOOP] Excluded so far    : {sorted(excluded_set)}")

        # ── WLS estimation ────────────────────────────────────────────────────
        V_est, T_est, r_final, converged = ac_wls_estimator(
            buses, branches, bus_idx, G, B, z, W, meta_list
        )

        if not converged:
            print(f"\n[LOOP] ✗ WLS did not converge at pass {pass_no}.")
            print(f"[LOOP]   Stopping bad-data removal loop.")
            V_final=V_est; T_final=T_est; r_N_final=np.array([])
            stop_reason = "WLS did not converge"
            break

        V_final = V_est; T_final = T_est

        # ── Bad data single pass ──────────────────────────────────────────────
        bad_det, worst_orig, worst_rN, r_N = bad_data_single_pass(
            z, W, meta_list, V_est, T_est, G, B, n,
            bus_idx, branches,
            iteration_label=f"Pass {pass_no}"
        )

        if (not plotted) and bad_det:
            plt.figure()
            plt.stem(r_N)
            plt.grid(True)
            plt.title("Normalised Residuals")
            plt.xlabel("Index")
            plt.ylabel("Normalised Residuals Values")
            plt.show(block=False)
            plotted = True

        r_N_final = r_N
        if not bad_det or worst_orig is None:
            # No bad data detected → clean solution
            print(f"\n[LOOP] ✓ No bad data detected at pass {pass_no}.")
            print(f"[LOOP]   Clean solution achieved.")
            stop_reason = f"clean solution at pass {pass_no}"
            break


        # ── Tentative removal ─────────────────────────────────────────────────
        print(f"\n[LOOP] Tentatively removing measurement orig_idx={worst_orig} "
              f"(r_N={worst_rN:.4f}) ...")
        excluded_set.add(worst_orig)

        # Rebuild measurement vector WITHOUT the removed measurement
        z_new, W_new, meta_new = build_measurement_vector(
            file_meas, buses, branches, bus_idx, G, B,
            pmu_buses=pmu_buses, pmu_sigma=pmu_sigma,
            excluded_indices=excluded_set,
        )

        # ── Re-check observability ────────────────────────────────────────────
        print(f"\n[LOOP] Checking observability after removing "
              f"orig_idx={worst_orig} ...")

        V_true = np.array([b['v_mag'] for b in buses])
        T_true = np.array([b['v_ang'] for b in buses])
        n = len(buses)
        n_states = 2 * n - 1
        H_init1 = build_H(V_true, T_true, G, B, n, meta_new, bus_idx, branches)
        rank, sv, observable, islands, unobs_branches = observability_analysis(H_init1, n_states, buses, branches, bus_idx)

        if rank != n_states:
            # Removing this measurement makes the system unobservable
            # → it is a critical measurement; restore it and stop
            print(f"\n[LOOP] ✗ Removing orig_idx={worst_orig} makes the system "
                  f"UNOBSERVABLE!")
            print(f"[LOOP]   This measurement is critical for observability.")
            print(f"[LOOP]   Restoring it and stopping the removal loop.")
            excluded_set.discard(worst_orig)
            stop_reason = (f"measurement orig_idx={worst_orig} is critical "
                           f"for observability – cannot remove")
            # Rebuild final measurement set WITH the critical measurement restored
            z, W, meta_list = build_measurement_vector(
                file_meas, buses, branches, bus_idx, G, B,
                pmu_buses=pmu_buses, pmu_sigma=pmu_sigma,
                excluded_indices=excluded_set,
            )
            # Re-run WLS one last time on the restored set
            print(f"\n[LOOP] Re-running WLS on restored measurement set ...")
            V_final, T_final, r_N_arr, _ = ac_wls_estimator(
                buses, branches, bus_idx, G, B, z, W, meta_list
            )
            r_N_final = r_N_arr
            break

        # System is still observable after removal → accept and continue
        print(f"\n[LOOP] ✓ System remains OBSERVABLE after removing "
              f"orig_idx={worst_orig}.")
        print(f"[LOOP]   Measurement permanently removed. "
              f"Total excluded: {sorted(excluded_set)}")
        z = z_new; W = W_new; meta_list = meta_new

    return V_final, T_final, r_N_final, excluded_set, stop_reason


############ main starts here ##########
# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  EE574 – AC WLS State Estimator                                      ║")
    print("║  Contributors: Rana Muhammad Adnan Khalil & Eren Deniz Eminagaoglu   ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  Pipeline:                                                           ║")
    print("║   1. Parse CDF network file                                          ║")
    print("║   2. Parse measurement file  (8 blocks)                              ║")
    print("║   3. Build Y-bus                                                     ║")
    print("║   4. Identify PMU buses from Block 2 (V_ang)                         ║")
    print("║   5. Assemble measurement vector z and weight matrix W               ║")
    print("║   6. Observability analysis  (SVD rank of H)                         ║")
    print("║   7. AC-WLS estimation  (Newton iteration)                           ║")
    print("║   8. Bad data detection & identification  (LNR)                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    #CDF_FILE = "ieee_cdf_sample.dat"
    #CDF_FILE = "ieee_cdf_modified.dat"
    #CDF_FILE = "example2cdf.dat"
    CDF_FILE = input("Please write the name of CDF file with extension: ")

    #MEAS_FILE = "measure.dat"
    #MEAS_FILE = "measure_modified.dat"
    #MEAS_FILE = "example2measure.dat"
    MEAS_FILE = input("Please write the name of compatible measurement file with extension: ")

    buses, branches, base_mva = parse_ieee_cdf(CDF_FILE)
    n = len(buses)

    file_meas = parse_measure_dat(MEAS_FILE)

    # ── Build admittance matrix ───────────────────────────────────────────────
    Y, G, B, bus_idx = build_ybus(buses, branches)

    # CDF load-flow solution = ground truth for error metrics

    n = len(buses)
    n_states = 2 * n - 1
    print(f"\n[MAIN] n_buses={n}, n_branches={len(branches)}, "
          f"n_states=2n-1={n_states}")
    print(f"[MAIN] File measurements available: {len(file_meas)}")
    ########################################################################
    v_ang_records = [m for m in file_meas if m['type'] == 'V_ang']
    pmu_buses = [m['i'] for m in v_ang_records]
    pmu_sigma = v_ang_records[0]['sigma'] if v_ang_records else 0.001

    print(f"\n[MAIN] PMU buses (from Block 2 V_ang records) : {pmu_buses}")
    print(f"[MAIN] PMU sigma (from file)                   : {pmu_sigma}")

    if not pmu_buses:
        print("[MAIN] WARNING: No V_ang measurements in file – "
              "no PMU buses will be integrated.")

    # ── Step 5: Assemble measurement vector ───────────────────────────────────
    z, W, meta_list = build_measurement_vector(
        file_meas, buses, branches, bus_idx, G, B,
        pmu_buses=pmu_buses,
        pmu_sigma=pmu_sigma,
    )

    # ── Step 6: Observability analysis ───────────────────────────────────────
    # Evaluate H at the CDF true state for the initial observability check.
    V_true = np.array([b['v_mag'] for b in buses])
    T_true = np.array([b['v_ang'] for b in buses])

    H_init = build_H(V_true, T_true, G, B, n, meta_list, bus_idx, branches)
    rank, sv, observable, islands, unobs_branches = observability_analysis(H_init, n_states, buses, branches, bus_idx)

    # ── Step 7: AC-WLS state estimation ──────────────────────────────────────

    if(rank == n_states):
        V_final, T_final, r_N_final, excluded_set, stop_reason = \
            iterative_bad_data_removal(
                file_meas, buses, branches, bus_idx, G, B,
                pmu_buses, pmu_sigma,
                z_init=z, W_init=W, meta_init=meta_list
            )

        # ── Final pipeline summary ────────────────────────────────────────────────
        banner("PIPELINE COMPLETE")
        print(f"  System observable (initial)  : True")
        print(f"  Stop reason                  : {stop_reason}")
        print(f"  Measurements removed (bad)   : {len(excluded_set)}")
        if excluded_set:
            print(f"  Removed orig_idx set         : {sorted(excluded_set)}")
        if r_N_final is not None and len(r_N_final) > 0:
            n_bad_final = sum(1 for v in r_N_final if v > 3.0)
            print(f"  Bad data remaining (r_N>3.0) : {n_bad_final}")
        print("=" * 70)

        # ── Final status ──────────────────────────────────────────────────────────
        ################## plots ############

        plt.figure()
        plt.stem(r_N_final)
        plt.grid(True)
        plt.title("Final Normalised Residuals")
        plt.xlabel("Index")
        plt.ylabel("Final Normalised Residuals Values")
        plt.show()#block=False)

        # Create stem plot
        """
        plt.figure()
        plt.stem(r_N)
        # Add grid
        plt.grid(True)
        #plt.plot(r_N)
        # Add labels (Optional)
        plt.title("Normalised Residuals")
        plt.xlabel("Index")
        plt.ylabel("Normalised Residuals Values")

        plt.show()
        plt.show(block=False)
        """
########## main ends here #################

if __name__ == "__main__":
    main()
