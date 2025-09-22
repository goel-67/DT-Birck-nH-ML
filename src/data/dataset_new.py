from sqlalchemy import create_engine, inspect
import pandas as pd

username = 'glance'
password = 'glance'
host = '10.165.25.65'
port = '5432' # Default PostgreSQL port
database = 'logger'

connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'

engine = create_engine(connection_string)
connection = engine.connect()

# ─── Step 1: Pull out every “etchM/D/YYYY…” run OR any “MM_DD_YYYY[…]” run ────
idruns_lotnames_query = r"""
SELECT DISTINCT idruns, lotname
FROM runs
WHERE 
     lotname ~ '^etch[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}.*$'        -- etchM/D/YYYY… 
  OR lotname ~ '^[0-9]{2}_[0-9]{2}_[0-9]{4}\[.*\]$'             -- MM_DD_YYYY[anything]
"""
idruns_lotnames_df = pd.read_sql(idruns_lotnames_query, con=engine)
idruns = idruns_lotnames_df['idruns'].tolist()

# ─── Step 2: Dynamically look up the parameter IDs ────────────────────────────
param_names = [
    'P Chamber Pressure', 'G O2 Flow', 'G CF4 Flow',
    'T Temp Chamber', 'T Temp Electrode', 
    'RF1 Fwd Power', 'RF2 Fwd Power'
]
param_ids_query = f"""
SELECT DISTINCT idparameters, name
FROM parameters
WHERE name IN ({', '.join("'" + p + "'" for p in param_names)})
"""
param_ids_df = pd.read_sql(param_ids_query, con=engine)
param_ids = dict(zip(param_ids_df['name'], param_ids_df['idparameters']))

# ─── Step 3: Fetch all of the matching data rows ──────────────────────────────
values_query = f"""
SELECT 
    d.value,
    d.idparameters,
    s.time,
    r.idruns,
    r.lotname
FROM data AS d
JOIN samplerecord AS s ON d.idsamplerecord = s.idsamplerecord
JOIN runs         AS r ON s.idruns           = r.idruns
WHERE r.idruns IN ({','.join(map(str, idruns))})
  AND d.idparameters IN ({','.join(map(str, param_ids.values()))})
ORDER BY r.idruns, d.idparameters, s.time ASC
"""
values_df = pd.read_sql(values_query, con=engine)

# ─── Step 4: Ensure the timestamp column is a true datetime ──────────────────
values_df['time'] = pd.to_datetime(values_df['time'], utc=True)

import os
import re
import csv
import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ─── Config for post-SQL processing ─────────────────────────────────────────

param_csv_mapping = {
    'P Chamber Pressure': 'AvgPres',
    'G O2 Flow':           'AvgO2Flow',
    'G CF4 Flow':          'Avgcf4Flow',
    'T Temp Chamber':      'Avg_ChTemp',
    'T Temp Electrode':    'Avg_ElecTemp',
    'RF1 Fwd Power':       'Avg_Rf1_Pow',
    'RF2 Fwd Power':       'Avg_Rf2_Pow'
}

step_durations = {
    'Chuck':    30,
    'Gas Stab': 10,
    'Dechuck':  30
}

steps_order = ['Initial', 'Chuck', 'Gas Stab', 'Etch', 'Dechuck', 'End']
fimap_folder = "fimap"

# ─── Helper functions ────────────────────────────────────────────────────────

def parse_lotname(lot):
    raw = lot.strip()
    if raw.lower().startswith('etch'):
        raw = raw[4:]
    # accept slash, dash, or underscore as date separators
    m = re.match(r'(\d{1,2})[\/_\-](\d{1,2})[\/_\-](\d{4})(.*)', raw)
    if not m:
        return None, None
    mo, d, y, tk = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", tk.strip('_-')

def parse_fimap_filename(fn):
    base = os.path.splitext(fn)[0]
    m = re.match(r'(\d{1,2})[_\-](\d{1,2})[_\-](\d{2,4})(.*)', base)
    if not m:
        return None, None
    mo, d, y, tk = m.groups()
    if len(y) == 2:
        y = '20' + y
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}", tk.strip('_-')

def normalize_tok(tok):
    if not tok:
        return ''
    tok = re.sub(r'correct', '', tok, flags=re.I)
    tok = re.sub(r'\d+', lambda m: str(int(m.group())), tok)
    tok = re.sub(r'[^0-9A-Za-z]', '', tok)
    return tok.lower()

def dates_within_one_day(d1, d2):
    a = datetime.datetime.strptime(d1, '%Y-%m-%d').date()
    b = datetime.datetime.strptime(d2, '%Y-%m-%d').date()
    return abs((a - b).days) == 1

def match_lot_and_file(lot: str, base: str) -> bool:
    """
    Return True if the DataSet lotname (lot) corresponds to the .fimap filename base.
    Matching rules:
      1. Dates must be identical, or within one day if both tokens non‐empty.
      2. Tokens must normalize exactly, or both be in the allowed “blank” set.
    """
    # extract date & token from the run’s LOTNAME
    ds_date, ds_tok = parse_lotname(lot)
    # extract date & token from the .fimap filename (without extension)
    f_date,  f_tok  = parse_fimap_filename(base)

    # if we can’t parse either, no match
    if not ds_date or not f_date:
        return False

    # if dates differ, only allow ±1 day when both tokens are non‐empty
    if ds_date != f_date:
        n1, n2 = normalize_tok(ds_tok), normalize_tok(f_tok)
        if not (n1 and n2 and dates_within_one_day(ds_date, f_date)):
            return False

    # normalize tokens for exact comparison
    ds_norm = normalize_tok(ds_tok)
    fs_norm = normalize_tok(f_tok)

    # exact token match
    if ds_norm == fs_norm:
        return True

    # allow “blank” variants (e.g. no token, “normal”, “400w”, etc.)
    blanks = {'', 'normal', 'normaletch', '400w', '450w', 'highpower'}
    if ds_norm in blanks and fs_norm in blanks:
        return True

    # otherwise, no match
    return False

def parse_csv_line(line):
    return next(csv.reader([line]))

def try_cast_numeric(val):
    try:
        return float(val)
    except:
        return val

def parse_fimap(filepath):
    metadata = {}
    measurement_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [L.strip() for L in f if L.strip()]
    i = 0
    # skip leading numeric lines
    while i < len(lines) and re.match(r'^\d+$', lines[i]):
        i += 1
    meas_header  = re.compile(r'^"Die x \(mm\)"')
    stats_header = re.compile(r'^"Parameter","Min","Max","Mean"')
    # collect metadata until measurement header
    while i < len(lines) and not meas_header.search(lines[i]) and not stats_header.search(lines[i]):
        parts = parse_csv_line(lines[i])
        if len(parts) >= 2:
            metadata[parts[0].strip('"')] = try_cast_numeric(parts[1])
        i += 1
    # if no measurement section, return empty
    if i >= len(lines) or not meas_header.search(lines[i]):
        return metadata, pd.DataFrame()
    cols = parse_csv_line(lines[i]); i += 1
    # collect measurement rows
    while i < len(lines) and not stats_header.search(lines[i]) and re.match(r'^-?\d', lines[i]):
        measurement_data.append(parse_csv_line(lines[i]))
        i += 1
    if measurement_data:
        df = pd.DataFrame(measurement_data, columns=cols[:len(measurement_data[0])])
        df.columns = [c.strip('"').replace('(mm)','mm').replace('(nm)','nm') for c in df.columns]
        for c in df.columns:
            df[c] = df[c].apply(try_cast_numeric)
    else:
        df = pd.DataFrame()
    return metadata, df

def compute_step_averages(run_data, param_ids, param_csv_mapping):
    run_data['time'] = pd.to_datetime(run_data['time'])
    cf4 = run_data[run_data['idparameters']==param_ids['G CF4 Flow']].copy()
    rf1 = run_data[run_data['idparameters']==param_ids['RF1 Fwd Power']].copy()
    rf2 = run_data[run_data['idparameters']==param_ids['RF2 Fwd Power']].copy()
    for df in (cf4, rf1, rf2):
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
    start_time = run_data['time'].min()
    end_time   = run_data['time'].max()
    gas_pts    = cf4['value'].diff().gt(22)
    if not gas_pts.any():
        return None
    gas_start     = gas_pts.idxmax()
    rf_times      = pd.concat([rf1[rf1['value']>20], rf2[rf2['value']>20]]).index
    if rf_times.empty:
        return None
    etch_start    = rf_times.min()
    below         = cf4[cf4['value']<5].loc[etch_start:]
    if below.empty:
        return None
    dechuck_start = below.index[0]
    chuck_start   = gas_start - pd.Timedelta(seconds=step_durations['Chuck'])
    bounds = {
        'Initial':  (start_time, chuck_start),
        'Chuck':    (chuck_start, gas_start),
        'Gas Stab': (gas_start, etch_start),
        'Etch':     (etch_start, dechuck_start),
        'Dechuck':  (dechuck_start, dechuck_start + pd.Timedelta(seconds=step_durations['Dechuck'])),
        'End':      (dechuck_start + pd.Timedelta(seconds=step_durations['Dechuck']), end_time)
    }
    row = {
        'idruns':         run_data['idruns'].iloc[0],
        'LOTNAME':        run_data['LOTNAME'].iloc[0],
        'run_start_time': start_time,
        'run_end_time':   end_time
    }
    for step in steps_order:
        start, stop = bounds[step]
        for pname, pid in param_ids.items():
            seg = run_data[
                (run_data['idparameters']==pid) &
                (run_data['time']>=start) &
                (run_data['time']<stop)
            ]['value']
            row[f"{step}_{param_csv_mapping[pname]}"] = seg.mean() if not seg.empty else np.nan
        row[f"{step}_duration"] = (stop - start).total_seconds()
    return row

# ─── Main processing ─────────────────────────────────────────────────────────

# 1) Rename .fimap files for consistency
for fname in os.listdir(fimap_folder):
    if not fname.lower().endswith('.fimap'):
        continue
    date, tok = parse_fimap_filename(fname)
    if not date:
        print(f"Skipping unrecognized file: {fname}")
        continue
    yyyy, mm, dd = date.split('-')
    base_date     = f"{mm.zfill(2)}_{dd.zfill(2)}_{yyyy}"
    variant_clean = re.sub(r'[^0-9A-Za-z]', '', tok or '').upper()
    new_name      = f"{base_date}_{variant_clean}.fimap" if variant_clean else f"{base_date}.fimap"
    orig, dest    = os.path.join(fimap_folder, fname), os.path.join(fimap_folder, new_name)
    if fname != new_name and not os.path.exists(dest):
        os.rename(orig, dest)
        print(f"Renamed: {fname} → {new_name}")

# 2) Gather list of available .fimap files
file_list = [f for f in os.listdir(fimap_folder) if f.lower().endswith('.fimap')]

# 3) Compute step averages
step_rows = []
for idrun, lot in zip(idruns_lotnames_df['idruns'], idruns_lotnames_df['lotname']):
    sub = values_df[values_df['idruns']==idrun].copy()
    sub['LOTNAME'] = lot
    sa = compute_step_averages(sub, param_ids, param_csv_mapping)
    if sa is not None:
        step_rows.append(sa)
step_averages_df = pd.DataFrame(step_rows)

# 4) Map LOTNAMEs to .fimap files (with all special cases)
mapped, unmapped = [], []
all_lots = idruns_lotnames_df['lotname'].tolist()
for lot in all_lots:
    if lot == 'etch12/18/2024 _50W':
        unmapped.append(lot)
        continue
    if lot == 'etch11/05/2024':
        mapped.append((lot, '11_05_2024_500W.fimap'))
        continue
    if lot == 'etch10/30/2024_Highpower':
        mapped.append((lot, '10_30_2024_450W.fimap'))
        continue
    if lot == 'etch1/22/2025_Normal_400W':
        mapped.append((lot, '01_22_2025.fimap'))
        continue
    '''if lot == '07_30_2025[_367W]':
        mapped.append((lot, '07_30_2025_367W.fimap'))
        continue'''
    if lot in step_averages_df['LOTNAME'].values:
        match = next(
            (f for f in file_list
             if match_lot_and_file(lot, os.path.splitext(f)[0])),
            None
        )
        if match:
            mapped.append((lot, match))
            continue
    unmapped.append(lot)

if unmapped:
    print("⚠️ Unmapped LOTNAMEs:", unmapped)

# 5) Parse each matched .fimap and compute etch metrics
records = []
for lot, fname in mapped:
    path = os.path.join(fimap_folder, fname)
    meta, dfm = parse_fimap(path)
    if dfm.empty:
        continue
    tcol = next((c for c in dfm.columns if "Thickness" in c), None)
    if not tcol:
        continue
    x = dfm['Die x mm'].astype(float).values
    y = dfm['Die y mm'].astype(float).values
    thick = pd.to_numeric(dfm[tcol], errors='coerce').dropna().values
    t = meta.get("Etch/Dep Time:")
    u = meta.get("Etch/Dep Time Unit")
    if t is None or u is None:
        continue
    etch_min = float(t)/60.0 if u == 0 else float(t)
    rates    = thick / etch_min
    
    # Ensure all arrays have the same length
    min_length = min(len(x), len(y), len(rates))
    if min_length == 0:
        print(f"   Skipping {lot}: No valid measurements found")
        continue
    
    # Truncate arrays to the same length
    x = x[:min_length]
    y = y[:min_length]
    rates = rates[:min_length]
    
    print(f"   Processing {lot}: {min_length} measurement points")
    
    avg_r, min_r, max_r = rates.mean(), rates.min(), rates.max()
    rng_r    = max_r - min_r
    
    # Warning for range etchrate over 7 and outlier handling
    if rng_r > 7:
        print(f"⚠️ WARNING: Range etchrate is {rng_r:.2f} (> 7) for lotname: {lot}")
        print(f"   Detecting and removing outliers...")
        
        # Store original rates for comparison
        original_rates = rates.copy()
        original_x = x.copy()
        original_y = y.copy()
        
        # Detect outliers using 3 standard deviations from mean
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        outlier_threshold = 3 * std_rate
        
        # Find outlier indices
        outlier_mask = np.abs(rates - mean_rate) > outlier_threshold
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) > 0:
            print(f"   Found {len(outlier_indices)} outliers: {outlier_indices + 1}")
            print(f"   Outlier values: {rates[outlier_indices]}")
            
            # Remove outliers
            clean_rates = rates[~outlier_mask]
            clean_x = x[~outlier_mask]
            clean_y = y[~outlier_mask]
            
            # Recalculate metrics without outliers
            clean_avg_r = np.mean(clean_rates)
            clean_min_r = np.min(clean_rates)
            clean_max_r = np.max(clean_rates)
            clean_rng_r = clean_max_r - clean_min_r
            
            print(f"   Clean range etchrate: {clean_rng_r:.2f} (was {rng_r:.2f})")
            
            # Use clean metrics for calculations
            avg_r, min_r, max_r = clean_avg_r, clean_min_r, clean_max_r
            rng_r = clean_rng_r
            
            # Fill outlier positions with mean of remaining rates
            for idx in outlier_indices:
                rates[idx] = clean_avg_r
                # Note: x and y coordinates remain unchanged for outlier positions
        else:
            print(f"   No outliers detected using 3σ threshold")
    
    nu_pct   = ((max_r - min_r) / (2 * avg_r)) * 100
    uni      = 100 - nu_pct
    row = {
        'LOTNAME':          lot,
        'FIMAP_FILE':       fname,
        'AvgEtchRate':      avg_r,
        'MinEtchRate':      min_r,
        'MaxEtchRate':      max_r,
        'RangeEtchRate':    rng_r,
        'NonUniformityPct': nu_pct,
        'UniformityScore':  uni
    }
    for i, (xi, yi, ri) in enumerate(zip(x, y, rates), start=1):
        row[f'X_{i}_mm']            = xi
        row[f'Y_{i}_mm']            = yi
        row[f'Rate_{i}_nm_per_min'] = ri
    records.append(row)
fimap_metrics_df = pd.DataFrame(records)

# 6) Merge step averages with fimap metrics
final_df = (
    step_averages_df
    .merge(fimap_metrics_df, on='LOTNAME', how='inner')
    .drop_duplicates(subset='idruns', keep='first')
    .sort_values('idruns')
    .reset_index(drop=True)
)

# 7) Integrate chamber cleaning info
clean_dates = [
    datetime.datetime.strptime(d, '%Y-%m-%d').date()
    for d in ['2024-07-22', '2024-09-18', '2024-09-27', '2025-02-12', '2025-06-17']
]
run_dates     = set(final_df['run_start_time'].dt.date)
clean_no_runs = set(clean_dates) - run_dates

def get_last_clean_date(run_date):
    past = [cd for cd in clean_dates if cd <= run_date]
    return max(past) if past else None

final_df['run_date']              = final_df['run_start_time'].dt.date
final_df['last_clean_date']       = final_df['run_date'].apply(get_last_clean_date)
final_df['days_since_last_clean'] = (
    final_df['run_date'] - final_df['last_clean_date']
).apply(lambda x: x.days if x is not None else np.nan)

def determine_cleaned(row):
    rd = row['run_date']
    if rd in clean_dates:
        _, tok = parse_lotname(row['LOTNAME'])
        tok_norm = normalize_tok(tok)
        if 'lh' in tok_norm:
            return False
        return True
    for cd in clean_no_runs:
        if rd == cd - datetime.timedelta(days=1) or rd == cd + datetime.timedelta(days=1):
            return True
    return False

final_df['cleaned']               = final_df.apply(determine_cleaned, axis=1)
final_df = final_df.sort_values('run_start_time')
final_df['runs_since_last_clean'] = final_df.groupby('last_clean_date').cumcount()

# 8) Save full dataset
import os
# Ensure data directory exists
os.makedirs("data", exist_ok=True)
final_df.to_csv("data/full_dataset.csv", index=False)
print(f"✅ Wrote data/full_dataset.csv with {len(final_df)} rows, including cleaning flags")
