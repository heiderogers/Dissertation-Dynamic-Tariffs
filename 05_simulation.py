"""
Simulation — Pipeline 3a
Computes hourly bills for all archetype × tariff × scenario × year combinations
Aggregates to monthly and saves to data_simulation/bills.csv

Archetypes:
  A — Basic household (3,500 kWh/yr): base load only
  B — Heat pump (7,500 kWh/yr): base load + HP (4,000 kWh/yr)
  C — HP + EV (10,000 kWh/yr): base load + HP + EV (2,500 kWh/yr)
  D — EV only (6,000 kWh/yr): base load + EV (2,500 kWh/yr)

Tariffs:
  T1 — Flat: fixed wholesale + retail + grid + taxes (BDEW April 2026: 37.0 ct/kWh)
  T2 — Dynamic theoretical: hourly EPEX + fixed grid + taxes (no provider fee)
  T3 — Dynamic retail: T2 + Tibber Aufschlag (2.15 ct/kWh) + Grundgebühr (€71.88/yr)
  T4 — Dynamic retail + grid: T3 + time-variable grid fees (Stromnetz Berlin §14a Module 3)
       T4 applies to Archetypes B, C, D only (§14a device required)

Scenarios:
  passive   — no load shifting; household consumes at scheduled times
  automated — HEMS optimises load timing using day-ahead price signal

Flat adjustments (monthly):
  Grundgebühr: €71.88/yr ÷ 12 = €5.99/month (Tibber, April 2026)
    Applied to T3 and T4 for all archetypes
  Module 1 reduction: -€123.18/yr ÷ 12 = -€10.265/month (Stromnetz Berlin, 2026)
    Applied to T1, T2, T3 for Archetypes B, C, D (§14a device holders)

Load shifting assumptions:
  White goods (Archetype A, automated):
    Annual shiftable load: 350 kWh/yr (~10% of 3,500 kWh; washing machine,
      dryer, dishwasher)
    Passive hours: 18-21 (evening concentration assumed)
    Automated: daily budget (0.959 kWh) moved to cheapest hour in 00:00-24:00
    Source: methodology assumption; load embedded in BDEW H0 profile

  Heat pump (Archetypes B, C, automated):
    Annual HP load: 4,000 kWh/yr, temperature-driven via heating degree hours
      below 15°C (VDI 3807 Blatt 1, 2013; DWD station Potsdam ID 03987)
    Shifting window: ±2h — consistent with empirical evidence that 2-hour HP-off
      periods cause ≤1°C indoor temperature change in well-insulated homes
      (Gupta & Morey, 2023); IEA HPT Magazine (2024) expert consensus 1-3hr window
    Minimum adjustment interval: 2h — consistent with real HEMS implementations
      to avoid short-cycling (Tibber Node-RED implementation, 2024)
    Maximum hourly capacity: 3 kWh — nominal electrical capacity of residential
      HP compressors in German field installations (Knoop et al., 2022)
    Resulting shifting rate: ~22% of annual HP load, mean 5 adjustments/day
    No building thermal state model — simplification relative to full MPC
      (Mascherbauer et al., 2022); justified by Gupta & Morey (2023) comfort evidence

  EV (Archetypes C, D, automated):
    Annual EV load: 2,500 kWh/yr; daily budget 6.85 kWh/day
      (Märtz et al., 2022, DOI: 10.3390/en15186575)
    Passive hours: 18-21 — peak evening charging (Märtz et al., 2022)
    Plug-in window: 18:00-07:00 — consistent with overnight HEMS shifting
      (Spencer et al., 2021, DOI: 10.1016/j.trd.2021.103023)
    2-night lookahead: if tomorrow night's cheapest hour < tonight's cheapest hour,
      defer charging by one day. Reflects day-ahead price transparency available
      to real HEMS (Tibber publishes EPEX day-ahead prices at ~12:30 for next day)
    At 11kW wallbox, 6.85 kWh requires ~37 minutes — concentrating in one hour
      is physically realistic
    Daily averaging understates day-to-day price variation (actual charging
      frequency ~3x/week per Märtz et al., 2022) — automated savings are a
      conservative lower bound
    Known limitation: cross-year deferral modelled continuously across year
      boundaries; last day of dataset (31.12.2025) has no following day so
      no deferral possible

Outputs:
  data_simulation/bills.csv — long-format monthly bills
    Columns: archetype, tariff, scenario, year, month, monthly_bill_eur
    Expected rows: 1,080 (4 archetypes × 4 tariffs × 2 scenarios × 3 years × 12 months,
    minus excluded combinations: A×T4 passive and automated = 72 rows excluded)
"""

import pandas as pd
import numpy as np
import os

os.makedirs('data_simulation', exist_ok=True)

# ============================================================
# LOAD INPUTS
# ============================================================
tariffs = pd.read_csv('data_simulation/tariff_prices.csv',
                      index_col='datetime', parse_dates=True)
load_A = pd.read_csv('data_simulation/load_profile_A.csv',
                     index_col='datetime', parse_dates=True)
load_B = pd.read_csv('data_simulation/load_profile_B.csv',
                     index_col='datetime', parse_dates=True)
load_C = pd.read_csv('data_simulation/load_profile_C.csv',
                     index_col='datetime', parse_dates=True)
load_D = pd.read_csv('data_simulation/load_profile_D.csv',
                     index_col='datetime', parse_dates=True)

print(f"Tariffs: {len(tariffs)} rows")
print(f"Load A: {len(load_A)} rows")
print(f"Load B: {len(load_B)} rows")
print(f"Load C: {len(load_C)} rows")
print(f"Load D: {len(load_D)} rows")

# ============================================================
# FLAT MONTHLY ADJUSTMENTS
# Source: Tibber (April 2026) — Grundgebühr €5.99/month = €71.88/yr
# Source: Stromnetz Berlin Preisblatt 2026 — Module 1 reduction -€123.18/yr
# ============================================================
GRUNDGEBUEHR_MONTHLY = 71.88 / 12   # Tibber April 2026
MODULE1_MONTHLY = 123.18 / 12       # Stromnetz Berlin 2026, §14a Module 1


def get_annual_adjustment(archetype, tariff):
    """
    Returns monthly flat adjustment (€) for a given archetype × tariff combination.
    Returns None if combination is excluded (A × T4: no §14a device).
    """
    if archetype == 'A' and tariff == 'T4':
        return None

    adjustment = 0.0

    # Grundgebühr applies to T3 and T4 (Tibber provider fee)
    if tariff in ['T3', 'T4']:
        adjustment += GRUNDGEBUEHR_MONTHLY

    # Module 1 reduction applies to §14a device holders (B, C, D) on T1, T2, T3
    # T4 already incorporates time-variable grid fees; Module 1 not additionally applied
    if archetype in ['B', 'C', 'D'] and tariff in ['T1', 'T2', 'T3']:
        adjustment -= MODULE1_MONTHLY

    return adjustment


print("\n=== ANNUAL ADJUSTMENT CHECKS ===")
for arch in ['A', 'B', 'C', 'D']:
    for tariff in ['T1', 'T2', 'T3', 'T4']:
        adj = get_annual_adjustment(arch, tariff)
        if adj is not None and adj != 0:
            print(f"Archetype {arch}, {tariff}: {adj:+.3f} €/month ({adj * 12:+.2f} €/yr)")

# ============================================================
# MERGE LOADS AND TARIFFS
# ============================================================
sim = tariffs.copy()
sim['load_A'] = load_A['load_A']
sim['load_B'] = load_B['load_B']
sim['load_C'] = load_C['load_C']
sim['load_D'] = load_D['load_D']

print(f"\nSimulation dataframe: {len(sim)} rows")
print(f"Missing values: {sim.isnull().sum().sum()}")

# ============================================================
# PASSIVE BILL CALCULATION
# bill = load (kWh) × price (ct/kWh) / 100 → €
# ============================================================
sim['bill_A_T1'] = sim['load_A'] * sim['t1_ct_kwh'] / 100
sim['bill_A_T2'] = sim['load_A'] * sim['t2_ct_kwh'] / 100
sim['bill_A_T3'] = sim['load_A'] * sim['t3_ct_kwh'] / 100

sim['bill_B_T1'] = sim['load_B'] * sim['t1_ct_kwh'] / 100
sim['bill_B_T2'] = sim['load_B'] * sim['t2_ct_kwh'] / 100
sim['bill_B_T3'] = sim['load_B'] * sim['t3_ct_kwh'] / 100
sim['bill_B_T4'] = sim['load_B'] * sim['t4_ct_kwh'] / 100

sim['bill_C_T1'] = sim['load_C'] * sim['t1_ct_kwh'] / 100
sim['bill_C_T2'] = sim['load_C'] * sim['t2_ct_kwh'] / 100
sim['bill_C_T3'] = sim['load_C'] * sim['t3_ct_kwh'] / 100
sim['bill_C_T4'] = sim['load_C'] * sim['t4_ct_kwh'] / 100

sim['bill_D_T1'] = sim['load_D'] * sim['t1_ct_kwh'] / 100
sim['bill_D_T2'] = sim['load_D'] * sim['t2_ct_kwh'] / 100
sim['bill_D_T3'] = sim['load_D'] * sim['t3_ct_kwh'] / 100
sim['bill_D_T4'] = sim['load_D'] * sim['t4_ct_kwh'] / 100

# ============================================================
# AGGREGATE TO MONTHLY + ADD FLAT ADJUSTMENTS
# ============================================================
monthly = sim.groupby(['year', 'month'])[[
    'bill_A_T1', 'bill_A_T2', 'bill_A_T3',
    'bill_B_T1', 'bill_B_T2', 'bill_B_T3', 'bill_B_T4',
    'bill_C_T1', 'bill_C_T2', 'bill_C_T3', 'bill_C_T4',
    'bill_D_T1', 'bill_D_T2', 'bill_D_T3', 'bill_D_T4',
]].sum().reset_index()

for col, arch, tariff in [
    ('bill_A_T3', 'A', 'T3'),
    ('bill_B_T1', 'B', 'T1'), ('bill_B_T2', 'B', 'T2'),
    ('bill_B_T3', 'B', 'T3'), ('bill_B_T4', 'B', 'T4'),
    ('bill_C_T1', 'C', 'T1'), ('bill_C_T2', 'C', 'T2'),
    ('bill_C_T3', 'C', 'T3'), ('bill_C_T4', 'C', 'T4'),
    ('bill_D_T1', 'D', 'T1'), ('bill_D_T2', 'D', 'T2'),
    ('bill_D_T3', 'D', 'T3'), ('bill_D_T4', 'D', 'T4'),
]:
    monthly[col] = monthly[col] + get_annual_adjustment(arch, tariff)

print("\n=== PASSIVE BILL CHECKS (annual, averaged across 3 years) ===")
for col in ['bill_A_T1', 'bill_A_T2', 'bill_A_T3',
            'bill_B_T1', 'bill_B_T2', 'bill_B_T3', 'bill_B_T4',
            'bill_C_T1', 'bill_C_T2', 'bill_C_T3', 'bill_C_T4',
            'bill_D_T1', 'bill_D_T2', 'bill_D_T3', 'bill_D_T4']:
    annual = monthly.groupby('year')[col].sum().mean()
    print(f"{col}: €{annual:.2f}/yr")

# ============================================================
# AUTOMATED SCENARIO — LOAD SHIFTING
# T1 automated = passive (flat price, no shifting incentive)
# T2/T3 produce identical shifting schedules (Aufschlag is fixed
#   per-kWh and does not change hourly price ranking)
# T4 uses time-variable grid fees which change hourly price ranking
# ============================================================

sim_auto = sim.copy()
sim_auto['date'] = sim_auto.index.date
dates = sim_auto['date'].unique()


def compute_automated_loads(base_df, dates, price_col):
    """
    Compute automated load profiles for all archetypes.
    All shifting decisions use price_col for cheapest-hour selection.

    White goods (Archetype A):
      350 kWh/yr shiftable; passive hours 18-21; shifted to cheapest hour 00-24
      Source: methodology assumption

    Heat pump (Archetypes B, C):
      ±2h window (Gupta & Morey, 2023; IEA HPT Magazine, 2024)
      2h minimum interval (Tibber Node-RED, 2024)
      3 kWh/hr cap (Knoop et al., 2022)

    EV (Archetypes C, D):
      2,500 kWh/yr; passive hours 18-21 (Märtz et al., 2022)
      Plug-in window 18:00-07:00 (Spencer et al., 2021)
      2-night lookahead using day-ahead prices (Tibber, 2024)
    """
    df = base_df.copy()

    # --------------------------------------------------------
    # White goods shifting
    # 350 kWh/yr shiftable load; passive concentration 18-21
    # Shifted to cheapest hour in full day (00:00-24:00)
    # Source: methodology assumption
    # --------------------------------------------------------
    WG_DAILY = 350 / 365                    # kWh/day shiftable
    WG_PASSIVE_HOURS = [18, 19, 20, 21]     # default evening hours
    WG_PASSIVE_KWH = WG_DAILY / len(WG_PASSIVE_HOURS)

    df['load_A_auto'] = df['load_A'].copy()

    for d in dates:
        day_mask = df['date'] == d
        day_data = df[day_mask].copy()

        for h in WG_PASSIVE_HOURS:
            hour_mask = day_mask & (df['hour'] == h)
            df.loc[hour_mask, 'load_A_auto'] -= WG_PASSIVE_KWH

        cheapest_hour = day_data[price_col].idxmin()
        df.loc[cheapest_hour, 'load_A_auto'] += WG_DAILY

    # --------------------------------------------------------
    # HP shifting
    # 4,000 kWh/yr temperature-driven load (VDI 3807 Blatt 1, 2013)
    # ±2h window (Gupta & Morey, 2023; IEA HPT Magazine, 2024)
    # 2h minimum interval (Tibber Node-RED, 2024)
    # 3 kWh/hr cap (Knoop et al., 2022)
    # --------------------------------------------------------
    HP_MAX_KWH_HOUR = 3.0               # Knoop et al. (2022)
    HP_MIN_ADJUSTMENT_INTERVAL = 2      # Tibber Node-RED (2024)
    HP_WINDOW = 2                       # Gupta & Morey (2023)

    df['load_hp_auto'] = load_B['load_hp'].copy()

    for d in dates:
        day_mask = df['date'] == d
        day_data = df[day_mask].copy()
        hp_load = day_data['load_hp_auto'].copy()
        prices = day_data[price_col].copy()
        hours = day_data['hour'].values
        new_hp = hp_load.copy()
        last_adjustment_hour = -99

        for idx, h in zip(day_data.index, hours):
            if hp_load[idx] <= 0:
                continue
            hours_since_last = h - last_adjustment_hour
            if hours_since_last < HP_MIN_ADJUSTMENT_INTERVAL:
                continue
            window_hours = [hh % 24 for hh in range(h - HP_WINDOW, h + HP_WINDOW + 1)]
            window_mask = day_data['hour'].isin(window_hours)
            window_prices = prices[window_mask]
            if window_prices.empty:
                continue
            sorted_window = window_prices.sort_values()
            cheapest_idx = sorted_window.index[0]
            if cheapest_idx == idx:
                continue
            if new_hp[cheapest_idx] + hp_load[idx] > HP_MAX_KWH_HOUR:
                continue
            new_hp[idx] -= hp_load[idx]
            new_hp[cheapest_idx] += hp_load[idx]
            last_adjustment_hour = h

        df.loc[day_mask, 'load_hp_auto'] = new_hp.values

    df['load_B_auto'] = df['load_A_auto'] + df['load_hp_auto']

    # --------------------------------------------------------
    # EV shifting with 2-night lookahead
    # 2,500 kWh/yr; 6.85 kWh/day (Märtz et al., 2022)
    # Passive hours 18-21 (Märtz et al., 2022)
    # Plug-in window 18:00-07:00 (Spencer et al., 2021)
    # 2-night lookahead: defer if tomorrow night cheaper
    #   (day-ahead prices available via Tibber at ~12:30)
    # Cross-year deferral permitted — day-ahead prices available
    #   on Dec 31 for Jan 1; last day of dataset (31.12.2025)
    #   cannot defer as no following day exists in data
    # --------------------------------------------------------
    EV_ANNUAL_KWH = 2500.0                              # Märtz et al. (2022)
    EV_DAILY = EV_ANNUAL_KWH / 365                      # 6.85 kWh/day
    EV_PASSIVE_HOURS = [18, 19, 20, 21]                 # Märtz et al. (2022)
    EV_PASSIVE_KWH = EV_DAILY / len(EV_PASSIVE_HOURS)
    EV_AUTO_WINDOW = list(range(18, 24)) + list(range(0, 8))  # Spencer et al. (2021)

    df['load_ev_auto'] = (df['load_C'] - df['load_B']).copy()

    deferred_dates = set()

    for i, d in enumerate(dates):
        day_mask = df['date'] == d
        day_data = df[day_mask].copy()

        # Remove passive load from source hours 18-21
        for h in EV_PASSIVE_HOURS:
            hour_mask = day_mask & (df['hour'] == h)
            df.loc[hour_mask, 'load_ev_auto'] -= EV_PASSIVE_KWH

        # Skip if already handled by previous day's deferral
        if d in deferred_dates:
            continue

        # Tonight's overnight window (18:00-07:00)
        tonight_window = day_data[day_data['hour'].isin(EV_AUTO_WINDOW)]
        tonight_best_price = tonight_window[price_col].min() if not tonight_window.empty else np.inf
        tonight_best_idx = tonight_window[price_col].idxmin() if not tonight_window.empty else None

        # Tomorrow night's window — cross-year deferral permitted
        if i + 1 < len(dates):
            tomorrow = dates[i + 1]
            tomorrow_mask = df['date'] == tomorrow
            tomorrow_data = df[tomorrow_mask].copy()
            tomorrow_window = tomorrow_data[tomorrow_data['hour'].isin(EV_AUTO_WINDOW)]
            tomorrow_best_price = tomorrow_window[price_col].min() if not tomorrow_window.empty else np.inf
            tomorrow_best_idx = tomorrow_window[price_col].idxmin() if not tomorrow_window.empty else None
        else:
            tomorrow_best_price = np.inf
            tomorrow_best_idx = None

        # Charge on cheaper night
        if tomorrow_best_price < tonight_best_price and tomorrow_best_idx is not None:
            # Reverse today's passive removal — today stays neutral
            for h in EV_PASSIVE_HOURS:
                hour_mask = day_mask & (df['hour'] == h)
                df.loc[hour_mask, 'load_ev_auto'] += EV_PASSIVE_KWH
            df.loc[tomorrow_best_idx, 'load_ev_auto'] += EV_DAILY
            deferred_dates.add(tomorrow)
        else:
            if tonight_best_idx is not None:
                df.loc[tonight_best_idx, 'load_ev_auto'] += EV_DAILY

    df['load_C_auto'] = df['load_A_auto'] + df['load_hp_auto'] + df['load_ev_auto']
    df['load_D_auto'] = df['load_A_auto'] + df['load_ev_auto']

    return df[['load_A_auto', 'load_hp_auto', 'load_B_auto',
               'load_ev_auto', 'load_C_auto', 'load_D_auto']]


# ============================================================
# RUN SHIFTING FOR EACH TARIFF PRICE SIGNAL
# ============================================================
print("\nComputing automated loads for T2 (price signal: t2_ct_kwh)...")
auto_t2 = compute_automated_loads(sim_auto, dates, 't2_ct_kwh')

print("Computing automated loads for T3 (price signal: t3_ct_kwh)...")
auto_t3 = compute_automated_loads(sim_auto, dates, 't3_ct_kwh')

print("Computing automated loads for T4 (price signal: t4_ct_kwh)...")
auto_t4 = compute_automated_loads(sim_auto, dates, 't4_ct_kwh')

# ============================================================
# LOAD CHECKS
# ============================================================
print("\n=== AUTOMATED LOAD CHECKS (all should match passive totals) ===")
expected = {'A': 3500, 'B': 7500, 'C': 10000, 'D': 6000}
for label, auto_df in [('T2', auto_t2), ('T3', auto_t3), ('T4', auto_t4)]:
    for year in [2023, 2024, 2025]:
        year_mask = sim_auto['year'] == year
        a = auto_df[year_mask]['load_A_auto'].sum()
        b = auto_df[year_mask]['load_B_auto'].sum()
        c = auto_df[year_mask]['load_C_auto'].sum()
        d = auto_df[year_mask]['load_D_auto'].sum()
        print(f"  {label} {year}: A={a:.0f} (exp {expected['A']}) "
              f"B={b:.0f} (exp {expected['B']}) "
              f"C={c:.0f} (exp {expected['C']}) "
              f"D={d:.0f} (exp {expected['D']})")

# ============================================================
# AUTOMATED BILLS
# T1 automated = passive (flat price, no shifting incentive)
# T2 automated = shifted on T2 prices, billed at T2
# T3 automated = shifted on T3 prices, billed at T3
# T4 automated = shifted on T4 prices, billed at T4
# ============================================================

# T1 automated = passive
sim_auto['bill_A_T1_auto'] = sim_auto['load_A'] * sim_auto['t1_ct_kwh'] / 100
sim_auto['bill_B_T1_auto'] = sim_auto['load_B'] * sim_auto['t1_ct_kwh'] / 100
sim_auto['bill_C_T1_auto'] = sim_auto['load_C'] * sim_auto['t1_ct_kwh'] / 100
sim_auto['bill_D_T1_auto'] = sim_auto['load_D'] * sim_auto['t1_ct_kwh'] / 100

# T2 automated
sim_auto['bill_A_T2_auto'] = auto_t2['load_A_auto'] * sim_auto['t2_ct_kwh'] / 100
sim_auto['bill_B_T2_auto'] = auto_t2['load_B_auto'] * sim_auto['t2_ct_kwh'] / 100
sim_auto['bill_C_T2_auto'] = auto_t2['load_C_auto'] * sim_auto['t2_ct_kwh'] / 100
sim_auto['bill_D_T2_auto'] = auto_t2['load_D_auto'] * sim_auto['t2_ct_kwh'] / 100

# T3 automated
sim_auto['bill_A_T3_auto'] = auto_t3['load_A_auto'] * sim_auto['t3_ct_kwh'] / 100
sim_auto['bill_B_T3_auto'] = auto_t3['load_B_auto'] * sim_auto['t3_ct_kwh'] / 100
sim_auto['bill_C_T3_auto'] = auto_t3['load_C_auto'] * sim_auto['t3_ct_kwh'] / 100
sim_auto['bill_D_T3_auto'] = auto_t3['load_D_auto'] * sim_auto['t3_ct_kwh'] / 100

# T4 automated
sim_auto['bill_B_T4_auto'] = auto_t4['load_B_auto'] * sim_auto['t4_ct_kwh'] / 100
sim_auto['bill_C_T4_auto'] = auto_t4['load_C_auto'] * sim_auto['t4_ct_kwh'] / 100
sim_auto['bill_D_T4_auto'] = auto_t4['load_D_auto'] * sim_auto['t4_ct_kwh'] / 100

# ============================================================
# AGGREGATE AUTOMATED TO MONTHLY + ADD FLAT ADJUSTMENTS
# ============================================================
auto_bill_cols = [
    'bill_A_T1_auto', 'bill_A_T2_auto', 'bill_A_T3_auto',
    'bill_B_T1_auto', 'bill_B_T2_auto', 'bill_B_T3_auto', 'bill_B_T4_auto',
    'bill_C_T1_auto', 'bill_C_T2_auto', 'bill_C_T3_auto', 'bill_C_T4_auto',
    'bill_D_T1_auto', 'bill_D_T2_auto', 'bill_D_T3_auto', 'bill_D_T4_auto',
]

monthly_auto = sim_auto.groupby(['year', 'month'])[auto_bill_cols].sum().reset_index()

for col, arch, tariff in [
    ('bill_A_T3_auto', 'A', 'T3'),
    ('bill_B_T1_auto', 'B', 'T1'), ('bill_B_T2_auto', 'B', 'T2'),
    ('bill_B_T3_auto', 'B', 'T3'), ('bill_B_T4_auto', 'B', 'T4'),
    ('bill_C_T1_auto', 'C', 'T1'), ('bill_C_T2_auto', 'C', 'T2'),
    ('bill_C_T3_auto', 'C', 'T3'), ('bill_C_T4_auto', 'C', 'T4'),
    ('bill_D_T1_auto', 'D', 'T1'), ('bill_D_T2_auto', 'D', 'T2'),
    ('bill_D_T3_auto', 'D', 'T3'), ('bill_D_T4_auto', 'D', 'T4'),
]:
    monthly_auto[col] = monthly_auto[col] + get_annual_adjustment(arch, tariff)

print("\n=== AUTOMATED BILL CHECKS (annual, 3yr avg) ===")
for col in auto_bill_cols:
    annual = monthly_auto.groupby('year')[col].sum().mean()
    print(f"{col}: €{annual:.2f}/yr")

# ============================================================
# BUILD LONG-FORMAT bills.csv
# ============================================================
passive_map = {
    ('A','T1'): 'bill_A_T1', ('A','T2'): 'bill_A_T2', ('A','T3'): 'bill_A_T3',
    ('B','T1'): 'bill_B_T1', ('B','T2'): 'bill_B_T2',
    ('B','T3'): 'bill_B_T3', ('B','T4'): 'bill_B_T4',
    ('C','T1'): 'bill_C_T1', ('C','T2'): 'bill_C_T2',
    ('C','T3'): 'bill_C_T3', ('C','T4'): 'bill_C_T4',
    ('D','T1'): 'bill_D_T1', ('D','T2'): 'bill_D_T2',
    ('D','T3'): 'bill_D_T3', ('D','T4'): 'bill_D_T4',
}
auto_map = {k: v + '_auto' for k, v in passive_map.items()}

rows = []
for (arch, tariff), col in passive_map.items():
    for _, row in monthly.iterrows():
        rows.append({'archetype': arch, 'tariff': tariff, 'scenario': 'passive',
                     'year': int(row['year']), 'month': int(row['month']),
                     'monthly_bill_eur': row[col]})

for (arch, tariff), col in auto_map.items():
    for _, row in monthly_auto.iterrows():
        rows.append({'archetype': arch, 'tariff': tariff, 'scenario': 'automated',
                     'year': int(row['year']), 'month': int(row['month']),
                     'monthly_bill_eur': row[col]})

bills = (pd.DataFrame(rows)
           .sort_values(['archetype', 'tariff', 'scenario', 'year', 'month'])
           .reset_index(drop=True))

print(f"\n=== BILLS.CSV SHAPE ===")
print(f"Total rows: {len(bills)} (expected 1080)")
print(bills.groupby(['archetype', 'scenario'])['tariff'].count().rename('rows'))

bills.to_csv('data_simulation/bills.csv', index=False)
print("\nSaved: data_simulation/bills.csv")

# ============================================================
# VERIFICATION CHECKS
# ============================================================
print("\n" + "="*60)
print("VERIFICATION CHECKS")
print("="*60)

# CHECK 1: T2 and T3 load profiles should be identical
print("\n--- CHECK 1: T2 vs T3 load profiles identical ---")
for col in ['load_A_auto', 'load_B_auto', 'load_C_auto', 'load_D_auto']:
    diff = (auto_t2[col] - auto_t3[col]).abs().max()
    status = "PASS" if diff < 0.001 else "FAIL"
    print(f"  {col}: max diff = {diff:.6f} [{status}]")

# CHECK 2: Automated bill <= passive bill per tariff
print("\n--- CHECK 2: Automated bill <= passive bill per tariff ---")
check2_pairs = [
    ('bill_A_T2_auto', 'bill_A_T2'),
    ('bill_A_T3_auto', 'bill_A_T3'),
    ('bill_B_T2_auto', 'bill_B_T2'),
    ('bill_B_T3_auto', 'bill_B_T3'),
    ('bill_B_T4_auto', 'bill_B_T4'),
    ('bill_C_T2_auto', 'bill_C_T2'),
    ('bill_C_T3_auto', 'bill_C_T3'),
    ('bill_C_T4_auto', 'bill_C_T4'),
    ('bill_D_T2_auto', 'bill_D_T2'),
    ('bill_D_T3_auto', 'bill_D_T3'),
    ('bill_D_T4_auto', 'bill_D_T4'),
]
for auto_col, pass_col in check2_pairs:
    auto_annual = monthly_auto.groupby('year')[auto_col].sum().mean()
    pass_annual = monthly.groupby('year')[pass_col].sum().mean()
    status = "PASS" if auto_annual <= pass_annual + 0.01 else "FAIL"
    print(f"  {auto_col}: auto={auto_annual:.2f} passive={pass_annual:.2f} [{status}]")

# CHECK 3: EV deferral count per year per tariff
print("\n--- CHECK 3: EV deferral days per year per tariff ---")
for label, price_col in [('T2', 't2_ct_kwh'), ('T3', 't3_ct_kwh'), ('T4', 't4_ct_kwh')]:
    df_check = sim_auto.copy()
    EV_AUTO_WINDOW_CHECK = list(range(18, 24)) + list(range(0, 8))
    deferred_count = {2023: 0, 2024: 0, 2025: 0}
    dates_check = df_check['date'].unique()
    deferred_set = set()
    for i, d in enumerate(dates_check):
        day_mask = df_check['date'] == d
        day_data = df_check[day_mask]
        year = day_data['year'].iloc[0]
        if d in deferred_set:
            continue
        tonight_window = day_data[day_data['hour'].isin(EV_AUTO_WINDOW_CHECK)]
        tonight_best_price = tonight_window[price_col].min() if not tonight_window.empty else np.inf
        if i + 1 < len(dates_check):
            tomorrow = dates_check[i + 1]
            tomorrow_data = df_check[df_check['date'] == tomorrow]
            tomorrow_window = tomorrow_data[tomorrow_data['hour'].isin(EV_AUTO_WINDOW_CHECK)]
            tomorrow_best_price = tomorrow_window[price_col].min() if not tomorrow_window.empty else np.inf
            if tomorrow_best_price < tonight_best_price:
                deferred_count[year] += 1
                deferred_set.add(tomorrow)
    print(f"  {label}: {deferred_count}")

# CHECK 4: EV destination hour distribution T3 vs T4
print("\n--- CHECK 4: EV destination hours T3 vs T4 (NT hours 23-6 higher for T4) ---")
for label, auto_df in [('T3', auto_t3), ('T4', auto_t4)]:
    ev_passive = (sim_auto['load_C'] - sim_auto['load_B']).values
    ev_auto = auto_df['load_ev_auto'].values
    diff = ev_auto - ev_passive
    dest_mask = diff > 0.001
    dest_hours = sim_auto['hour'].values[dest_mask]
    nt_hours = [23, 0, 1, 2, 3, 4, 5, 6]
    nt_pct = (np.isin(dest_hours, nt_hours).sum() / len(dest_hours) * 100)
    print(f"  {label}: NT hours (23-6) = {nt_pct:.1f}% of destinations")

# CHECK 5: HP weighted price T3 vs T4
print("\n--- CHECK 5: HP weighted price T3 vs T4 shifting ---")
for year in [2023, 2024, 2025]:
    year_mask = sim_auto['year'] == year
    prices_t3 = sim_auto[year_mask]['t3_ct_kwh'].values
    prices_t4 = sim_auto[year_mask]['t4_ct_kwh'].values
    hp_t3 = auto_t3[year_mask]['load_hp_auto'].values
    hp_t4 = auto_t4[year_mask]['load_hp_auto'].values
    wt3 = (prices_t3 * hp_t3).sum() / hp_t3.sum()
    wt4 = (prices_t4 * hp_t4).sum() / hp_t4.sum()
    print(f"  {year}: HP weighted price T3-shifted={wt3:.2f} | T4-shifted={wt4:.2f}")

print("\n--- HP LOAD WEIGHTED PRICE CHECK (T3 shifting) ---")
for year in [2023, 2024, 2025]:
    year_mask = sim_auto['year'] == year
    subset = sim_auto[year_mask].copy()
    hp_passive = load_B[year_mask]['load_hp'].values
    passive_weighted = (subset['t3_ct_kwh'].values * hp_passive).sum() / hp_passive.sum()
    hp_auto = auto_t3[year_mask]['load_hp_auto'].values
    auto_weighted = (subset['t3_ct_kwh'].values * hp_auto).sum() / hp_auto.sum()
    flat = 37.0
    print(f"\n{year}:")
    print(f"  HP passive weighted avg T3 price: {passive_weighted:.2f} ct/kWh")
    print(f"  HP automated weighted avg T3 price: {auto_weighted:.2f} ct/kWh")
    print(f"  Flat tariff: {flat:.2f} ct/kWh")
    print(f"  Gap passive vs flat: {passive_weighted - flat:.2f} ct/kWh")