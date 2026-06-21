"""
Simulation Inputs — Pipeline 2a
Builds hourly tariff price series (T1-T4) and archetype load profiles (A, B, C, D).
Flat annual adjustments (Grundgebühr, Module 1 reduction) are added later in 05_simulation.py.

Tariffs:
  T1      — Flat: year-specific retail price (BDEW Strompreise-Dossier, Stand 04/2026)
  T2      — Dynamic theoretical: hourly EPEX + year-specific grid + taxes, no provider fee
  T3_TIB  — Dynamic retail: T2 + Tibber Aufschlag (2.15 ct/kWh netto)
  T3_OCT  — Dynamic retail: T2 + Octopus Aufschlag (0 ct/kWh netto)
  T3_AWA  — Dynamic retail: T2 + aWATTar Beschaffungskomponente (1.785 ct/kWh netto)
  T4      — Dynamic retail + grid: T3_TIB + time-variable grid fees (Stromnetz Berlin §14a Module 3)

Provider fee sources (primary, June 2026):
  Tibber:  2.15 ct/kWh Aufschlag + €5.99/month
           https://support.tibber.com/de/articles/12310314-grund-und-arbeitspreis-bei-tibber
  Octopus: 0 ct/kWh Aufschlag + €10.13/month Grundpreis
           dynamicOctopus tariff checkout, Berlin 10585, June 2026
  aWATTar: 1.785 ct/kWh Beschaffungskomponente + €4.58/month Grundgebühr
           tado° Home Hourly Tarifblatt, Berlin 10585, 17.06.2026

Note on T3 price signals for load shifting:
  T3_TIB, T3_OCT, T3_AWA differ only by a fixed per-kWh Aufschlag and flat Grundgebühr.
  A fixed Aufschlag does not change the hourly price ranking (cheapest hour remains cheapest).
  Therefore all T3 variants share the same automated shifting schedule (computed using
  t3_tib_ct_kwh as the price signal in 05_simulation.py). Only billing differs.

Note on T4 2023:
  §14a Module 3 variable grid fees came into force April 2024. T4 for 2023 is included
  as a hypothetical lower bound only.

Note on year-specific components:
  T1, grid fees and taxes vary by year using BDEW Strompreise-Dossier annual averages.
  Provider fees (Aufschläge, Grundgebühren) held constant at 2026 values as historical
  provider fee data not publicly available. T4 Module 3 fees held constant at 2026 Stromnetz Berlin values.

Archetypes:
  A — Basic household (3,500 kWh/yr): BDEW H25 base load, Dynamisierung applied
  B — Heat pump (7,500 kWh/yr): A + temperature-driven HP load (4,000 kWh/yr)
  C — EV only (6,000 kWh/yr): A + passive EV charging load (2,500 kWh/yr)
  D — HP + EV (10,000 kWh/yr): A + temperature-driven HP load (4,000 kWh/yr) + passive EV charging load (2,500 kWh/yr)

Key sources:
  BDEW Strompreise-Dossier (Stand 04/2026) — year-specific flat tariff and fixed component stack
  Tibber, Octopus, aWATTar (June 2026) — Aufschlag and Grundgebühr
  Stromnetz Berlin Preisblatt 2026 — §14a Module 3 time-variable grid fees
  BDEW H25 standard load profile (March 2025) — base load shape + Dynamisierung coefficients
  VDI 3807 Blatt 1 (2013) / DWD — heating threshold 15°C
  Märtz et al. (2022) — EV passive charging hours

Outputs:
  data_simulation/tariff_prices.csv — hourly tariff prices for all tariffs
  data_simulation/load_profile_A.csv — hourly load profile Archetype A
  data_simulation/load_profile_B.csv — hourly load profile Archetype B (incl. load_hp)
  data_simulation/load_profile_C.csv — hourly load profile Archetype C (incl. load_ev)
  data_simulation/load_profile_D.csv — hourly load profile Archetype D (incl. load_hp, load_ev)
"""

import pandas as pd
import numpy as np
import os

os.makedirs('data_simulation', exist_ok=True)

# ============================================================
# LOAD MASTER DATASET
# ============================================================
master = pd.read_csv('data_clean/master_hourly.csv', index_col='datetime', parse_dates=True)
print(f"Loaded master: {len(master)} rows")

# ============================================================
# ANNUAL CONSUMPTION TARGETS
# ============================================================
# A: 3,500 kWh/yr — BDEW Musterhaushalt, 3-person single-family house
#    Source: BDEW Strompreisanalyse April 2026; BDEW Zahl der Woche (household size)
#
# B: 7,500 kWh/yr — A + 4,000 kWh/yr HP; WPuQ median (n=30 German SFH)
#    Source: Schlemminger et al. (2022), Scientific Data
#
# C: 6,000 kWh/yr — A + 2,500 kWh/yr EV; based on KBA average annual mileage
#    Source: KBA Verkehr in Kilometern
#
# D: 10,000 kWh/yr — A + HP + EV combined
ANNUAL_KWH_A = 3500
ANNUAL_KWH_B = 7500
ANNUAL_KWH_C = 6000
ANNUAL_KWH_D = 10000

ANNUAL_HP_KWH = ANNUAL_KWH_B - ANNUAL_KWH_A   # 4,000 kWh/yr HP component
ANNUAL_EV_KWH = ANNUAL_KWH_C - ANNUAL_KWH_A   # 2,500 kWh/yr EV component

# ============================================================
# TARIFF CONSTRUCTION
# Year-specific values reflect actual household electricity price components.
# Provider fees (Aufschläge, Grundgebühren) held constant at 2026 values.
# T4 Module 3 grid fees held constant at 2026 Stromnetz Berlin values;
# Module 3 only available from April 2024 — T4 2023 is hypothetical.
# ============================================================

# Year-specific flat tariff and fixed cost components (ct/kWh brutto)
# Source: BDEW Strompreise-Dossier, Stand 04/2026
T1_CT           = {2023: 47.0, 2024: 40.2, 2025: 39.3}
GRID_FEE_CT     = {2023:  9.3, 2024: 11.4, 2025: 10.9}
TAXES_LEVIES_CT = {2023: 12.6, 2024: 11.7, 2025: 12.6}

VAT = 1.19  # 19% VAT multiplier

# Provider Aufschläge — netto (VAT applied in tariff construction below)
# Grundgebühren — brutto annual totals; applied as flat monthly charges in 05_simulation.py
AUFSCHLAG_TIB    = 2.15    # Tibber: ct/kWh netto; Tibber April 2026
GRUNDGEBUEHR_TIB = 71.88   # Tibber: €/yr brutto (€5.99/month × 12)

AUFSCHLAG_OCT    = 0.0     # Octopus: ct/kWh netto; dynamicOctopus, Berlin 10585, June 2026
GRUNDGEBUEHR_OCT = 121.56  # Octopus: €/yr brutto (€10.13/month × 12)

AUFSCHLAG_AWA    = 1.785   # aWATTar: ct/kWh netto; Tarifblatt 17.06.2026
GRUNDGEBUEHR_AWA = 54.96   # aWATTar: €/yr brutto (€4.58/month × 12)

# Module 1 flat annual reduction — applied in 05_simulation.py for Archetypes B, C, D
# Source: Stromnetz Berlin Preisblatt 2026, §14a Module 1
MODULE1_REDUCTION_EUR = 123.18  # €/yr netto

# Module 3 time-variable grid fees (netto ct/kWh)
# Source: Stromnetz Berlin Preisblatt 2026 (valid from 01.01.2026)
# NT (Niedriglasttarifzeit): 22:15-06:30 → hours 23, 0, 1, 2, 3, 4, 5, 6
# HT (Hochlasttarifzeit):    17:15-20:15 → hours 17, 18, 19, 20
# ST (Standardlasttarifzeit): all other hours

MODULE3_HT = 13.94  # ct/kWh netto
MODULE3_ST = 7.46   # ct/kWh netto
MODULE3_NT = 2.61   # ct/kWh netto

# Convert EPEX from €/MWh to ct/kWh
master['epex_ct_kwh'] = master['price_eur_mwh'] / 10

# Apply year-specific cost components
master['grid_fee_ct']     = master['year'].map(GRID_FEE_CT)
master['taxes_levies_ct'] = master['year'].map(TAXES_LEVIES_CT)
master['t1_ct_kwh']       = master['year'].map(T1_CT)

# --- T2: Dynamic theoretical ---
# Hourly EPEX + year-specific grid + taxes netto, then VAT
# No provider fee — theoretical lower bound
master['t2_ct_kwh'] = (master['epex_ct_kwh'] + master['grid_fee_ct'] + master['taxes_levies_ct']) * VAT

# --- T3_TIB: Dynamic retail Tibber ---
# T2 + Tibber Aufschlag netto, then VAT
# Grundgebühr added as flat monthly charge in 05_simulation.py
master['t3_tib_ct_kwh'] = (master['epex_ct_kwh'] + master['grid_fee_ct'] + master['taxes_levies_ct'] + AUFSCHLAG_TIB) * VAT

# --- T3_OCT: Dynamic retail Octopus ---
# Aufschlag = 0; hourly price identical to T2; Grundgebühr differs (added in 05_simulation.py)
master['t3_oct_ct_kwh'] = (master['epex_ct_kwh'] + master['grid_fee_ct'] + master['taxes_levies_ct'] + AUFSCHLAG_OCT) * VAT

# --- T3_AWA: Dynamic retail aWATTar ---
# Aufschlag 1.785 ct/kWh netto; Grundgebühr added in 05_simulation.py
master['t3_awa_ct_kwh'] = (master['epex_ct_kwh'] + master['grid_fee_ct'] + master['taxes_levies_ct'] + AUFSCHLAG_AWA) * VAT

# --- T4: Dynamic retail + time-variable grid fee (Module 3) ---
# Replaces fixed grid fee with Stromnetz Berlin HT/ST/NT schedule
# Applies to Archetypes B, C, D only (§14a device + smart meter required)
# Uses Tibber fees only as primary provider and to compare directly to T3 shown in dissertation
def module3_grid_fee(hour):
    if hour in [23, 0, 1, 2, 3, 4, 5, 6]:
        return MODULE3_NT
    elif hour in [17, 18, 19, 20]:
        return MODULE3_HT
    else:
        return MODULE3_ST

master['module3_grid_ct'] = master['hour'].apply(module3_grid_fee)
master['t4_ct_kwh'] = (master['epex_ct_kwh'] + master['module3_grid_ct'] + master['taxes_levies_ct'] + AUFSCHLAG_TIB) * VAT

# ============================================================
# TARIFF CHECKS
# ============================================================
print("\n=== TARIFF CHECKS ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    print(f"\n{year}:")
    print(f"  T1: {subset['t1_ct_kwh'].mean():.2f} ct/kWh (should be {T1_CT[year]})")
    print(f"  Grid fee: {subset['grid_fee_ct'].mean():.2f} ct/kWh (should be {GRID_FEE_CT[year]})")
    print(f"  Taxes: {subset['taxes_levies_ct'].mean():.2f} ct/kWh (should be {TAXES_LEVIES_CT[year]})")
    print(f"  T2 mean: {subset['t2_ct_kwh'].mean():.2f} ct/kWh")
    print(f"  T2 min:  {subset['t2_ct_kwh'].min():.2f} ct/kWh")
    print(f"  T2 max:  {subset['t2_ct_kwh'].max():.2f} ct/kWh")
    print(f"  T3_TIB mean: {subset['t3_tib_ct_kwh'].mean():.2f} ct/kWh")
    print(f"  T3_OCT mean: {subset['t3_oct_ct_kwh'].mean():.2f} ct/kWh (should equal T2: Aufschlag=0)")
    print(f"  T3_AWA mean: {subset['t3_awa_ct_kwh'].mean():.2f} ct/kWh")
    print(f"  T4 mean: {subset['t4_ct_kwh'].mean():.2f} ct/kWh")

# Confirm T3_OCT == T2 (zero Aufschlag)
diff_oct_t2 = (master['t3_oct_ct_kwh'] - master['t2_ct_kwh']).abs().max()
print(f"\nT3_OCT vs T2 max diff: {diff_oct_t2:.6f} (should be 0.0)")

# Confirm ranking preserved across T3 variants (fixed Aufschlag does not change hourly ranking)
rank_check = master[['t3_tib_ct_kwh', 't3_oct_ct_kwh', 't3_awa_ct_kwh']].rank(axis=0)
rank_same = (rank_check['t3_tib_ct_kwh'] == rank_check['t3_oct_ct_kwh']).all()
print(f"T3 hourly ranking identical across providers: {rank_same} (should be True)")

# ============================================================
# SAVE TARIFF PRICES
# ============================================================
tariff_prices = master[['epex_ct_kwh', 't1_ct_kwh', 't2_ct_kwh',
                         't3_tib_ct_kwh', 't3_oct_ct_kwh', 't3_awa_ct_kwh',
                         't4_ct_kwh', 'module3_grid_ct',
                         'grid_fee_ct', 'taxes_levies_ct',
                         'year', 'month', 'hour', 'dayofweek']].copy()
tariff_prices.to_csv('data_simulation/tariff_prices.csv')
print(f"\nSaved: data_simulation/tariff_prices.csv ({len(tariff_prices)} rows)")

# ============================================================
# LOAD PROFILES — BASE SETUP
# ============================================================

# Load H25 standard load profile
# Source: BDEW H25 standard load profile
h25 = pd.read_csv('data_clean/h25_hourly_profile.csv')

# Map day of week to BDEW day types
# 0-4 = Monday-Friday = WT (Werktag)
# 5 = Saturday = SA
# 6 = Sunday/Holiday = FT (Feiertag/Sonntag)
def get_day_type(dow):
    if dow == 5:
        return 'SA'
    elif dow == 6:
        return 'FT'
    else:
        return 'WT'

master['day_type'] = master['dayofweek'].apply(get_day_type)

# Apply Dynamisierung polynomial
# Seasonal adjustment factor; coefficients from BDEW H25 profile documentation
# t = day of year (1-365)
master['day_of_year'] = master.index.dayofyear.astype(float)
master['dynamisierung'] = (-3.92e-10 * master['day_of_year']**4
                          + 3.20e-7  * master['day_of_year']**3
                          - 7.02e-5  * master['day_of_year']**2
                          + 2.10e-3  * master['day_of_year']
                          + 1.24)

# Reset index before merge to preserve datetime as column
master = master.reset_index()

# Merge H2 base values by month, day_type, hour
master = master.merge(h25, on=['month', 'day_type', 'hour'], how='left')
master = master.rename(columns={'load_kwh': 'h25_base'})
master = master.set_index('datetime')

master['h25_dynamic'] = master['h25_base'] * master['dynamisierung']

print("\n=== DYNAMISIERUNG CHECK ===")
print(f"Mean multiplier: {master['dynamisierung'].mean():.4f} (should be ~1.0)")
print(f"Jan 1 multiplier: {master.loc[master['day_of_year']==1, 'dynamisierung'].mean():.4f} (should be ~1.24)")
print(f"Jul 1 multiplier: {master.loc[master['day_of_year']==182, 'dynamisierung'].mean():.4f} (should be ~0.8)")
print(f"Missing h25_base values: {master['h25_base'].isna().sum()}")

# ============================================================
# ARCHETYPE A — Base load only (3,500 kWh/yr)
# BDEW H25 profile with Dynamisierung, scaled per year
# ============================================================
for year in [2023, 2024, 2025]:
    year_mask = master['year'] == year
    annual_sum = master.loc[year_mask, 'h25_dynamic'].sum()
    scaling_factor = ANNUAL_KWH_A / annual_sum
    master.loc[year_mask, 'load_A'] = master.loc[year_mask, 'h25_dynamic'] * scaling_factor

print("\n=== ARCHETYPE A CHECK ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    print(f"{year}: annual load = {subset['load_A'].sum():.1f} kWh (should be {ANNUAL_KWH_A})")

print("\n=== PROFILE SHAPE CHECK ARCHETYPE A ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    winter = subset[subset['month'].isin([1, 2, 3])]['load_A'].sum()
    summer = subset[subset['month'].isin([6, 7, 8])]['load_A'].sum()
    print(f"{year}: winter (Jan-Mar) = {winter:.1f} kWh, summer (Jun-Aug) = {summer:.1f} kWh, ratio = {winter/summer:.2f}")

load_A = master[['load_A', 'year', 'month', 'hour', 'dayofweek']].copy()
load_A.to_csv('data_simulation/load_profile_A.csv')
print(f"Saved: data_simulation/load_profile_A.csv")

# ============================================================
# ARCHETYPE B — Base load + heat pump (7,500 kWh/yr)
# HP load proportional to heating degree hours below 15°C
# Source: VDI 3807 Blatt 1 (2013); DWD station Potsdam (ID 03987)
# ============================================================
HEATING_THRESHOLD = 15.0  # °C; VDI 3807 Blatt 1 (2013), confirmed by DWD

master['heating_degrees'] = (HEATING_THRESHOLD - master['temperature_c']).clip(lower=0)

for year in [2023, 2024, 2025]:
    year_mask = master['year'] == year
    annual_hd_sum = master.loc[year_mask, 'heating_degrees'].sum()
    hp_scaling_factor = ANNUAL_HP_KWH / annual_hd_sum
    master.loc[year_mask, 'load_hp'] = master.loc[year_mask, 'heating_degrees'] * hp_scaling_factor

master['load_B'] = master['load_A'] + master['load_hp']

print("\n=== ARCHETYPE B CHECK ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    print(f"{year}: total = {subset['load_B'].sum():.1f} kWh "
          f"(base = {subset['load_A'].sum():.1f}, HP = {subset['load_hp'].sum():.1f})")

print("\n=== PROFILE SHAPE CHECK ARCHETYPE B ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    winter = subset[subset['month'].isin([1, 2, 3])]['load_B'].sum()
    summer = subset[subset['month'].isin([6, 7, 8])]['load_B'].sum()
    print(f"{year}: winter (Jan-Mar) = {winter:.1f} kWh, summer (Jun-Aug) = {summer:.1f} kWh, ratio = {winter/summer:.2f}")

load_B = master[['load_B', 'load_A', 'load_hp', 'year', 'month', 'hour', 'dayofweek']].copy()
load_B.to_csv('data_simulation/load_profile_B.csv')
print(f"Saved: data_simulation/load_profile_B.csv")

# ============================================================
# ARCHETYPE C — Base load + EV only (6,000 kWh/yr)
# Märtz et al. (2022), plug-in times concentrate between 18:00 and 21:59 from German EV charging data
# ============================================================
EV_CHARGING_HOURS = [18, 19, 20, 21]

master['load_ev_raw'] = master['hour'].apply(
    lambda h: 1.0 if h in EV_CHARGING_HOURS else 0.0
)

for year in [2023, 2024, 2025]:
    year_mask = master['year'] == year
    annual_ev_sum = master.loc[year_mask, 'load_ev_raw'].sum()
    ev_scaling_factor = ANNUAL_EV_KWH / annual_ev_sum
    master.loc[year_mask, 'load_ev'] = master.loc[year_mask, 'load_ev_raw'] * ev_scaling_factor

master['load_C'] = master['load_A'] + master['load_ev']

print("\n=== ARCHETYPE C CHECK ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    print(f"{year}: total = {subset['load_C'].sum():.1f} kWh (should be {ANNUAL_KWH_C}) "
          f"(base = {subset['load_A'].sum():.1f}, EV = {subset['load_ev'].sum():.1f})")

print("\n=== PROFILE SHAPE CHECK ARCHETYPE C ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    winter = subset[subset['month'].isin([1, 2, 3])]['load_C'].sum()
    summer = subset[subset['month'].isin([6, 7, 8])]['load_C'].sum()
    ev_winter = subset[subset['month'].isin([1, 2, 3])]['load_ev'].sum()
    ev_summer = subset[subset['month'].isin([6, 7, 8])]['load_ev'].sum()
    print(f"{year}: winter = {winter:.1f} kWh, summer = {summer:.1f} kWh, ratio = {winter/summer:.2f}")
    print(f"  EV: winter = {ev_winter:.1f} kWh, summer = {ev_summer:.1f} kWh, ratio = {ev_winter/ev_summer:.2f}")

load_C = master[['load_C', 'load_A', 'load_ev', 'year', 'month', 'hour', 'dayofweek']].copy()
load_C.to_csv('data_simulation/load_profile_C.csv')
print(f"Saved: data_simulation/load_profile_C.csv")

# ============================================================
# ARCHETYPE D — Base load + heat pump + EV (10,000 kWh/yr)
# Combines HP and EV components; uses load_hp from Archetype B
# and load_ev from Archetype C
# ============================================================
master['load_D'] = master['load_A'] + master['load_hp'] + master['load_ev']

print("\n=== ARCHETYPE D CHECK ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    total = subset['load_D'].sum()
    base = subset['load_A'].sum()
    hp = subset['load_hp'].sum()
    ev = subset['load_ev'].sum()
    print(f"{year}: total = {total:.1f} kWh (should be {ANNUAL_KWH_D}) "
          f"(base = {base:.1f}, HP = {hp:.1f}, EV = {ev:.1f})")

load_D = master[['load_D', 'load_A', 'load_hp', 'load_ev', 'year', 'month', 'hour', 'dayofweek']].copy()
load_D.to_csv('data_simulation/load_profile_D.csv')
print(f"Saved: data_simulation/load_profile_D.csv")

print("\nDone.")