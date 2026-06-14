"""
Data Cleaning — Pipeline 1a
Reads 5 raw data sources, cleans and merges into master hourly dataset.

Outputs:
  data_clean/master_hourly.csv — 26,301 hourly rows, 2023-2025
    (26,304 hours in 3 years minus 3 spring-forward DST hours with no price data)
  data_clean/H25_hourly_profile.csv — 864 rows (12 months x 3 day types x 24 hours)

DST handling:
  Spring forward (March): 3 hours per year do not exist in local time (CET/CEST);
    SMARD data has no price entry for these hours → NaN → dropped via dropna
  Fall back (October): 1 hour per year appears twice in local time;
    duplicate removed via keep='first'

Sources:
  01_day_ahead_prices.csv — EPEX Spot Day-Ahead DE/LU, SMARD.de
    https://www.smard.de/home/downloadcenter/download-marktdaten/
  02_realised_generation.csv — Realised generation by source, SMARD.de
    https://www.smard.de/home/downloadcenter/download-marktdaten/
  03_realised_consumption.csv — Realised consumption, SMARD.de
    https://www.smard.de/home/downloadcenter/download-marktdaten/
  04_temperature_potsdam.txt — Hourly air temperature, DWD station Potsdam (ID 03987)
    https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/air_temperature/
  05_bdew_standard_load_profiles.xlsx — BDEW H0 standard load profile (March 2025)
    https://www.bdew.de/energie/standardlastprofile-strom/
"""

import pandas as pd
import os

os.makedirs('data_clean', exist_ok=True)

# ============================================================
# 1. DAY-AHEAD PRICES
# Source: SMARD.de, EPEX Spot Day-Ahead DE/LU market area
# ============================================================
prices = pd.read_csv('data_raw/01_day_ahead_prices.csv',
                     sep=';', encoding='utf-8-sig')

prices['datetime'] = pd.to_datetime(prices.iloc[:, 0], format='%d.%m.%Y %H:%M')

# DE/LU price column — handle German number format (. = thousands, , = decimal)
price_col = [c for c in prices.columns if 'Deutschland/Luxemburg' in c][0]
prices['price_eur_mwh'] = (prices[price_col]
                           .str.replace('.', '', regex=False)
                           .str.replace(',', '.', regex=False)
                           .astype(float))

prices = prices[['datetime', 'price_eur_mwh']].set_index('datetime')
prices = prices.loc['2023':'2025']

print("=== 1. PRICES ===")
print(f"Rows: {len(prices)}")
print(f"Range: {prices.index.min()} to {prices.index.max()}")
print(f"Missing values: {prices['price_eur_mwh'].isna().sum()}")
print(f"Mean: {prices['price_eur_mwh'].mean():.2f} €/MWh")
print(f"Min: {prices['price_eur_mwh'].min():.2f}, Max: {prices['price_eur_mwh'].max():.2f}")
print(f"Negative price hours: {(prices['price_eur_mwh'] < 0).sum()}")
print()

for year in [2023, 2024, 2025]:
    subset = prices.loc[str(year)]
    print(f"{year}: mean={subset['price_eur_mwh'].mean():.2f}, "
          f"negative hours={(subset['price_eur_mwh'] < 0).sum()}, "
          f"rows={len(subset)}")
print()

# ============================================================
# 2. REALISED GENERATION
# Source: SMARD.de
# ============================================================
gen = pd.read_csv('data_raw/02_realised_generation.csv',
                  sep=';', encoding='utf-8-sig')

gen['datetime'] = pd.to_datetime(gen.iloc[:, 0], format='%d.%m.%Y %H:%M')

col_map = {}
for c in gen.columns:
    if 'Biomasse' in c: col_map[c] = 'biomass'
    elif 'Wasserkraft' in c: col_map[c] = 'hydro'
    elif 'Wind Offshore' in c: col_map[c] = 'wind_offshore'
    elif 'Wind Onshore' in c: col_map[c] = 'wind_onshore'
    elif 'Photovoltaik' in c: col_map[c] = 'solar'
    elif 'Sonstige Erneuerbare' in c: col_map[c] = 'other_renewables'
    elif 'Kernenergie' in c: col_map[c] = 'nuclear'
    elif 'Braunkohle' in c: col_map[c] = 'lignite'
    elif 'Steinkohle' in c: col_map[c] = 'hard_coal'
    elif 'Erdgas' in c: col_map[c] = 'gas'
    elif 'Pumpspeicher' in c: col_map[c] = 'pumped_hydro'
    elif 'Sonstige Konventionelle' in c: col_map[c] = 'other_conventional'

gen = gen.rename(columns=col_map)

gen_cols = list(col_map.values())
for c in gen_cols:
    gen[c] = gen[c].replace('-', pd.NA)
    gen[c] = gen[c].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

gen = gen[['datetime'] + gen_cols].set_index('datetime')
gen = gen.loc['2023':'2025']

print("=== 2. GENERATION ===")
print(f"Rows: {len(gen)}, Columns: {gen.columns.tolist()}")
print(f"Range: {gen.index.min()} to {gen.index.max()}")
print(f"Missing values:\n{gen.isna().sum()}")
print()

# ============================================================
# 3. REALISED CONSUMPTION
# Source: SMARD.de
# ============================================================
cons = pd.read_csv('data_raw/03_realised_consumption.csv',
                   sep=';', encoding='utf-8-sig')

cons['datetime'] = pd.to_datetime(cons.iloc[:, 0], format='%d.%m.%Y %H:%M')

col_map_cons = {}
for c in cons.columns:
    if 'Residuallast' in c: col_map_cons[c] = 'residual_load'
    elif 'Netzlast' in c and 'inkl' in c: col_map_cons[c] = 'grid_load_incl_pumped'
    elif 'Netzlast' in c: col_map_cons[c] = 'grid_load'
    elif 'Pumpspeicher' in c: col_map_cons[c] = 'pumped_hydro_consumption'

cons = cons.rename(columns=col_map_cons)

cons_cols = ['grid_load', 'residual_load']
for c in cons_cols:
    cons[c] = cons[c].replace('-', pd.NA)
    cons[c] = cons[c].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

cons = cons[['datetime'] + cons_cols].set_index('datetime')
cons = cons.loc['2023':'2025']

print("=== 3. CONSUMPTION ===")
print(f"Rows: {len(cons)}")
print(f"Range: {cons.index.min()} to {cons.index.max()}")
print(f"Missing values:\n{cons.isna().sum()}")
print()

# ============================================================
# 4. TEMPERATURE
# Source: DWD CDC, station Potsdam (ID 03987)
# MESS_DATUM format: YYYYMMDDHH
# TT_TU: air temperature in °C; -999.0 = missing value sentinel
# ============================================================
temp = pd.read_csv('data_raw/04_temperature_potsdam.txt',
                   sep=';', encoding='utf-8-sig')

temp['datetime'] = pd.to_datetime(temp['MESS_DATUM'].astype(str).str.strip(), format='%Y%m%d%H')
temp['temperature_c'] = temp['TT_TU'].astype(float)
temp['temperature_c'] = temp['temperature_c'].replace(-999.0, pd.NA)  # DWD missing value sentinel

temp = temp[['datetime', 'temperature_c']].set_index('datetime')
temp = temp.loc['2023':'2025']

print("Missing temperature hours:")
print(temp[temp['temperature_c'].isna()])

print("=== 4. TEMPERATURE ===")
print(f"Rows: {len(temp)}")
print(f"Range: {temp.index.min()} to {temp.index.max()}")
print(f"Missing values: {temp['temperature_c'].isna().sum()}")
print(f"Mean: {temp['temperature_c'].mean():.1f}°C")
print(f"Min: {temp['temperature_c'].min():.1f}°C, Max: {temp['temperature_c'].max():.1f}°C")
print(f"Coldest hour: {temp['temperature_c'].idxmin()} ({temp['temperature_c'].min():.1f}°C)")
print(f"Hottest hour: {temp['temperature_c'].idxmax()} ({temp['temperature_c'].max():.1f}°C)")
print()

# ============================================================
# 5. BDEW H0 STANDARD LOAD PROFILE
# Source: BDEW H0 standard load profile (March 2025, 2018-2023 measurements)
# Sheet: H25; 15-minute resolution; aggregated to hourly
# Columns: SA (Saturday), FT (Sunday/holiday), WT (weekday) x 12 months
# ============================================================
bdew = pd.read_excel('data_raw/05_bdew_standard_load_profiles.xlsx',
                     sheet_name='H25', header=None, skiprows=4)

time_slots = bdew.iloc[:, 1]

rows = []
months = list(range(1, 13))
day_types = ['SA', 'FT', 'WT']

for m_idx, month in enumerate(months):
    for d_idx, day_type in enumerate(day_types):
        col = 2 + m_idx * 3 + d_idx
        for row_idx in range(len(time_slots)):
            slot = time_slots.iloc[row_idx]
            if pd.isna(slot):
                continue
            val = bdew.iloc[row_idx, col]
            if pd.notna(val):
                rows.append({
                    'month': month,
                    'day_type': day_type,
                    'time_slot': slot,
                    'load_kwh': float(val)
                })

H25 = pd.DataFrame(rows)
H25['hour'] = H25['time_slot'].str.split('-').str[0].str.split(':').str[0].astype(int)

# Aggregate 15-min slots to hourly (sum four 15-min values per hour)
H25_hourly = H25.groupby(['month', 'day_type', 'hour'])['load_kwh'].sum().reset_index()

print("=== 5. BDEW H0 PROFILE ===")
print(f"Rows: {len(H25_hourly)} (expected 864: 12 months x 3 day types x 24 hours)")
print(f"Day types: {H25_hourly['day_type'].unique().tolist()}")
print(f"Sample (January weekday):")
print(H25_hourly[(H25_hourly['month'] == 1) & (H25_hourly['day_type'] == 'WT')].head())
print()

# ============================================================
# 6. MERGE INTO MASTER DATASET
# ============================================================
master = prices.join(gen, how='outer')
master = master.join(cons, how='outer')
master = master.join(temp, how='outer')

master['temperature_c'] = pd.to_numeric(master['temperature_c'], errors='coerce')

# Remove fall-back DST duplicate hours — keep first occurrence
# (October: clock goes back, one hour appears twice in local time)
master = master[~master.index.duplicated(keep='first')]

# Interpolate missing temperature values (linear, fills gaps from DWD missing sentinels)
master['temperature_c'] = master['temperature_c'].interpolate(method='linear')

# Add time columns
master['year'] = master.index.year
master['month'] = master.index.month
master['hour'] = master.index.hour
master['dayofweek'] = master.index.dayofweek  # 0=Monday, 6=Sunday

print("=== 6. MASTER DATASET (pre-save) ===")
print(f"Rows: {len(master)}")
print(f"Columns: {master.columns.tolist()}")
print(f"Range: {master.index.min()} to {master.index.max()}")
print(f"Missing values:\n{master.isna().sum()}")
print()

# ============================================================
# 7. SAVE
# ============================================================
# Drop spring-forward DST hours — 3 hours per year do not exist in CET/CEST
# local time; SMARD has no price entry for these hours → price_eur_mwh is NaN
master = master.dropna(subset=['price_eur_mwh'])

master.to_csv('data_clean/master_hourly.csv')
H25_hourly.to_csv('data_clean/H25_hourly_profile.csv', index=False)

print("Saved: data_clean/master_hourly.csv")
print("Saved: data_clean/H25_hourly_profile.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n=== FINAL SUMMARY ===")
print(f"Total rows: {len(master)} (26,304 hours minus 3 spring-forward DST hours = 26,301)")
for year in [2023, 2024, 2025]:
    subset = master.loc[master['year'] == year]
    print(f"{year}: {len(subset)} hours, "
          f"mean price={subset['price_eur_mwh'].mean():.1f} €/MWh, "
          f"negative hours={(subset['price_eur_mwh'] < 0).sum()}, "
          f"mean temp={subset['temperature_c'].mean():.1f}°C")

print("\nDone.")