"""
Exploratory Analysis — Pipeline 4
Produces descriptive statistics and charts for dissertation Chapter 4 (EDA).

Outputs:
  data_outputs/chart_01_price_distribution.png
    — Day-ahead price distribution by year (histogram, 3 panels)
  data_outputs/chart_02_daily_price_profile.png
    — Average hourly price profile by year
  data_outputs/chart_03_negative_hours_by_month.png
    — Negative price hours by month and year
  data_outputs/chart_04_renewables_vs_price.png
    — Renewable generation vs day-ahead price (hexbin)
  data_outputs/chart_05_residual_load_vs_price.png
    — Residual load vs day-ahead price (hexbin)
  data_outputs/chart_06_temperature_distribution.png
    — Temperature distribution with 15°C heating threshold (VDI 3807 Blatt 1, 2013)
  data_outputs/chart_07_monthly_averages.png
    — Monthly averages: price, wind, solar, temperature
  data_outputs/chart_08_H25_load_profile.png
    — BDEW H0 standard load profile by day type and season

Inputs:
  data_clean/master_hourly.csv
  data_clean/H25_hourly_profile.csv

Sources:
  SMARD.de — EPEX Spot Day-Ahead prices, realised generation, realised consumption
  DWD CDC — Hourly temperature, station Potsdam (ID 03987)
  BDEW H0 standard load profile (March 2025)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('data_outputs', exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
master = pd.read_csv('data_clean/master_hourly.csv', index_col='datetime', parse_dates=True)
print(f"Loaded: {len(master)} rows")
print(f"Columns: {master.columns.tolist()}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

# Total demand by year
print("\n=== DEMAND BY YEAR ===")
for year in [2023, 2024, 2025]:
    subset = master.loc[master['year'] == year, 'grid_load']
    print(f"{year}: total demand = {subset.sum()/1e6:.1f} TWh")

# Renewable share by year
# Note: other_renewables may include small amounts of non-renewable generation;
# treated as renewable here following SMARD categorisation
print("\n=== RENEWABLE SHARE BY YEAR ===")
for year in [2023, 2024, 2025]:
    subset = master.loc[master['year'] == year]
    total_gen = subset[['biomass', 'hydro', 'wind_offshore', 'wind_onshore', 'solar',
                        'other_renewables', 'nuclear', 'lignite', 'hard_coal', 'gas',
                        'pumped_hydro', 'other_conventional']].sum().sum()
    renewable_gen = subset[['biomass', 'hydro', 'wind_offshore', 'wind_onshore', 'solar',
                            'other_renewables']].sum().sum()
    print(f"{year}: renewable share = {renewable_gen/total_gen*100:.1f}% "
          f"({renewable_gen/1e6:.1f} TWh)")

# Hourly renewable share
master['renewables_total'] = (master['wind_onshore'] + master['wind_offshore']
                              + master['solar'] + master['biomass']
                              + master['hydro'].fillna(0))

master['renewable_share'] = (master['renewables_total'] /
    (master['renewables_total'] +
     master[['nuclear', 'lignite', 'hard_coal', 'gas',
              'pumped_hydro', 'other_conventional']].sum(axis=1)) * 100)

print("\n=== HOURLY RENEWABLE SHARE ===")
for year in [2023, 2024, 2025]:
    subset = master.loc[master['year'] == year, 'renewable_share']
    print(f"{year}: mean={subset.mean():.1f}%, "
          f"hours >80%: {(subset > 80).sum()}, "
          f"hours >90%: {(subset > 90).sum()}, "
          f"hours >100%: {(subset > 100).sum()}")

# ============================================================
# CHART 1 — PRICE DISTRIBUTION BY YEAR
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, year in enumerate([2023, 2024, 2025]):
    subset = master.loc[master['year'] == year, 'price_eur_mwh']
    axes[i].hist(subset, bins=100, color='steelblue', edgecolor='none')
    axes[i].axvline(x=0, color='red', linestyle='--', linewidth=0.8)
    axes[i].set_title(f'{year}')
    axes[i].set_xlabel('€/MWh')
    neg = (subset < 0).sum()
    mean = subset.mean()
    axes[i].text(0.95, 0.95, f'Mean: {mean:.0f}\nNeg hrs: {neg}',
                 transform=axes[i].transAxes, ha='right', va='top', fontsize=9)

axes[0].set_ylabel('Hours')
fig.suptitle('Day-Ahead Price Distribution by Year', fontsize=14)
fig.text(0.5, 0.01, 'Source: SMARD.de (EPEX Spot Day-Ahead DE/LU)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_01_price_distribution.png', dpi=150)
plt.close()
print("Chart 1 saved")

print(f"\nHours exactly at 0: {(master['price_eur_mwh'] == 0).sum()}")
print(f"Hours between -5 and 5: {((master['price_eur_mwh'] > -5) & (master['price_eur_mwh'] < 5)).sum()}")

# ============================================================
# CHART 2 — AVERAGE DAILY PRICE PROFILE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

for year in [2023, 2024, 2025]:
    subset = master.loc[master['year'] == year]
    hourly_avg = subset.groupby('hour')['price_eur_mwh'].mean()
    ax.plot(hourly_avg.index, hourly_avg.values, label=str(year), linewidth=2)

ax.set_xlabel('Hour of day')
ax.set_ylabel('Average price (€/MWh)')
ax.set_title('Average Daily Price Profile by Year')
ax.legend()
ax.set_xticks(range(0, 24))
fig.text(0.5, 0.01, 'Source: SMARD.de (EPEX Spot Day-Ahead DE/LU)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_02_daily_price_profile.png', dpi=150)
plt.close()
print("Chart 2 saved")

# ============================================================
# CHART 3 — NEGATIVE PRICE HOURS BY MONTH
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

neg_by_month = master[master['price_eur_mwh'] < 0].groupby(['year', 'month']).size().unstack(level=0)
neg_by_month.plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Month')
ax.set_ylabel('Negative price hours')
ax.set_title('Negative Price Hours by Month and Year')
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
ax.legend(title='Year')
fig.text(0.5, 0.01, 'Source: SMARD.de (EPEX Spot Day-Ahead DE/LU)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_03_negative_hours_by_month.png', dpi=150)
plt.close()
print("Chart 3 saved")

# ============================================================
# CHART 4 — RENEWABLE GENERATION VS PRICE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.hexbin(master['renewables_total'].dropna() / 1000,
          master.loc[master['renewables_total'].notna(), 'price_eur_mwh'],
          gridsize=80, cmap='YlOrRd', mincnt=1)
ax.set_xlabel('Total renewable generation (GWh)')
ax.set_ylabel('Price (€/MWh)')
ax.set_title('Renewable Generation vs Day-Ahead Price')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylim(-100, 300)
plt.colorbar(ax.collections[0], ax=ax, label='Hours')
fig.text(0.5, 0.01, 'Source: SMARD.de (EPEX Spot, realised generation)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_04_renewables_vs_price.png', dpi=150)
plt.close()
print("Chart 4 saved")

# ============================================================
# CHART 5 — RESIDUAL LOAD VS PRICE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.hexbin(master['residual_load'].dropna() / 1000,
          master.loc[master['residual_load'].notna(), 'price_eur_mwh'],
          gridsize=80, cmap='YlOrRd', mincnt=1)
ax.set_xlabel('Residual load (GWh)')
ax.set_ylabel('Price (€/MWh)')
ax.set_title('Residual Load vs Day-Ahead Price')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylim(-100, 300)
plt.colorbar(ax.collections[0], ax=ax, label='Hours')
fig.text(0.5, 0.01, 'Source: SMARD.de (EPEX Spot, realised consumption)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_05_residual_load_vs_price.png', dpi=150)
plt.close()
print("Chart 5 saved")

# ============================================================
# CHART 6 — TEMPERATURE DISTRIBUTION
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(master['temperature_c'], bins=80, color='steelblue', edgecolor='none')
ax.axvline(x=15, color='red', linestyle='--', linewidth=0.8,
           label='Heating threshold 15°C (VDI 3807 Blatt 1, 2013)')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Hours')
ax.set_title('Temperature Distribution (Potsdam, 2023-2025)')
ax.legend()
fig.text(0.5, 0.01, 'Source: DWD CDC, station Potsdam (ID 03987)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_06_temperature_distribution.png', dpi=150)
plt.close()

heating_hours = (master['temperature_c'] < 15).sum()
print(f"\nHours below 15°C: {heating_hours} ({heating_hours/len(master)*100:.0f}%)")
print("Chart 6 saved")

# ============================================================
# CHART 7 — MONTHLY AVERAGES
# ============================================================
master['wind_total'] = master['wind_onshore'] + master['wind_offshore']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

monthly_price = master.groupby(['year', 'month'])['price_eur_mwh'].mean().unstack(level=0)
monthly_price.plot(ax=axes[0, 0], marker='o')
axes[0, 0].set_title('Average Price by Month')
axes[0, 0].set_ylabel('€/MWh')
axes[0, 0].set_xticks(range(1, 13))

monthly_wind = master.groupby(['year', 'month'])['wind_total'].mean().unstack(level=0)
monthly_wind.plot(ax=axes[0, 1], marker='o')
axes[0, 1].set_title('Average Wind Generation by Month')
axes[0, 1].set_ylabel('MWh')
axes[0, 1].set_xticks(range(1, 13))

monthly_solar = master.groupby(['year', 'month'])['solar'].mean().unstack(level=0)
monthly_solar.plot(ax=axes[1, 0], marker='o')
axes[1, 0].set_title('Average Solar Generation by Month')
axes[1, 0].set_ylabel('MWh')
axes[1, 0].set_xticks(range(1, 13))

monthly_temp = master.groupby(['year', 'month'])['temperature_c'].mean().unstack(level=0)
monthly_temp.plot(ax=axes[1, 1], marker='o')
axes[1, 1].set_title('Average Temperature by Month')
axes[1, 1].set_ylabel('°C')
axes[1, 1].set_xticks(range(1, 13))

fig.suptitle('Monthly Patterns (2023-2025)', fontsize=14)
fig.text(0.5, 0.01, 'Source: SMARD.de (prices, generation); DWD CDC (temperature)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_07_monthly_averages.png', dpi=150)
plt.close()
print("Chart 7 saved")

# ============================================================
# CHART 8 — BDEW H0 LOAD PROFILE SHAPES
# ============================================================
H25 = pd.read_csv('data_clean/H25_hourly_profile.csv')

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

day_labels = {'WT': 'Weekday', 'SA': 'Saturday', 'FT': 'Sunday/Holiday'}
season_colors = {'Jan': '#000000', 'Apr': '#FFD700', 'Jul': '#cc0000', 'Oct': '#888888'}

for i, (day_type, label) in enumerate(day_labels.items()):
    for month in [1, 4, 7, 10]:
        subset = H25[(H25['day_type'] == day_type) & (H25['month'] == month)]
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
        axes[i].plot(subset['hour'], subset['load_kwh'],
                     label=month_name, linewidth=2, color=season_colors[month_name])
    axes[i].set_title(label)
    axes[i].set_xlabel('Hour of day')
    axes[i].set_xticks(range(0, 24, 3))
    axes[i].legend()

axes[0].set_ylabel('Load (kWh, normalised to 1M kWh/yr)')
fig.suptitle('BDEW H0 Standard Household Load Profile', fontsize=14)
fig.text(0.5, 0.01, 'Source: BDEW H0 standard load profile (March 2025)',
         ha='center', fontsize=8, color='grey')
plt.tight_layout()
plt.savefig('data_outputs/chart_08_H25_load_profile.png', dpi=150)
plt.close()
print("Chart 8 saved")

print("\nDone.")