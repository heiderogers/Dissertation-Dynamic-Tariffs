"""
Exploratory Analysis — Pipeline 1b
Produces descriptive statistics and charts

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
  data_outputs/chart_08a_H25_load_profile.png
    — BDEW H25 standard load profile by day type and season
  data_outputs/chart_08b_H25_dynamisierung.png
    — BDEW H25 load profile with Dynamisierung applied, representative days by season
  data_outputs/chart_09_monthly_patterns.png
    — Monthly patterns: price, negative hours, solar, wind (2 x 2 panel)

Inputs:
  data_clean/master_hourly.csv
  data_clean/H25_hourly_profile.csv

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('data_outputs', exist_ok=True)

def get_day_type(dow):
    if dow == 5:
        return 'SA'
    elif dow == 6:
        return 'FT'
    else:
        return 'WT'

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

print("\n=== SOLAR VS WIND TOTAL GENERATION ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    solar_total = subset['solar'].sum() / 1e6
    wind_total = (subset['wind_onshore'] + subset['wind_offshore']).sum() / 1e6
    print(f"  {year}: solar={solar_total:.1f} TWh, wind={wind_total:.1f} TWh, ratio wind/solar={wind_total/solar_total:.2f}x")

print("\n=== SOLAR VS WIND HOURLY VOLATILITY ===")
print(f"  Solar: mean={master['solar'].mean()/1000:.1f} GWh/hr, std={master['solar'].std()/1000:.1f} GWh/hr, max={master['solar'].max()/1000:.1f} GWh/hr")
print(f"  Wind:  mean={(master['wind_onshore']+master['wind_offshore']).mean()/1000:.1f} GWh/hr, std={(master['wind_onshore']+master['wind_offshore']).std()/1000:.1f} GWh/hr, max={(master['wind_onshore']+master['wind_offshore']).max()/1000:.1f} GWh/hr")
print(f"  Solar coefficient of variation: {master['solar'].std()/master['solar'].mean()*100:.0f}%")

print("\n=== SOLAR VS WIND VOLATILITY — THREE MEASURES ===")

# 1. Overall CV across all hourly observations (current measure)
print("  1. Overall CV (all 26,301 hourly observations):")
print(f"     Solar: {master['solar'].std()/master['solar'].mean()*100:.0f}%")
wind = master['wind_onshore'] + master['wind_offshore']
print(f"     Wind:  {wind.std()/wind.mean()*100:.0f}%")

# 2. CV of monthly averages — captures seasonal variation
print("  2. CV of monthly averages (seasonal variation):")
solar_monthly = master.groupby(['year','month'])['solar'].mean()
wind_monthly = master.groupby(['year','month'])[['wind_onshore','wind_offshore']].sum().sum(axis=1)
print(f"     Solar: {solar_monthly.std()/solar_monthly.mean()*100:.0f}%")
print(f"     Wind:  {wind_monthly.std()/wind_monthly.mean()*100:.0f}%")

# 3. CV of average hourly profile — captures intraday variation
print("  3. CV of average hourly profile (intraday variation):")
solar_hourly = master.groupby('hour')['solar'].mean()
wind_hourly = master.groupby('hour')[['wind_onshore','wind_offshore']].sum().sum(axis=1)
print(f"     Solar: {solar_hourly.std()/solar_hourly.mean()*100:.0f}%")
print(f"     Wind:  {wind_hourly.std()/wind_hourly.mean()*100:.0f}%")

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
    axes[i].text(0.95, 0.95,
                 f'Mean: {mean:.0f}\nSD: {subset.std():.0f}\nNeg hrs: {neg}\nRenewables: {master.loc[master["year"] == year, "renewable_share"].mean():.0f}%\nSolar: {master.loc[master["year"] == year, "solar"].sum() / 1e6:.1f} TWh\nWind: {(master.loc[master["year"] == year, "wind_onshore"] + master.loc[master["year"] == year, "wind_offshore"]).sum() / 1e6:.1f} TWh',
                 transform=axes[i].transAxes, ha='right', va='top', fontsize=9)

axes[0].set_ylabel('Hours')
fig.suptitle('Day-Ahead Price Distribution by Year', fontsize=14)
plt.tight_layout()
plt.savefig('data_outputs/chart_01_price_distribution.png', dpi=150)
plt.close()
print("Chart 1 saved")

print(f"\nHours exactly at 0: {(master['price_eur_mwh'] == 0).sum()}")
print(f"Hours between -5 and 5: {((master['price_eur_mwh'] > -5) & (master['price_eur_mwh'] < 5)).sum()}")

# Chart 1 — Price distribution
print("\n=== CHART 1 KEY NUMBERS ===")
for year in [2023, 2024, 2025]:
    subset = master.loc[master['year'] == year, 'price_eur_mwh']
    print(f"  {year}: mean=€{subset.mean():.1f}/MWh, negative hours={(subset < 0).sum()}, "
          f"min=€{subset.min():.1f}, max=€{subset.max():.1f}")

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
plt.tight_layout()
plt.savefig('data_outputs/chart_02_daily_price_profile.png', dpi=150)
plt.close()
print("Chart 2 saved")

# Chart 2 — Daily price profile
print("\n=== CHART 2 KEY NUMBERS ===")
hourly_avg_all = master.groupby('hour')['price_eur_mwh'].mean()
cheapest_hour = hourly_avg_all.idxmin()
peak_hour = hourly_avg_all.idxmax()
print(f"  Cheapest hour: {cheapest_hour}:00 — €{hourly_avg_all[cheapest_hour]:.1f}/MWh")
print(f"  Most expensive hour: {peak_hour}:00 — €{hourly_avg_all[peak_hour]:.1f}/MWh")
print(f"  Intraday spread: €{hourly_avg_all[peak_hour] - hourly_avg_all[cheapest_hour]:.1f}/MWh")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    ha = subset.groupby('hour')['price_eur_mwh'].mean()
    print(f"  {year}: cheapest={ha.idxmin()}:00 €{ha.min():.1f}, peak={ha.idxmax()}:00 €{ha.max():.1f}, spread=€{ha.max()-ha.min():.1f}")

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

# Chart 3 — Negative hours by month
print("\n=== CHART 3 KEY NUMBERS ===")
neg_by_month_year = master[master['price_eur_mwh'] < 0].groupby(['year', 'month']).size()
for year in [2023, 2024, 2025]:
    peak_month = neg_by_month_year[year].idxmax()
    print(f"  {year}: peak month={peak_month}, hours={neg_by_month_year[year].max()}, "
          f"total negative={neg_by_month_year[year].sum()}")

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

# Chart 4 — Renewables vs price
print("\n=== CHART 4 KEY NUMBERS ===")
print(f"  Mean renewable generation: {master['renewables_total'].mean()/1000:.1f} GWh/hr")
print(f"  Hours with renewables >50 GWh/hr: {(master['renewables_total'] > 50000).sum()}")
print(f"  Mean price when renewables >50 GWh/hr: €{master.loc[master['renewables_total'] > 50000, 'price_eur_mwh'].mean():.1f}/MWh")
print(f"  Mean price when renewables <20 GWh/hr: €{master.loc[master['renewables_total'] < 20000, 'price_eur_mwh'].mean():.1f}/MWh")

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

# Chart 5 — Residual load vs price
print("\n=== CHART 5 KEY NUMBERS ===")
print(f"  Mean residual load: {master['residual_load'].mean()/1000:.1f} GWh/hr")
print(f"  Mean price when residual load >40 GWh/hr: €{master.loc[master['residual_load'] > 40000, 'price_eur_mwh'].mean():.1f}/MWh")
print(f"  Mean price when residual load <10 GWh/hr: €{master.loc[master['residual_load'] < 10000, 'price_eur_mwh'].mean():.1f}/MWh")
print(f"  Mean price when residual load <0 GWh/hr: €{master.loc[master['residual_load'] < 0, 'price_eur_mwh'].mean():.1f}/MWh")

# ============================================================
# CHART 6 — TEMPERATURE DISTRIBUTION
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(master['temperature_c'], bins=80, color='steelblue', edgecolor='none')
# 15°C heating threshold (Heizgrenztemperatur) — standard German threshold above which space heating is assumed unnecessary; per VDI 3807 Blatt 1 (2013)
ax.axvline(x=15, color='red', linestyle='--', linewidth=0.8,
           label='Heating threshold 15°C (VDI 3807 Blatt 1, 2013)')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Hours')
ax.set_title('Temperature Distribution (Potsdam, 2023-2025)')
ax.legend()
plt.tight_layout()
plt.savefig('data_outputs/chart_06_temperature_distribution.png', dpi=150)
plt.close()

print("Chart 6 saved")

# Chart 6 — Temperature distribution
print("\n=== CHART 6 KEY NUMBERS ===")
print(f"  Hours below 15°C: {(master['temperature_c'] < 15).sum()} ({(master['temperature_c'] < 15).mean()*100:.0f}%)")
print(f"  Mean temperature: {master['temperature_c'].mean():.1f}°C")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    print(f"  {year}: mean={subset['temperature_c'].mean():.1f}°C, hours below 15°C={(subset['temperature_c'] < 15).sum()}")

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
plt.tight_layout()
plt.savefig('data_outputs/chart_07_monthly_averages.png', dpi=150)
plt.close()
print("Chart 7 saved")

# Chart 7 — Monthly averages
print("\n=== CHART 7 KEY NUMBERS ===")
for year in [2023, 2024, 2025]:
    subset = master[master['year'] == year]
    monthly_price = subset.groupby('month')['price_eur_mwh'].mean()
    print(f"  {year}: cheapest month={monthly_price.idxmin()} €{monthly_price.min():.1f}, "
          f"most expensive month={monthly_price.idxmax()} €{monthly_price.max():.1f}")

# ============================================================
# CHART 8a — BDEW H25 LOAD PROFILE SHAPES
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
fig.suptitle('BDEW H25 Standard Household Load Profile', fontsize=14)
plt.tight_layout()
plt.savefig('data_outputs/chart_08a_H25_load_profile.png', dpi=150)
plt.close()
print("Chart 8a saved")

# Chart 8a — H25 load profile
print("\n=== CHART 8a KEY NUMBERS ===")
for day_type in ['WT', 'SA', 'FT']:
    subset = H25[H25['day_type'] == day_type]
    peak_hour = subset.groupby('hour')['load_kwh'].mean().idxmax()
    peak_val = subset.groupby('hour')['load_kwh'].mean().max()
    print(f"  {day_type}: peak hour={peak_hour}:00, peak load={peak_val:.1f} kWh")

# ============================================================
# CHART 8b — H25 load profile with Dynamisierung applied
# 1x3 panel: Weekday (Wednesday), Saturday, Sunday/Holiday
# Representative mid-month days in Jan, Apr, Jul, Oct 2025
# Dynamisierung polynomial applied per BDEW H25 documentation (March 2025)
# ============================================================
master_eda = master.copy()
H25 = pd.read_csv('data_clean/H25_hourly_profile.csv')

master_eda['day_type'] = master_eda['dayofweek'].apply(get_day_type)
master_eda['day_of_year'] = master_eda.index.dayofyear.astype(float)
master_eda['dynamisierung'] = (-3.92e-10 * master_eda['day_of_year']**4
                               + 3.20e-7  * master_eda['day_of_year']**3
                               - 7.02e-5  * master_eda['day_of_year']**2
                               + 2.10e-3  * master_eda['day_of_year']
                               + 1.24)

master_eda = master_eda.reset_index()
master_eda = master_eda.merge(H25, on=['month', 'day_type', 'hour'], how='left')
master_eda = master_eda.rename(columns={'load_kwh': 'h25_base'})
master_eda['h25_dynamic'] = master_eda['h25_base'] * master_eda['dynamisierung']

# Representative mid-month days: Wednesday, Saturday, Sunday
rep_days = {
    'Weekday (Wednesday)': {
        'Jan': '2025-01-15',
        'Apr': '2025-04-16',
        'Jul': '2025-07-16',
        'Oct': '2025-10-15',
    },
    'Saturday': {
        'Jan': '2025-01-18',
        'Apr': '2025-04-12',
        'Jul': '2025-07-12',
        'Oct': '2025-10-18',
    },
    'Sunday': {
        'Jan': '2025-01-12',
        'Apr': '2025-04-13',
        'Jul': '2025-07-13',
        'Oct': '2025-10-12',
    },
}

season_colors = {
    'Jan': '#000000',
    'Apr': '#FFD700',
    'Jul': '#cc0000',
    'Oct': '#888888',
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, (panel_label, dates) in enumerate(rep_days.items()):
    ax = axes[i]
    for month_name, date_str in dates.items():
        day_data = master_eda[master_eda['datetime'].dt.strftime('%Y-%m-%d') == date_str]
        ax.plot(day_data['hour'], day_data['h25_dynamic'],
                label=f'{month_name} {date_str[8:]}',
                linewidth=2, color=season_colors[month_name])
    ax.set_title(panel_label)
    ax.set_xlabel('Hour of day')
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

axes[0].set_ylabel('Load (kWh, normalised to 1M kWh/yr)')
fig.suptitle('BDEW H25 Load Profile with Dynamisierung — Representative Days by Season', fontsize=14)
plt.tight_layout()
plt.savefig('data_outputs/chart_08b_H25_dynamisierung.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 8b saved")

# Chart 8b — H25 with Dynamisierung
print("\n=== CHART 8b KEY NUMBERS ===")
print(f"  Dynamisierung factor range: min={master_eda['dynamisierung'].min():.3f} (day {master_eda.loc[master_eda['dynamisierung'].idxmin(), 'day_of_year']:.0f}), max={master_eda['dynamisierung'].max():.3f} (day {master_eda.loc[master_eda['dynamisierung'].idxmax(), 'day_of_year']:.0f})")
for day_type, label in [('WT', 'Weekday'), ('SA', 'Saturday'), ('FT', 'Sunday')]:
    subset = master_eda[master_eda['day_type'] == day_type]
    peak_hour = subset.groupby('hour')['h25_dynamic'].mean().idxmax()
    peak_val = subset.groupby('hour')['h25_dynamic'].mean().max()
    print(f"  {label}: peak hour={peak_hour}:00, peak load={peak_val:.1f} kWh (Dynamisierung-adjusted)")

# ============================================================
# CHART 9 — MONTHLY PATTERNS: PRICE, NEGATIVE HOURS, SOLAR, WIND
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

monthly_price = master.groupby(['year', 'month'])['price_eur_mwh'].mean().unstack(level=0)
monthly_price.plot(ax=axes[0, 0], marker='o')
axes[0, 0].set_title('Average Price by Month')
axes[0, 0].set_ylabel('€/MWh')
axes[0, 0].set_xticks(range(1, 13))

neg_by_month = master[master['price_eur_mwh'] < 0].groupby(['year', 'month']).size().unstack(level=0)
neg_by_month.plot(ax=axes[0, 1], marker='o')
axes[0, 1].set_title('Negative Price Hours by Month')
axes[0, 1].set_ylabel('Hours')
axes[0, 1].set_xticks(range(1, 13))

monthly_solar = master.groupby(['year', 'month'])['solar'].mean().unstack(level=0)
monthly_solar.plot(ax=axes[1, 0], marker='o')
axes[1, 0].set_title('Average Solar Generation by Month')
axes[1, 0].set_ylabel('MWh')
axes[1, 0].set_xticks(range(1, 13))

monthly_wind = master.groupby(['year', 'month'])['wind_total'].mean().unstack(level=0)
monthly_wind.plot(ax=axes[1, 1], marker='o')
axes[1, 1].set_title('Average Wind Generation by Month')
axes[1, 1].set_ylabel('MWh')
axes[1, 1].set_xticks(range(1, 13))

fig.suptitle('Monthly Patterns: Price, Negative Hours, Solar and Wind (2023-2025)', fontsize=14)
plt.tight_layout()
plt.savefig('data_outputs/chart_09_monthly_patterns.png', dpi=150)
plt.close()
print("Chart 9 saved")

# Chart 9 — Monthly patterns
print("\n=== CHART 9 KEY NUMBERS ===")
print("  Price:")
for year in [2023, 2024, 2025]:
    mp = master[master['year'] == year].groupby('month')['price_eur_mwh'].mean()
    print(f"    {year}: peak month {mp.idxmax()} at €{mp.max():.1f}/MWh, trough month {mp.idxmin()} at €{mp.min():.1f}/MWh")
print("  Negative hours:")
for year in [2023, 2024, 2025]:
    nh = master[(master['year'] == year) & (master['price_eur_mwh'] < 0)].groupby('month').size()
    print(f"    {year}: peak month {nh.idxmax()} with {nh.max()} negative hours")
print("  Solar:")
for year in [2023, 2024, 2025]:
    ms = master[master['year'] == year].groupby('month')['solar'].mean()
    print(f"    {year}: peak month {ms.idxmax()} at {ms.max()/1000:.1f} GWh/hr avg")
print("  Wind:")
for year in [2023, 2024, 2025]:
    mw = master[master['year'] == year].groupby('month')['wind_total'].mean()
    print(f"    {year}: peak month {mw.idxmax()} at {mw.max()/1000:.1f} GWh/hr avg")

print("\nDone.")