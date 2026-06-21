"""
Figure - Load Profiles for Methodology section
Produces a 2×2 panel figure comparing passive and automated load profiles
across all four archetypes, average day across 2025, with EPEX price overlay.

Figure (fig_load_profiles_methodology.png) — §3.2.3:
  Shows passive vs automated hourly load (kWh/h) for each archetype.
  EPEX day-ahead price (€/MWh) shown on secondary axis to illustrate
  that automated load shifts toward cheaper hours.
  Illustrates the load shifting logic: white goods (A), heat pump (B),
  EV (C), and combined (D).

Inputs:
  data_simulation/load_profile_A.csv
  data_simulation/load_profile_B.csv
  data_simulation/load_profile_C.csv
  data_simulation/load_profile_D.csv
  data_simulation/load_profiles_auto_t3.csv
  data_simulation/tariff_prices.csv

Outputs:
  data_outputs/figures/fig_load_profiles_methodology.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('data_outputs/figures', exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
load_A = pd.read_csv('data_simulation/load_profile_A.csv',
                     index_col='datetime', parse_dates=True)
load_B = pd.read_csv('data_simulation/load_profile_B.csv',
                     index_col='datetime', parse_dates=True)
load_C = pd.read_csv('data_simulation/load_profile_C.csv',
                     index_col='datetime', parse_dates=True)
load_D = pd.read_csv('data_simulation/load_profile_D.csv',
                     index_col='datetime', parse_dates=True)
auto_t3 = pd.read_csv('data_simulation/load_profiles_auto_t3.csv',
                      index_col='datetime', parse_dates=True)
tariffs = pd.read_csv('data_simulation/tariff_prices.csv',
                      index_col='datetime', parse_dates=True)

print("Loaded all inputs.")

# ============================================================
# BUILD AVERAGE HOURLY PROFILES — 2025, ALL DAYS
# ============================================================
year = 2025

def avg_hourly(series, year):
    """Average by hour across all days in the given year."""
    mask = series.index.year == year
    return series[mask].groupby(series[mask].index.hour).mean()

def smooth(series, window=3):
    """Circular 3-hour rolling average for presentation clarity."""
    s = pd.concat([series.iloc[-(window//2):], series, series.iloc[:(window//2)]])
    return s.rolling(window, center=True).mean().iloc[(window//2):-(window//2)].values

# Passive loads — unsmoothed; rectangular EV block (hours 18-21) is intentional
# and visually illustrates the passive charging assumption.
passive_A = avg_hourly(load_A['load_A'], year).values
passive_B = avg_hourly(load_B['load_B'], year).values
passive_C = avg_hourly(load_C['load_C'], year).values
passive_D = avg_hourly(load_D['load_D'], year).values

# Automated loads (T3 schedule = T2 schedule)
auto_A = smooth(avg_hourly(auto_t3['load_A_auto'], year))
auto_B = smooth(avg_hourly(auto_t3['load_B_auto'], year))
auto_C = smooth(avg_hourly(auto_t3['load_C_auto'], year))
auto_D = smooth(avg_hourly(auto_t3['load_D_auto'], year))

# EPEX day-ahead price — hourly ranking drives automated shifting logic
# Fixed grid fees and taxes are uniform additions that do not affect hourly ranking
# Raw EPEX in €/MWh is therefore the relevant price signal to show
price_epex = avg_hourly(tariffs['epex_ct_kwh'] * 10, year)

hours = np.arange(24)

# ============================================================
# STYLE CONSTANTS
# ============================================================
COLOUR_PASSIVE  = '#2c7bb6'   # blue — passive
COLOUR_AUTO     = '#d7191c'   # red — automated
COLOUR_PRICE    = '#636363'   # grey — EPEX price
ALPHA_PRICE     = 0.15

ARCHETYPE_LABELS = {
    'A': 'Archetype A: Base household (3,500 kWh/yr)',
    'B': 'Archetype B: Base + heat pump (7,500 kWh/yr)',
    'C': 'Archetype C: Base + EV (6,000 kWh/yr)',
    'D': 'Archetype D: Base + heat pump + EV (10,000 kWh/yr)',
}

archetypes = [
    ('A', passive_A, auto_A),
    ('B', passive_B, auto_B),
    ('C', passive_C, auto_C),
    ('D', passive_D, auto_D),
]

# ============================================================
# FIGURE — METHODOLOGY (§3.2.3)
# Passive vs automated load with EPEX price overlay on secondary axis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

for i, (label, passive, auto) in enumerate(archetypes):
    ax = axes[i]

    # EPEX price — secondary axis, grey shaded area
    ax2 = ax.twinx()
    ax2.fill_between(hours, price_epex.values,
                     color=COLOUR_PRICE, alpha=ALPHA_PRICE, label='EPEX price (€/MWh)')
    ax2.set_ylabel('EPEX price (€/MWh)', fontsize=7, color=COLOUR_PRICE)
    ax2.tick_params(axis='y', labelsize=7, labelcolor=COLOUR_PRICE)
    ax2.spines['top'].set_visible(False)

    # Load profiles — primary axis
    ax.plot(hours, passive, color=COLOUR_PASSIVE, linewidth=1.8,
            linestyle='-', label='Passive', zorder=3)
    ax.plot(hours, auto, color=COLOUR_AUTO, linewidth=1.8,
            linestyle='--', label='Automated', zorder=3)
    ax.set_title(ARCHETYPE_LABELS[label], fontsize=9, pad=6)
    ax.set_ylabel('Load (kWh/h)', fontsize=8)
    ax.set_xlabel('Hour of day', fontsize=8)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    if i == 0:
        ax.legend(fontsize=8, frameon=False, loc='upper left')

fig.suptitle(
    'Average hourly load profiles — passive vs automated scenario, 2025\n'
    'Grey shading: average EPEX day-ahead price (€/MWh); average across all days',
    fontsize=10, y=1.01
)
fig.tight_layout()
fig.savefig('data_outputs/figures/fig_load_profiles_methodology.png', dpi=300, bbox_inches='tight')
print("Saved: figures/fig_load_profiles_methodology.png")
plt.close(fig)

print("Done.")