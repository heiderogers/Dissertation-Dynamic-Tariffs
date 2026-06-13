"""
Breakeven — Pipeline 3b
Computes C* (break-even switching costs) per archetype x tariff x scenario x year
and saves to data_simulation/breakeven.csv

C* definition:
  C* = T1_passive_bill - dynamic_scenario_bill
  Positive C*: switching to the dynamic tariff saves money relative to the flat
    tariff baseline; the household has headroom to cover switching costs up to C*
    and still break even
  Negative C*: switching costs more than staying on the flat tariff; switching is
    irrational regardless of switching cost level

Baseline:
  T1 passive (flat tariff, no load shifting) is the counterfactual for all C*
  calculations — it represents what the household pays today doing nothing.
  T1 automated = T1 passive by construction (flat price provides no shifting
  incentive), so C* for T1 automated = 0 in all cases.

Risk metrics:
  Computed both per year (2023, 2024, 2025) and pooled across all three years.
  worst_month_eur: maximum single-month bill premium over T1 passive
  loss_month_prob: share of months where dynamic bill exceeds T1 passive
  Per-year metrics allow year-specific characterisation (2023 high-price,
  2024 low-price, 2025 moderate); pooled captures full observed volatility range.
  2025 is the primary year; 2023/2024 bracket the range across price environments.

Output columns:
  archetype, tariff, scenario, year — combination identifiers
  annual_bill_eur — total annual bill for this combination
  baseline_bill_eur — T1 passive annual bill for same archetype and year
  c_star — annual break-even switching cost (€/yr); positive = switching rational
  c_star_avg — 3-year average C* (reference; not primary result)
  worst_month_eur_2023/2024/2025 — per-year maximum monthly loss vs T1 passive (€)
  loss_month_prob_2023/2024/2025 — per-year share of months dynamic > T1 passive
  worst_month_eur_pooled — maximum monthly loss pooled across 2023-2025
  loss_month_prob_pooled — share of months dynamic > T1 passive pooled across 2023-2025

Inputs:
  data_simulation/bills.csv

Outputs:
  data_simulation/breakeven.csv
"""

import pandas as pd
import numpy as np
import os

pd.set_option('display.float_format', '{:.2f}'.format)

os.makedirs('data_simulation', exist_ok=True)

# ============================================================
# LOAD BILLS
# ============================================================
bills = pd.read_csv('data_simulation/bills.csv')
print(f"Loaded bills: {len(bills)} rows")

# ============================================================
# AGGREGATE TO ANNUAL BILLS
# ============================================================
annual = (bills.groupby(['archetype', 'tariff', 'scenario', 'year'])['monthly_bill_eur']
               .sum()
               .reset_index()
               .rename(columns={'monthly_bill_eur': 'annual_bill_eur'}))

print(f"\nAnnual bills: {len(annual)} rows")
print(annual)

# ============================================================
# EXTRACT T1 PASSIVE BASELINE
# ============================================================
baseline = (annual[(annual['tariff'] == 'T1') & (annual['scenario'] == 'passive')]
                [['archetype', 'year', 'annual_bill_eur']]
                .rename(columns={'annual_bill_eur': 'baseline_bill_eur'})
                .reset_index(drop=True))

print(f"\nBaseline bills (T1 passive):")
print(baseline)

# ============================================================
# COMPUTE C*
# ============================================================
# C* = baseline (T1 passive) - dynamic scenario bill
# T1 automated = T1 passive by construction (flat price, no shifting incentive)
# so C* for T1 automated = 0 in all cases — expected and correct

annual = annual.merge(baseline, on=['archetype', 'year'])
annual['c_star'] = annual['baseline_bill_eur'] - annual['annual_bill_eur']

# Verify T1 automated C* = 0
t1_auto_check = annual[(annual['tariff'] == 'T1') & (annual['scenario'] == 'automated')]
assert (t1_auto_check['c_star'].abs() < 0.01).all(), "T1 automated C* should be 0"
print("\nT1 automated C* = 0 check: PASS")

print("\n=== C* BREAK-EVEN SWITCHING COSTS (€/yr) ===")
print("Positive = switching saves money | Negative = switching costs more than flat")
print(annual[['archetype', 'tariff', 'scenario', 'year', 'c_star']].to_string())

# ============================================================
# 3-YEAR AVERAGE C*
# ============================================================
# Reference number — not primary result given structural differences across years
# (2023 high-price, 2024 low-price, 2025 moderate)

c_star_avg = (annual.groupby(['archetype', 'tariff', 'scenario'])['c_star']
                    .mean()
                    .reset_index()
                    .rename(columns={'c_star': 'c_star_avg'}))

print("\n=== C* 3-YEAR AVERAGE (€/yr) — reference only ===")
print(c_star_avg.to_string())

# ============================================================
# RISK METRICS — PER YEAR AND POOLED
# ============================================================

# Extract monthly T1 passive baseline
monthly_baseline = (bills[(bills['tariff'] == 'T1') & (bills['scenario'] == 'passive')]
                        [['archetype', 'year', 'month', 'monthly_bill_eur']]
                        .rename(columns={'monthly_bill_eur': 'baseline_monthly_eur'}))

bills_risk = bills.merge(monthly_baseline, on=['archetype', 'year', 'month'])
bills_risk['monthly_diff'] = bills_risk['monthly_bill_eur'] - bills_risk['baseline_monthly_eur']

# --- Per-year risk metrics ---
risk_peryear = {}
for year in [2023, 2024, 2025]:
    year_data = bills_risk[bills_risk['year'] == year]
    risk_y = (year_data.groupby(['archetype', 'tariff', 'scenario'])
                       .apply(lambda x: pd.Series({
                           f'worst_month_eur_{year}': x['monthly_diff'].max(),
                           f'loss_month_prob_{year}': (x['monthly_diff'] > 0).mean()
                       }))
                       .reset_index())
    risk_peryear[year] = risk_y

# --- Pooled risk metrics ---
risk_pooled = (bills_risk.groupby(['archetype', 'tariff', 'scenario'])
                         .apply(lambda x: pd.Series({
                             'worst_month_eur_pooled': x['monthly_diff'].max(),
                             'loss_month_prob_pooled': (x['monthly_diff'] > 0).mean()
                         }))
                         .reset_index())

print("\n=== RISK METRICS PER YEAR ===")
print("worst_month_eur: max monthly loss vs T1 passive (€)")
print("loss_month_prob: share of months where dynamic > T1 passive")
for year in [2023, 2024, 2025]:
    print(f"\n--- {year} ---")
    print(risk_peryear[year].to_string())

print("\n=== RISK METRICS POOLED (2023-2025) ===")
print(risk_pooled.to_string())

# ============================================================
# SAVE BREAKEVEN.CSV
# ============================================================
breakeven = annual.merge(
    c_star_avg, on=['archetype', 'tariff', 'scenario']
)

# Merge per-year risk metrics
for year in [2023, 2024, 2025]:
    breakeven = breakeven.merge(
        risk_peryear[year], on=['archetype', 'tariff', 'scenario']
    )

# Merge pooled risk metrics
breakeven = breakeven.merge(
    risk_pooled, on=['archetype', 'tariff', 'scenario']
)

breakeven = breakeven[[
    'archetype', 'tariff', 'scenario', 'year',
    'annual_bill_eur', 'baseline_bill_eur', 'c_star', 'c_star_avg',
    'worst_month_eur_2023', 'worst_month_eur_2024', 'worst_month_eur_2025',
    'loss_month_prob_2023', 'loss_month_prob_2024', 'loss_month_prob_2025',
    'worst_month_eur_pooled', 'loss_month_prob_pooled'
]].sort_values(['archetype', 'tariff', 'scenario', 'year']).reset_index(drop=True)

breakeven.to_csv('data_simulation/breakeven.csv', index=False)
print(f"\nSaved: data_simulation/breakeven.csv ({len(breakeven)} rows)")
print(breakeven.head(10))

print("\nDone.")