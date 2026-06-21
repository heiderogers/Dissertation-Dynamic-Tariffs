"""
Breakeven — Pipeline 3a
Computes C* (break-even switching costs) per archetype × tariff × scenario × year
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

# ============================================================
# WORST MONTH IDENTIFICATION — T2 automated, 2025
# ============================================================
print("\n=== WORST MONTH: T2 automated, 2025 ===")
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

baseline_monthly = (bills[(bills['tariff']=='T1') & (bills['scenario']=='passive') & (bills['year']==2025)]
                        [['archetype','month','monthly_bill_eur']]
                        .rename(columns={'monthly_bill_eur':'baseline'}))

t2_auto_monthly = (bills[(bills['tariff']=='T2') & (bills['scenario']=='automated') & (bills['year']==2025)]
                       [['archetype','month','monthly_bill_eur']])

merged = t2_auto_monthly.merge(baseline_monthly, on=['archetype','month'])
merged['diff'] = merged['monthly_bill_eur'] - merged['baseline']

worst = merged.loc[merged.groupby('archetype')['diff'].idxmax()][['archetype','month','diff']]
worst['month_name'] = worst['month'].map(month_names)
print(worst[['archetype','month_name','diff']].to_string())

# ============================================================
# Dissertation tables
# ============================================================

print("\n" + "="*60)
print("DISSERTATION TABLES")
print("="*60)

TARIFF_ORDER = ['T1', 'T2', 'T3_TIB', 'T4']
ARCH_ORDER = ['A', 'B', 'C', 'D']
SCENARIOS = ['passive', 'automated']

# --- TABLE 1: Bills Matrix 2025 ---
print("\n--- TABLE 1: Annual Bills by Archetype and Tariff, 2025 (€/yr) ---")
bills_2025 = annual[annual['year'] == 2025].copy()
bills_2025['bill'] = bills_2025['annual_bill_eur'].round(0).astype(int)

rows = []
for arch in ARCH_ORDER:
    row = {'Archetype': arch}
    # T1 passive
    t1 = bills_2025[(bills_2025['archetype']==arch) & (bills_2025['tariff']=='T1') & (bills_2025['scenario']=='passive')]
    row['T1'] = int(t1['annual_bill_eur'].values[0]) if len(t1) else 'n/a'
    for tariff in ['T2', 'T3_TIB', 'T4']:
        for scenario in SCENARIOS:
            val = bills_2025[(bills_2025['archetype']==arch) & (bills_2025['tariff']==tariff) & (bills_2025['scenario']==scenario)]
            col = f"{tariff.replace('_TIB','')} {scenario[:4]}"
            row[col] = int(val['annual_bill_eur'].values[0]) if len(val) else 'n/a'
    rows.append(row)

bills_table = pd.DataFrame(rows)
print(bills_table.to_string(index=False))

# --- TABLE 2: C* 2025 ---
print("\n--- TABLE 2: Break-Even Switching Costs C*, 2025 (€/yr) ---")
cstar_2025 = annual[annual['year'] == 2025].copy()

rows = []
for arch in ARCH_ORDER:
    row = {'Archetype': arch}
    for tariff in ['T2', 'T3_TIB', 'T4']:
        for scenario in SCENARIOS:
            val = cstar_2025[(cstar_2025['archetype']==arch) & (cstar_2025['tariff']==tariff) & (cstar_2025['scenario']==scenario)]
            col = f"{tariff.replace('_TIB','')} {scenario[:4]}"
            row[col] = int(round(val['c_star'].values[0])) if len(val) else 'n/a'
    rows.append(row)

cstar_table = pd.DataFrame(rows)
print(cstar_table.to_string(index=False))

# --- TABLE 3: C* T2 Automated by Year ---
print("\n--- TABLE 3: C* T2 Automated by Year (€/yr) ---")
t2_auto = annual[(annual['tariff']=='T2') & (annual['scenario']=='automated')].copy()

rows = []
for arch in ARCH_ORDER:
    row = {'Archetype': arch}
    for year in [2023, 2024, 2025]:
        val = t2_auto[(t2_auto['archetype']==arch) & (t2_auto['year']==year)]
        row[str(year)] = int(round(val['c_star'].values[0])) if len(val) else 'n/a'
    rows.append(row)

t2_table = pd.DataFrame(rows)
print(t2_table.to_string(index=False))

# --- TABLE 4: Risk Metrics T2 Automated 2025 ---
print("\n--- TABLE 4: Risk Metrics T2 Automated, 2025 ---")
risk_2025 = risk_peryear[2025]
t2_risk = risk_2025[(risk_2025['tariff']=='T2') & (risk_2025['scenario']=='automated')].copy()

rows = []
for arch in ARCH_ORDER:
    val = t2_risk[t2_risk['archetype']==arch]
    if len(val):
        rows.append({
            'Archetype': arch,
            'Worst month premium (€)': round(val['worst_month_eur_2025'].values[0], 2),
            'Loss month probability': round(val['loss_month_prob_2025'].values[0], 2)
        })

risk_table = pd.DataFrame(rows)
print(risk_table.to_string(index=False))

print("\nDone.")