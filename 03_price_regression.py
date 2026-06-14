"""
Price Regression — Pipeline 1c
OLS regression: price_eur_mwh ~ residual_load_gw + hour FEs + month FEs
Run separately for 2023, 2024, 2025 to characterise the price-residual load
relationship across different price environments.

Model specification:
  Dependent variable: EPEX Spot Day-Ahead price (€/MWh)
  Key regressor: residual load (GW) — grid load minus renewable generation
  Fixed effects: hour of day (24 levels), month (12 levels)
  Standard errors: HC3 heteroskedasticity-robust
  Rationale: residual load is the primary determinant of marginal price under
    the merit order; hour and month FEs absorb intraday and seasonal patterns
    not fully mediated through residual load

Outputs:
  data_outputs/table1_price_regression.csv — Table 1 for dissertation

Inputs:
  data_clean/master_hourly.csv

Sources:
  SMARD.de — EPEX Spot Day-Ahead prices and realised consumption
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

os.makedirs('data_outputs', exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
master = pd.read_csv('data_clean/master_hourly.csv', index_col='datetime', parse_dates=True)

# Safety dropna — spring-forward DST hours already removed in 01_data_cleaning.py
# This catches any residual NaN in residual_load or price_eur_mwh
master = master.dropna(subset=['residual_load', 'price_eur_mwh'])

# Rescale residual load from MWh to GW for cleaner coefficient interpretation
master['residual_load_gw'] = master['residual_load'] / 1000

# ============================================================
# RUN REGRESSION FOR EACH YEAR
# ============================================================
results = {}

for year in [2025, 2024, 2023]:
    subset = master[master['year'] == year].copy()
    model = smf.ols(
        formula='price_eur_mwh ~ residual_load_gw + C(hour) + C(month)',
        data=subset
    ).fit(cov_type='HC3')
    results[year] = model
    print(f"{year}: N={int(model.nobs):,}, Adj R²={model.rsquared_adj:.3f}, "
          f"residual_load_gw coef={model.params['residual_load_gw']:.3f}, "
          f"p={model.pvalues['residual_load_gw']:.4f}")

# ============================================================
# SIGNIFICANCE STARS — verified from p-values
# ============================================================
def stars(pval):
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    else:
        return ''

# ============================================================
# PRICE SUMMARY STATISTICS
# ============================================================
price_stats = {}
for year in [2025, 2024, 2023]:
    subset = master[master['year'] == year]['price_eur_mwh']
    price_stats[year] = {'mean': subset.mean(), 'sd': subset.std()}

# ============================================================
# PRINT TABLE 1
# ============================================================
print(f"\n{'':30} {'(1) 2025':>12} {'(2) 2024':>12} {'(3) 2023':>12}")
print("─" * 70)

coefs = [f"{results[y].params['residual_load_gw']:.3f}"
         f"{stars(results[y].pvalues['residual_load_gw'])}" for y in [2025, 2024, 2023]]
ses   = [f"({results[y].bse['residual_load_gw']:.3f})" for y in [2025, 2024, 2023]]

print(f"{'Residual load (GW)':30} {coefs[0]:>12} {coefs[1]:>12} {coefs[2]:>12}")
print(f"{'':30} {ses[0]:>12} {ses[1]:>12} {ses[2]:>12}")
print("─" * 70)
print(f"{'Hour FEs':30} {'Yes':>12} {'Yes':>12} {'Yes':>12}")
print(f"{'Month FEs':30} {'Yes':>12} {'Yes':>12} {'Yes':>12}")
print(f"{'N':30} {int(results[2025].nobs):>12,} {int(results[2024].nobs):>12,} {int(results[2023].nobs):>12,}")
print(f"{'Adj. R²':30} {results[2025].rsquared_adj:>12.3f} {results[2024].rsquared_adj:>12.3f} {results[2023].rsquared_adj:>12.3f}")
print(f"{'Mean price (€/MWh)':30} {price_stats[2025]['mean']:>12.1f} {price_stats[2024]['mean']:>12.1f} {price_stats[2023]['mean']:>12.1f}")
print(f"{'SD price (€/MWh)':30} {price_stats[2025]['sd']:>12.1f} {price_stats[2024]['sd']:>12.1f} {price_stats[2023]['sd']:>12.1f}")
print("─" * 70)
print("Note: HC3 robust SEs in parentheses. *** p<0.01, ** p<0.05, * p<0.10.")
print("      Adjusted R² reported.")

# ============================================================
# HOUR AND MONTH FIXED EFFECTS (2025 as primary year)
# ============================================================
print("\n── Hour fixed effects (baseline = hour 0) ──")
hour_fe = {int(k.split('[T.')[1].rstrip(']')): v
           for k, v in results[2025].params.items()
           if 'C(hour)' in k}
for h in sorted(hour_fe):
    sign = "+" if hour_fe[h] > 0 else "-"
    print(f"  Hour {h:02d}: {sign}€{abs(hour_fe[h]):.1f}/MWh")

print("\n── Month fixed effects (baseline = January) ──")
month_fe = {int(k.split('[T.')[1].rstrip(']')): v
            for k, v in results[2025].params.items()
            if 'C(month)' in k}
month_names = {2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
               7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
for m in sorted(month_fe):
    sign = "+" if month_fe[m] > 0 else "-"
    print(f"  {month_names[m]}: {sign}€{abs(month_fe[m]):.1f}/MWh")

# ============================================================
# R² DECOMPOSITION (2025 — for dissertation write-up)
# ============================================================
# Shows how much of price variation is explained by residual load alone
# vs after adding hour and month FEs
# Interpretation: daily/seasonal patterns largely mediated through residual
# load rather than independent of it — confirms price signal is structural
subset_2025 = master[master['year'] == 2025].copy()
m_no_fe = smf.ols('price_eur_mwh ~ residual_load_gw', data=subset_2025).fit()
m_hour  = smf.ols('price_eur_mwh ~ residual_load_gw + C(hour)', data=subset_2025).fit()
m_month = smf.ols('price_eur_mwh ~ residual_load_gw + C(month)', data=subset_2025).fit()
m_full  = smf.ols('price_eur_mwh ~ residual_load_gw + C(hour) + C(month)', data=subset_2025).fit()

print("\n── R² decomposition (2025) ──")
print(f"  Residual load only:        {m_no_fe.rsquared:.3f}")
print(f"  + Hour FEs:                {m_hour.rsquared:.3f}")
print(f"  + Month FEs:               {m_month.rsquared:.3f}")
print(f"  + Both FEs (full model):   {m_full.rsquared:.3f}")

# ============================================================
# EXPORT TABLE 1
# ============================================================
table1 = pd.DataFrame({
    '': ['Residual load (GW)', '', 'Hour FEs', 'Month FEs', 'N',
         'Adj. R²', 'Mean price (€/MWh)', 'SD price (€/MWh)'],
    '(1) 2025': [
        f"{results[2025].params['residual_load_gw']:.3f}"
        f"{stars(results[2025].pvalues['residual_load_gw'])}",
        f"({results[2025].bse['residual_load_gw']:.3f})",
        'Yes', 'Yes',
        f"{int(results[2025].nobs):,}",
        f"{results[2025].rsquared_adj:.3f}",
        f"{price_stats[2025]['mean']:.1f}",
        f"{price_stats[2025]['sd']:.1f}"
    ],
    '(2) 2024': [
        f"{results[2024].params['residual_load_gw']:.3f}"
        f"{stars(results[2024].pvalues['residual_load_gw'])}",
        f"({results[2024].bse['residual_load_gw']:.3f})",
        'Yes', 'Yes',
        f"{int(results[2024].nobs):,}",
        f"{results[2024].rsquared_adj:.3f}",
        f"{price_stats[2024]['mean']:.1f}",
        f"{price_stats[2024]['sd']:.1f}"
    ],
    '(3) 2023': [
        f"{results[2023].params['residual_load_gw']:.3f}"
        f"{stars(results[2023].pvalues['residual_load_gw'])}",
        f"({results[2023].bse['residual_load_gw']:.3f})",
        'Yes', 'Yes',
        f"{int(results[2023].nobs):,}",
        f"{results[2023].rsquared_adj:.3f}",
        f"{price_stats[2023]['mean']:.1f}",
        f"{price_stats[2023]['sd']:.1f}"
    ]
})

table1.to_csv('data_outputs/table1_price_regression.csv', index=False)
print("\nSaved: data_outputs/table1_price_regression.csv")
print("\nDone.")