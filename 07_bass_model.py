"""
Bass Diffusion Model — Pipeline 3b
Models dynamic tariff adoption over time with smart meter infrastructure ceiling.

M(t): time-varying addressable market — EV households with home charging based on simulation results (×72% filter to exclude industrial and not at-home charging)
I(t): infra barrier as smart meter installed base — current rollout (§45 MsbG, 4.23M by 2032); I0 = 1.1M (BNetzA Q4 2025); L = 42M long-run ceiling
p:    innovation parameter — 3 scenarios: low (0.003), central (0.010), high (0.025)
q:    imitation parameter — 3 scenarios: low (0.15), central (0.30), high (0.50)
      Total: 3p × 3q = 9 combinations

Effective ceiling at each t: min(M(t), I(t))
Adoption = min(unconstrained Bass, effective ceiling)

Key sources:
  Agora Verkehrswende & BCG (2024) — M(t) EV fleet trajectory, reference scenario
  Agora Energiewende & FfE (2023) §4.1.2 — 72% residential home charging filter
  KBA Pressemitteilung Nr. 08/2026 — cross-check: 2.03M BEV registered 01.01.2026
  BNetzA Q4 2025 — I0 = 1.1M smart meters installed
  Verma et al. (2025) — p and q empirical range for smart meter opt-in
  Becker et al. (2009) via Massiani & Gohs (2015) — p policy scenarios
  vzbv (2024) — 81% poorly informed about dynamic tariffs; anchors low p scenario
  Turk & Trkman (2012) — q upper bound from German broadband diffusion

Outputs:
  data_simulation/bass_adoption.csv
  data_simulation/plot_adoption_3panels.png
  data_simulation/plot_wedge_3panels.png
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs('data_simulation', exist_ok=True)

# ============================================================
# TIME AXIS — 2025-2035
# ============================================================
YEARS = list(range(2025, 2036))
T = np.array([y - 2025 for y in YEARS], dtype=float)

# ============================================================
# ADDRESSABLE MARKET M(t)
# ============================================================
M_SERIES = {
    2025: 1_730_000,
    2026: 2_300_000,
    2027: 3_100_000,
    2028: 4_100_000,
    2029: 5_260_000,
    2030: 6_410_000,
    2031: 7_340_000,
    2032: 8_280_000,
    2033: 9_220_000,
    2034: 10_150_000,
    2035: 11_090_000,
}

M_T = np.array([M_SERIES[y] for y in YEARS], dtype=float)

print(f"Addressable market M(t): EV households with home charging (x72% filter)")
print(f"  2025: {M_T[0]/1e6:.2f}M -> 2035: {M_T[-1]/1e6:.2f}M")

# ============================================================
# BASS PARAMETERS — 3 p x 3 q = 9 scenarios
# ============================================================
P_SCENARIOS = {
    'low':     {'p': 0.003, 'label': 'Low awareness (p=0.003)'},
    'central': {'p': 0.010, 'label': 'Central awareness (p=0.010)'},
    'high':    {'p': 0.025, 'label': 'High awareness (p=0.025)'},
}

Q_SCENARIOS = {
    'low':     {'q': 0.15, 'label': 'Low imitation (q=0.15)'},
    'central': {'q': 0.30, 'label': 'Central imitation (q=0.30)'},
    'high':    {'q': 0.50, 'label': 'High imitation (q=0.50)'},
}

# ============================================================
# SMART METER ROLLOUT I(t) — single scenario: current rollout
# 4.23M = 90% of §14a device holders mandated by §45 MsbG by 2032
# I0 = 1.1M reflects §14a subpopulation installed base (BNetzA Q4 2025),
# variation from total smart meter rollout in 2025 as I(t) follows the §45 MsbG mandate trajectory
# ============================================================
I0 = 1_100_000
L = 42_000_000

# Target of 90% of households in mandate by 2032

I_WAYPOINTS = {
    'current': (4_230_000, 2032),
}

def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def find_k(waypoint, t_target, t0, L=L, I0=I0):
    def objective(k):
        raw0 = logistic(0, L, k, t0)
        raw_t = logistic(t_target, L, k, t0)
        scaled_t = raw_t * (I0 / raw0)
        return scaled_t - waypoint
    k = brentq(objective, 0.001, 50.0)
    return k

def make_I(scenario):
    waypoint, waypoint_year = I_WAYPOINTS[scenario]
    t_target = waypoint_year - 2025
    t0 = t_target * 0.7  # inflection point at 70% of waypoint year (≈2030); reflects expected rollout acceleration through late 2020s before mandatory segment saturates at 2032 waypoint
    k = find_k(waypoint, t_target, t0)
    raw = logistic(T, L, k, t0)
    scaled = raw * (I0 / raw[0])
    result = np.clip(scaled, 0, L)
    return result

I_T = make_I('current')

# ============================================================
# BASS DIFFUSION FUNCTIONS
# ============================================================
def bass_cumulative(t, p, q, M):
    num = 1 - np.exp(-(p + q) * t)
    den = 1 + (q / p) * np.exp(-(p + q) * t)
    return M * (num / den)

def compute_adoption(t, p, q, M_t, I_t):
    M_longrun = M_t[-1]
    unc = bass_cumulative(t, p, q, M_longrun)
    ceiling = np.minimum(M_t, I_t)
    con = np.minimum(unc, ceiling)
    return unc, con

# ============================================================
# RUN MODEL — 3 p x 3 q = 9 combinations
# ============================================================
rows = []

for p_label, p_params in P_SCENARIOS.items():
    for q_label, q_params in Q_SCENARIOS.items():
        p = p_params['p']
        q = q_params['q']
        unc, con = compute_adoption(T, p, q, M_T, I_T)

        for i, year in enumerate(YEARS):
            rows.append({
                'year': year,
                'p_scenario': p_label,
                'q_scenario': q_label,
                'p': p,
                'q': q,
                'M_t': M_T[i],
                'I_t': I_T[i],
                'effective_ceiling': min(M_T[i], I_T[i]),
                'adoption_unconstrained': unc[i],
                'adoption_constrained': con[i],
                'adoption_rate_of_M': con[i] / M_T[i] if M_T[i] > 0 else 0,
                'adoption_rate_of_I': con[i] / I_T[i] if I_T[i] > 0 else 0,
                'infra_wedge': unc[i] - con[i],
                'infra_wedge_pct': (unc[i] - con[i]) / unc[i] * 100 if unc[i] > 0 else 0,
            })

df = pd.DataFrame(rows)
df.to_csv('data_simulation/bass_adoption.csv', index=False)
print(f"\nSaved: data_simulation/bass_adoption.csv ({len(df)} rows)")

# ============================================================
# PRINT: I(t) verification
# ============================================================
print("\n=== SMART METER ROLLOUT I(t) — current scenario ===")
print(f"{'Year':<8}{'I(t)':>12}")
for i, year in enumerate(YEARS):
    print(f"{year:<8}{I_T[i]/1e6:>11.2f}M")

# ============================================================
# PRINT: SUMMARY TABLE
# ============================================================
print("\n=== ADOPTION SUMMARY — 9 combinations (ordered by q scenario) ===")
print(f"\n{'q scenario':<12}{'p scenario':<12}{'2030':>14}{'2032':>14}{'2035':>14}")
print("-" * 68)
for q_label in Q_SCENARIOS.keys():
    for p_label in P_SCENARIOS.keys():
        subset = df[(df['p_scenario'] == p_label) & (df['q_scenario'] == q_label)]
        y30 = subset[subset['year'] == 2030]['adoption_constrained'].values[0]
        y32 = subset[subset['year'] == 2032]['adoption_constrained'].values[0]
        y35 = subset[subset['year'] == 2035]['adoption_constrained'].values[0]
        print(f"{q_label:<12}{p_label:<12}"
              f"{y30/1e6:>8.2f}M ({y30/M_T[5]*100:.0f}%)"
              f"{y32/1e6:>8.2f}M ({y32/M_T[7]*100:.0f}%)"
              f"{y35/1e6:>8.2f}M ({y35/M_T[10]*100:.0f}%)")
    print()

# ============================================================
# PRINT: PARAMETER SUMMARY
# ============================================================
print("\n=== PARAMETER SUMMARY ===")
print(f"\nM(t): EV households with home charging (x72% filter)")
print(f"  2025: {M_T[0]/1e6:.2f}M -> 2035: {M_T[-1]/1e6:.2f}M")
print(f"  Source: Agora Verkehrswende & BCG (2024), Abbildung 1, reference scenario")
print(f"  Post-2030: linear extrapolation at +1.3M/yr")
print(f"\nI(t): current rollout (§45 MsbG), waypoint 4.23M by 2032")
print(f"  I0 = {I0/1e6:.1f}M (BNetzA Q4 2025)")
print(f"  L  = {L/1e6:.0f}M (long-run smart meter ceiling)")
print(f"\np scenarios:")
for label, params in P_SCENARIOS.items():
    print(f"  {label}: p={params['p']} — {params['label']}")
print(f"\nq scenarios:")
for label, params in Q_SCENARIOS.items():
    print(f"  {label}: q={params['q']} — {params['label']}")

# ============================================================
# FIGURE 1 — 3 panels, one per q scenario
# Each panel: 3 adoption lines (low/central/high p) + M(t) and I(t)
# ============================================================
colors_p = {
    'low':     '#888780',
    'central': '#1D9E75',
    'high':    '#D85A30',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, (q_label, q_params) in enumerate(Q_SCENARIOS.items()):
    ax = axes[idx]

    ax.plot(YEARS, M_T / 1e6,
            color='#E8A838', linestyle='--', linewidth=1.5,
            label='M(t): EV households x72%', alpha=0.8)

    ax.plot(YEARS, I_T / 1e6,
            color='#378ADD', linestyle='--', linewidth=1.5,
            label='I(t): smart meter rollout', alpha=0.8)

    for p_label, p_params in P_SCENARIOS.items():
        subset = df[(df['p_scenario'] == p_label) & (df['q_scenario'] == q_label)]
        ax.plot(subset['year'],
                subset['adoption_constrained'] / 1e6,
                color=colors_p[p_label],
                linewidth=2,
                label=p_params['label'])

    ax.set_title(q_params['label'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_xlim(2025, 2035)
    ax.set_ylim(0, 12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.set_ylabel('Households on dynamic tariff (millions)', fontsize=10)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Dynamic tariff adoption — 3p x 3q scenarios, current I(t) rollout\n'
             'M(t) = EV households x72% (orange), I(t) = §45 MsbG mandate (blue)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('data_simulation/plot_adoption_3panels.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: data_simulation/plot_adoption_3panels.png")

# ============================================================
# FIGURE 2 — Infrastructure wedge
# 3 panels, one per q scenario
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for idx, (q_label, q_params) in enumerate(Q_SCENARIOS.items()):
    ax = axes[idx]

    for p_label, p_params in P_SCENARIOS.items():
        subset = df[(df['p_scenario'] == p_label) & (df['q_scenario'] == q_label)]
        years = subset['year'].values
        con = subset['adoption_constrained'].values / 1e6
        unc = subset['adoption_unconstrained'].values / 1e6

        ax.plot(years, con,
                color=colors_p[p_label],
                linewidth=2,
                label=p_params['label'])

        ax.plot(years, unc,
                color=colors_p[p_label],
                linewidth=1,
                linestyle='--',
                alpha=0.5)

        ax.fill_between(years, con, unc,
                        color=colors_p[p_label],
                        alpha=0.10)

    ax.set_title(q_params['label'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10)
    ax.set_xlim(2025, 2035)
    ax.set_ylim(0, 12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))
    ax.grid(True, alpha=0.2)
    if idx == 0:
        ax.set_ylabel('Households on dynamic tariff (millions)', fontsize=10)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Infrastructure wedge — unconstrained (dashed) vs constrained (solid)\n'
             'Shaded area = households blocked by min(M(t), I(t)) ceiling',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('data_simulation/plot_wedge_3panels.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: data_simulation/plot_wedge_3panels.png")

print("\nDone.")