"""Run after analysis.py (same folder). Draws Figures 3-6 of the revised manuscript from the outputs of analysis.py."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT = os.environ.get("ANALYSIS_OUT", os.path.join(HERE, "out")) + "/"; FIG = os.environ.get("FIG_OUT", os.path.join(HERE, "fig_paper")) + "/"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
W = 3.4
mo = pd.read_csv(OUT + "tab/T5_monthly_M-cal.csv"); x = np.arange(1, 13); lab = [f"{m:02d}" for m in x]
# Fig 3: monthly Macro-F1
fig, ax = plt.subplots(figsize=(W, 2.3))
ax.plot(x, mo["Macro-F1"], marker="o", color="#2a6fdb", lw=1.6)
for xi, v in zip(x, mo["Macro-F1"]): ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, -10 if xi == 3 else 5), ha="center", fontsize=5.5)
ax.set_xticks(x, lab); ax.set_xlabel("Month of 2024"); ax.set_ylabel("Macro-F1"); ax.set_ylim(0.70, 0.93); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG + "fig3.png", dpi=300); plt.close()
# Fig 4: Sulphide recall with Wilson intervals (+ n), and true vs predicted Sulphide share
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
r = mo["Sulphide recall"].values; lo, hi = mo["Sul recall Wilson lo"].values, mo["Sul recall Wilson hi"].values
a.errorbar(x, r, yerr=[r - lo, hi - r], fmt="o-", color="#e8590c", ecolor="#999999", capsize=2, lw=1.4, ms=3.5)
for xi, n in zip(x, mo["Sulphide n"]): a.annotate(f"n={n}", (xi, 0.05), ha="center", fontsize=5)
a.set_ylim(0, 1.05); a.set_ylabel("Sulphide recall\n(95% Wilson CI)"); a.grid(alpha=.3)
b.plot(x, 100 * mo["True Sulphide share"], "o-", color="black", lw=1.4, ms=3, label="True share (needs labels)")
b.plot(x, 100 * mo["Predicted Sulphide share"], "s--", color="#2a6fdb", lw=1.4, ms=3, label="Predicted share (label-free)")
b.axhline(16.6, color="grey", ls=":", lw=1, label="Share in calibration slice (Oct–Dec 2023)")
b.set_ylabel("Sulphide share (%)"); b.set_xlabel("Month of 2024"); b.set_xticks(x, lab); b.set_ylim(0, 19); b.grid(alpha=.3)
b.legend(fontsize=5.5, loc="center", bbox_to_anchor=(0.5, 0.6), ncol=1, frameon=False)
plt.tight_layout(); plt.savefig(FIG + "fig4.png", dpi=300); plt.close()
# Fig 5: monthly Macro-F1 under P0, P1, P2 (θ = 0.15), label delay 1 month
ore = pd.read_csv(os.path.join(HERE, "data", "ore_type_data.csv"), parse_dates=["date"]).sort_values(["date", "sample_id"]).reset_index(drop=True)
te = ore[ore.date.dt.year == 2024]; yte = pd.Categorical(te.ore_type_label, categories=["Mixed", "Oxide", "Sulphide"]).codes; mth = te.date.dt.month.values
pol = pickle.load(open(OUT + "policy_cache.pkl", "rb"))
rows = []
for (d, nm), (p, P, log) in pol.items():
    for m in range(1, 13):
        k = mth == m; rows.append({"Δ": d, "Policy": nm, "month": m, "Macro-F1": f1_score(yte[k], p[k], average="macro")})
pm = pd.DataFrame(rows); pm.to_csv(FIG + "policy_monthly_macroF1.csv", index=False)
fig, ax = plt.subplots(figsize=(W, 2.6))
for nm, col, mk, lb in [("P0 No retraining", "black", "o", "P0 no retraining"), ("P1 Monthly retraining", "#2a6fdb", "s", "P1 monthly retraining"),
                        ("P2 Feature-PSI trigger θ=0.15", "#e8590c", "^", "P2 PSI-triggered (θ = 0.15)")]:
    s = pm[(pm["Δ"] == 1) & (pm.Policy == nm)].sort_values("month"); ax.plot(s.month, s["Macro-F1"], marker=mk, color=col, lw=1.4, ms=3.5, label=lb)
ax.set_xticks(x, lab); ax.set_xlabel("Month of 2024"); ax.set_ylabel("Macro-F1"); ax.grid(alpha=.3)
ax.legend(fontsize=6, frameon=False, loc="lower right"); plt.tight_layout(); plt.savefig(FIG + "fig5.png", dpi=300); plt.close()
# Fig 6: MAPE per walk-forward window
fw = pd.read_csv(OUT + "tab/T_forecast_windows.csv")
fig, ax = plt.subplots(figsize=(W, 2.6))
for nm, col, mk, lb in [("Naïve (t−1)", "black", "o", "Naïve (t−1)"), ("Rolling mean 30 d", "grey", "s", "30-day rolling mean"),
                        ("[Leaky] LightGBM (lags + same-day ore)", "#e8590c", "v", "LightGBM, original (same-day ore)"),
                        ("LightGBM (lags + calendar + ore, t−1)", "#2a6fdb", "^", "LightGBM, corrected (ore of t−1)")]:
    s = fw[fw.Model == nm]; ax.plot(pd.to_datetime(s.cut), s["MAPE%"], marker=mk, color=col, lw=1.2, ms=3, label=lb)
ax.set_ylabel("MAPE (%) per 30-day window"); ax.set_xlabel("Window start"); ax.grid(alpha=.3); ax.legend(fontsize=5.5, frameon=False)
fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(FIG + "fig6.png", dpi=300); plt.close()
print(pm[(pm["Δ"] == 1) & (pm.month == 3)].round(4).to_string())
print(fw.Model.unique())
