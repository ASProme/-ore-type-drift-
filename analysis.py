# %% [markdown]
# # Ore-type classification and tonnage forecasting under drift — revised, single-run analysis
#
# Every number in the revised manuscript comes from THIS script in ONE run, with ONE model
# configuration (CFG below) and fixed seeds. Outputs: out/tab/*.csv, out/fig/*.png (or out/fig_col with
# COLUMN_FIGS=1), out/results.json. Figures 3-6 of the paper are drawn from these outputs by make_figs.py.
# Run:  COLUMN_FIGS=1 python analysis.py   (about 3-20 min depending on the machine)

# %%
import json, time, warnings, os
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize_scalar
from scipy.stats import norm

warnings.filterwarnings("ignore")
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); TAB = os.path.join(OUT, "tab")
# COLUMN_FIGS=1 renders every figure at single-column journal width (3.4 in) into out/fig_col
COL = os.environ.get("COLUMN_FIGS") == "1"
FIG = os.path.join(OUT, "fig_col" if COL else "fig")
COLSIZE = {"f1": (3.45, 3.6), "f2": (3.45, 2.5), "f3": (3.45, 4.6), "f4": (3.45, 2.5), "f5": (3.45, 3.4), "f6": (3.45, 2.8), "f7": (3.45, 2.9)}
SM = 5.5 if COL else 7
def FS(key, full):
    return COLSIZE[key] if COL else full
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
R = {}  # all headline numbers -> results.json

SEED = 42
N_BOOT = 2000
CLASSES = np.array(["Mixed", "Oxide", "Sulphide"])
SUL = 2
NUM = ["depth_mean", "fe_pct", "cu_pct", "sio2_pct", "moisture_pct", "au_gpt"]
TIME = ["month", "quarter", "dayofyear"]
MISS = [f"{c}_is_missing" for c in NUM]
CAT = ["pit"]
FEATS = NUM + TIME + MISS + CAT

# ONE classifier configuration used everywhere (main model, policies, pit hold-out, ablations)
CFG = dict(objective="multiclass", n_estimators=700, learning_rate=0.03, num_leaves=63,
           subsample=0.85, subsample_freq=1, colsample_bytree=0.85, random_state=SEED,
           n_jobs=2, verbose=-1)
R["config"] = {k: v for k, v in CFG.items() if k not in ("n_jobs", "verbose")}

# Palette (reference categorical slots; validated adjacent-pair CVD-safe)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAG, C_GREEN, C_VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
C_GREY = "#8a8985"
plt.rcParams.update({"font.size": 7 if COL else 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.3 if COL else 2, "lines.markersize": 3.2 if COL else 5, "savefig.dpi": 300,
                     "figure.dpi": 100, "axes.titlesize": 10, "legend.frameon": False})

def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close()

def savetab(df, name):
    df.to_csv(os.path.join(TAB, name), index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows symbols correctly

# %% [markdown]
# ## 1. Data loading and audit  (Reviewer 2: data table, units, missingness, class counts by year × pit)

# %%
ore = pd.read_csv(os.path.join(HERE, "data", "ore_type_data.csv"), parse_dates=["date"])
ton = pd.read_csv(os.path.join(HERE, "data", "simulated_tonnage.csv"), parse_dates=["date"])
ore = ore.sort_values(["date", "sample_id"]).reset_index(drop=True)
ore["year"] = ore.date.dt.year
ore["month"] = ore.date.dt.month
ore["quarter"] = ore.date.dt.quarter
ore["dayofyear"] = ore.date.dt.dayofyear
ore["mi"] = (ore.date.dt.year - 2023) * 12 + ore.date.dt.month - 1   # global month index 0..23
for c in NUM:
    ore[f"{c}_is_missing"] = ore[c].isna().astype(int)
ore["y"] = pd.Categorical(ore.ore_type_label, categories=CLASSES).codes

units = {"depth_mean": "m", "fe_pct": "wt%", "cu_pct": "wt%", "sio2_pct": "wt%",
         "moisture_pct": "wt%", "au_gpt": "g/t (= ppm)"}
desc = {"depth_mean": "Mean sample depth", "fe_pct": "Iron grade", "cu_pct": "Copper grade",
        "sio2_pct": "Silica content", "moisture_pct": "Moisture content", "au_gpt": "Gold grade"}
rows = []
for c in NUM:
    s = ore[c]
    rows.append({"Variable": c, "Description": desc[c], "Unit (assumed)": units[c],
                 "Missing n": int(s.isna().sum()), "Missing %": round(100 * s.isna().mean(), 2),
                 "Min": round(s.min(), 3), "Median": round(s.median(), 3), "Mean": round(s.mean(), 3),
                 "Max": round(s.max(), 3), "Exact zeros": int((s == 0).sum())})
rows.append({"Variable": "pit", "Description": "Pit identifier (PitA/PitB/PitC)", "Unit (assumed)": "categorical",
             "Missing n": 0, "Missing %": 0.0})
rows.append({"Variable": "date", "Description": "Sampling date (2023-01-01 to 2024-12-30; 730 sampling days)", "Unit (assumed)": "day",
             "Missing n": 0, "Missing %": 0.0})
rows.append({"Variable": "ore_type_label", "Description": "Target class (Mixed/Oxide/Sulphide)",
             "Unit (assumed)": "categorical", "Missing n": 0, "Missing %": 0.0})
data_table = pd.DataFrame(rows); savetab(data_table, "T_data_variables.csv")

cc = (ore.groupby(["year", "pit", "ore_type_label"]).size().unstack("ore_type_label").fillna(0).astype(int))
cc["Total"] = cc.sum(axis=1)
cc["Sulphide %"] = (100 * cc["Sulphide"] / cc["Total"]).round(1)
cc = cc.reset_index()
yr = ore.groupby(["year", "ore_type_label"]).size().unstack().reset_index()
yr["pit"] = "All"; yr["Total"] = yr[CLASSES].sum(axis=1); yr["Sulphide %"] = (100 * yr.Sulphide / yr.Total).round(1)
class_counts = pd.concat([cc, yr[cc.columns]], ignore_index=True).sort_values(["year", "pit"])
savetab(class_counts, "T_class_counts_year_pit.csv")

monthly_counts = ore.groupby(["year", "month", "ore_type_label"]).size().unstack().fillna(0).astype(int).reset_index()
monthly_counts["Total"] = monthly_counts[CLASSES].sum(axis=1)
monthly_counts["Sulphide %"] = (100 * monthly_counts.Sulphide / monthly_counts.Total).round(1)
savetab(monthly_counts, "T_class_counts_month.csv")
mp = ore.groupby(["year", "month", "pit", "ore_type_label"]).size().unstack().fillna(0).astype(int).reset_index()
savetab(mp, "T_class_counts_month_pit.csv")

audit = {
    "n_samples": int(len(ore)), "n_days": int(ore.date.nunique()),
    "samples_per_day_min": int(ore.groupby("date").size().min()),
    "samples_per_day_max": int(ore.groupby("date").size().max()),
    "rows_with_any_missing": int(ore[NUM].isna().any(axis=1).sum()),
    "rows_with_any_missing_pct": round(100 * ore[NUM].isna().any(axis=1).mean(), 1),
    "missing_rate_by_class": ore.assign(m=ore[NUM].isna().any(axis=1)).groupby("ore_type_label").m.mean().round(3).to_dict(),
    "fe_gt_72_4": int((ore.fe_pct > 72.4).sum()), "fe_gt_100": int((ore.fe_pct > 100).sum()),
    "fe_max": round(float(ore.fe_pct.max()), 1),
    "sulphide_share_2023": round(float((ore[ore.year == 2023].y == SUL).mean()), 4),
    "sulphide_share_2024": round(float((ore[ore.year == 2024].y == SUL).mean()), 4),
    "oxide_share_2023": round(float((ore[ore.year == 2023].y == 1).mean()), 4),
    "oxide_share_2024": round(float((ore[ore.year == 2024].y == 1).mean()), 4),
    "class_means": ore.groupby("ore_type_label")[NUM].mean().round(3).to_dict(orient="index"),
    "ton_n": int(len(ton)), "ton_start": str(ton.date.min().date()), "ton_end": str(ton.date.max().date()),
    "ton_mean_2023": round(float(ton[ton.date.dt.year == 2023].tonnage.mean()), 1),
    "ton_mean_2024": round(float(ton[ton.date.dt.year == 2024].tonnage.mean()), 1),
}
R["audit"] = audit
from sklearn.tree import DecisionTreeClassifier
_cc = ore.dropna(subset=NUM)
R["label_tree_depth3_acc"] = float(DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(_cc[NUM], _cc.y).score(_cc[NUM], _cc.y))
R["ton_autocorr_lag1"] = float(ton.sort_values("date").tonnage.autocorr(1))
print("audit done", round(time.time() - T0), "s")

# %% [markdown]
# ## 2. Model builders and metrics (identical everywhere)

# %%
def make_pre(scale=False):
    num_pipe = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("num", Pipeline(num_pipe), NUM + TIME + MISS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])

def make_lgbm(feats=FEATS):
    numf = [f for f in feats if f != "pit"]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), numf),
                             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("oh", OneHotEncoder(handle_unknown="ignore"))]), ["pit"])])
    return Pipeline([("pre", pre), ("clf", LGBMClassifier(**CFG))])

def fit_lgbm(df, feats=FEATS):
    m = make_lgbm(feats); m.fit(df[feats], df.y.values); return m

class IsoCal:
    """One-vs-rest isotonic calibration of a FROZEN base model, fitted on a held-out time slice to each class's
    predicted PROBABILITY, followed by renormalisation (Zadrozny & Elkan, 2002). Note: scikit-learn's
    CalibratedClassifierCV(method='isotonic') fits the curves to the raw decision-function scores instead when the
    estimator exposes them (a Pipeline ending in LightGBM does); that variant is reported as a sensitivity check."""
    def fit(self, P, y):
        self.iso = [IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(P[:, k], (y == k).astype(float))
                    for k in range(P.shape[1])]
        return self
    def transform(self, P):
        Q = np.column_stack([self.iso[k].predict(P[:, k]) for k in range(P.shape[1])])
        s = Q.sum(1, keepdims=True); Q = np.where(s > 0, Q / np.where(s > 0, s, 1), 1 / P.shape[1])
        return Q

class TempCal:
    def fit(self, P, y):
        L = np.log(np.clip(P, 1e-12, 1))
        f = lambda t: -np.mean(np.log(np.clip(self._sm(L / t)[np.arange(len(y)), y], 1e-12, 1)))
        self.t = minimize_scalar(f, bounds=(0.05, 20), method="bounded").x; return self
    @staticmethod
    def _sm(Z):
        Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)
    def transform(self, P):
        return self._sm(np.log(np.clip(P, 1e-12, 1)) / self.t)

class Deployed:
    """Base LightGBM fitted on all-but-last-3-months of the available labelled data,
    isotonic calibrator fitted on the last 3 months (time-ordered hold-out)."""
    def __init__(self, labelled, cal_months=3, feats=FEATS):
        cut = labelled.mi.max() - cal_months + 1
        fit_df, cal_df = labelled[labelled.mi < cut], labelled[labelled.mi >= cut]
        self.base = fit_lgbm(fit_df, feats); self.feats = feats
        self.cal = IsoCal().fit(self.base.predict_proba(cal_df[feats]), cal_df.y.values)
        self.train_cutoff_mi = int(labelled.mi.max()); self.ref = labelled
        self.cal_pred_dist = np.bincount(self.predict(cal_df), minlength=3) / len(cal_df)
    def proba(self, df):
        return self.cal.transform(self.base.predict_proba(df[self.feats]))
    def predict(self, df):
        return self.proba(df).argmax(1)

# ---- fast metrics on integer labels ----
def confmat(y, p, k=3):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)

def macro_f1(y, p):
    cm = confmat(y, p); tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn; f = np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)
    return float(f.mean())

def bal_acc(y, p):
    cm = confmat(y, p); n = cm.sum(1); r = np.diag(cm) / np.where(n > 0, n, 1)
    return float(r[n > 0].mean())

def recall_k(y, p, k=SUL):
    n = (y == k).sum(); return float(((y == k) & (p == k)).sum() / n) if n else np.nan

def ece_bin(yb, pb, n_bins=10):
    bins = np.minimum((pb * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins); sp = np.bincount(bins, pb, n_bins); sy = np.bincount(bins, yb, n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(sy[nz] - sp[nz])) / len(pb))

def brier_bin(yb, pb): return float(np.mean((pb - yb) ** 2))
def nll_bin(yb, pb):
    pb = np.clip(pb, 1e-12, 1 - 1e-12); return float(-np.mean(yb * np.log(pb) + (1 - yb) * np.log(1 - pb)))
def toplabel_ece(y, P, n_bins=10):
    return ece_bin((P.argmax(1) == y).astype(float), P.max(1), n_bins)
def classwise_ece(y, P, n_bins=10):
    return float(np.mean([ece_bin((y == k).astype(float), P[:, k], n_bins) for k in range(P.shape[1])]))
def mc_brier(y, P):
    Y = np.eye(P.shape[1])[y]; return float(np.mean(np.sum((P - Y) ** 2, 1)))
def mc_nll(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return (max(0.0, c - h), min(1.0, c + h))

def rc_curve(y, P):
    """Risk–coverage curve when ranking by max calibrated probability; returns coverage, risk, AURC."""
    conf = P.max(1); err = (P.argmax(1) != y).astype(float)
    o = np.argsort(-conf, kind="stable"); e = err[o]
    n = len(y); k = np.arange(1, n + 1)
    risk = np.cumsum(e) / k; cov = k / n
    return cov, risk, float(risk.mean())

# day-block bootstrap (resample whole sampling days -> respects within-day dependence)
def day_blocks(dates):
    codes, uniq = pd.factorize(pd.Series(dates).dt.normalize())
    order = np.argsort(codes, kind="stable"); counts = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return [order[s:s + c] for s, c in zip(starts, counts)]

def boot_idx(blocks, rng):
    pick = rng.integers(0, len(blocks), len(blocks))
    return np.concatenate([blocks[i] for i in pick])

def boot(stat_fn, blocks, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        vals.append(stat_fn(boot_idx(blocks, rng)))
    vals = np.array(vals, dtype=float)
    return vals

def ci(v, a=0.05):
    v = v[np.isfinite(v)] if np.ndim(v) == 1 else v
    return [float(np.quantile(v, a / 2, axis=0)) if np.ndim(v) == 1 else np.quantile(v, a / 2, axis=0),
            float(np.quantile(v, 1 - a / 2, axis=0)) if np.ndim(v) == 1 else np.quantile(v, 1 - a / 2, axis=0)]

def fmt_ci(x, lo, hi, d=3): return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

# %% [markdown]
# ## 3. Splits.  Train = 2023 (fit Jan–Sep, calibration Oct–Dec), Test = 2024

# %%
d23 = ore[ore.year == 2023].copy(); test = ore[ore.year == 2024].copy().reset_index(drop=True)
fit_df = d23[d23.month <= 9]; cal_df = d23[d23.month >= 10]
yte = test.y.values; blocks_te = day_blocks(test.date)
R["splits"] = {"train_2023": int(len(d23)), "fit_jan_sep_2023": int(len(fit_df)), "cal_oct_dec_2023": int(len(cal_df)),
               "test_2024": int(len(test)), "test_sulphide": int((yte == SUL).sum()),
               "cal_sulphide_share": round(float((cal_df.y == SUL).mean()), 4)}

# %% [markdown]
# ## 4. Model M-full: LightGBM trained on all of 2023 (uncalibrated) + baseline comparison (Table 3)

# %%
M_full = fit_lgbm(d23)
P_full = M_full.predict_proba(test[FEATS])

def eval_row(name, p, P=None, model_id=""):
    r = {"Model": name, "Model ID": model_id, "Macro-F1": macro_f1(yte, p), "Balanced Acc": bal_acc(yte, p),
         "Sulphide recall": recall_k(yte, p)}
    r["Sulphide PR-AUC"] = average_precision_score((yte == SUL).astype(int), P[:, SUL]) if P is not None else np.nan
    f = boot(lambda i: macro_f1(yte[i], p[i]), blocks_te, seed=1)
    r["Macro-F1 95% CI"] = ci(f)
    return r

base_rows = []
def run_baseline(name, est, scale):
    pipe = Pipeline([("pre", make_pre(scale)), ("clf", est)]); pipe.fit(d23[FEATS], d23.y.values)
    p = pipe.predict(test[FEATS]); P = pipe.predict_proba(test[FEATS]) if hasattr(pipe, "predict_proba") else None
    return eval_row(name, p, P, "B-" + name.split()[0])

base_rows.append(run_baseline("Logistic regression (multinomial)", LogisticRegression(max_iter=5000), True))
base_rows.append(run_baseline("Linear SVM", LinearSVC(dual="auto", max_iter=20000), True))
base_rows.append(run_baseline("Random forest", RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample",
                                                                      random_state=SEED, n_jobs=2), False))
try:
    from xgboost import XGBClassifier
    base_rows.append(run_baseline("XGBoost", XGBClassifier(n_estimators=700, learning_rate=0.03, max_depth=6, subsample=0.85,
                                                           colsample_bytree=0.85, random_state=SEED, n_jobs=2,
                                                           eval_metric="mlogloss"), False))
except ImportError:
    pass
base_rows.append(eval_row("LightGBM (all 2023, uncalibrated)", P_full.argmax(1), P_full, "M-full"))
print("baselines done", round(time.time() - T0), "s")

# %% [markdown]
# ## 5. Calibration with the base model held fixed (Reviewer 1, comment 1)
# M-base: LightGBM fitted on Jan–Sep 2023 only.  Calibrators (isotonic; temperature scaling as a
# comparator) fitted on Oct–Dec 2023 predictions of the FROZEN M-base.  Evaluated on 2024.

# %%
M_base = fit_lgbm(fit_df)
P_cal_raw = M_base.predict_proba(cal_df[FEATS])
P_te_raw = M_base.predict_proba(test[FEATS])
iso = IsoCal().fit(P_cal_raw, cal_df.y.values); tmp_ = TempCal().fit(P_cal_raw, cal_df.y.values)
P_te_iso = iso.transform(P_te_raw); P_te_ts = tmp_.transform(P_te_raw)
P_cal_iso = iso.transform(P_cal_raw)
R["temperature"] = float(tmp_.t)
# sensitivity: isotonic fitted to raw decision-function scores (scikit-learn default for this estimator)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
_raw = CalibratedClassifierCV(FrozenEstimator(M_base), method="isotonic").fit(cal_df[FEATS], cal_df.y.values)
_Pr = _raw.predict_proba(test[FEATS]); _pr = _Pr.argmax(1)
R["iso_rawscore"] = {"Macro-F1": macro_f1(yte, _pr), "Balanced Acc": bal_acc(yte, _pr), "Sulphide recall": recall_k(yte, _pr),
                     "Top-label ECE@10": toplabel_ece(yte, _Pr), "Multiclass NLL": mc_nll(yte, _Pr),
                     "Sul ECE@10": ece_bin((yte == SUL).astype(float), _Pr[:, SUL])}

variants = {"M-base (uncalibrated)": P_te_raw, "M-base + isotonic (M-cal)": P_te_iso,
            "M-base + temperature scaling": P_te_ts}
for nm, P in variants.items():
    base_rows.append(eval_row(nm, P.argmax(1), P, {"M-base (uncalibrated)": "M-base",
                                                    "M-base + isotonic (M-cal)": "M-cal",
                                                    "M-base + temperature scaling": "M-ts"}[nm]))
# paired day-block bootstrap: M-cal minus M-base on classification metrics
_pb, _pc = P_te_raw.argmax(1), P_te_iso.argmax(1)
_d = boot(lambda i: [macro_f1(yte[i], _pc[i]) - macro_f1(yte[i], _pb[i]), bal_acc(yte[i], _pc[i]) - bal_acc(yte[i], _pb[i]),
                     recall_k(yte[i], _pc[i]) - recall_k(yte[i], _pb[i])], blocks_te, seed=21)
R["mcal_minus_mbase"] = {k: {"diff": float(v0), "ci": [float(np.quantile(_d[:, j], 0.025)), float(np.quantile(_d[:, j], 0.975))]}
                         for j, (k, v0) in enumerate([("Macro-F1", macro_f1(yte, _pc) - macro_f1(yte, _pb)), ("Balanced Acc", bal_acc(yte, _pc) - bal_acc(yte, _pb)),
                                                       ("Sulphide recall", recall_k(yte, _pc) - recall_k(yte, _pb))])}
t1 = pd.DataFrame(base_rows)
t1["Macro-F1 (95% CI)"] = [fmt_ci(r["Macro-F1"], *r["Macro-F1 95% CI"], 4) for _, r in t1.iterrows()]
t1 = t1.drop(columns=["Macro-F1 95% CI"])
savetab(t1.round(8), "T1_classification_models.csv")
R["table1"] = t1.round(8).to_dict(orient="records")

def calib_metrics(P, y):
    yb = (y == SUL).astype(float); pb = P[:, SUL]
    return {"Sul ECE@10": ece_bin(yb, pb, 10), "Sul ECE@15": ece_bin(yb, pb, 15), "Sul Brier": brier_bin(yb, pb),
            "Sul NLL": nll_bin(yb, pb), "Top-label ECE@10": toplabel_ece(y, P, 10),
            "Classwise ECE@10": classwise_ece(y, P, 10), "Multiclass Brier": mc_brier(y, P), "Multiclass NLL": mc_nll(y, P)}

cal_rows = []; calib_boot = {}
for nm, P in variants.items():
    pt = calib_metrics(P, yte)
    bv = boot(lambda i: list(calib_metrics(P[i], yte[i]).values()), blocks_te, n=N_BOOT, seed=3)
    calib_boot[nm] = bv
    row = {"Model": nm}
    for j, k in enumerate(pt):
        lo, hi = np.quantile(bv[:, j], [0.025, 0.975]); row[k] = pt[k]; row[k + " CI"] = [float(lo), float(hi)]
    cal_rows.append(row)
# paired differences isotonic - uncalibrated
diff_row = {"Model": "Δ isotonic − uncalibrated (paired)"}
dv = calib_boot["M-base + isotonic (M-cal)"] - calib_boot["M-base (uncalibrated)"]
for j, k in enumerate(calib_metrics(P_te_raw, yte)):
    diff_row[k] = cal_rows[1][k] - cal_rows[0][k]
    diff_row[k + " CI"] = [float(np.quantile(dv[:, j], 0.025)), float(np.quantile(dv[:, j], 0.975))]
cal_rows.append(diff_row)
t2 = pd.DataFrame(cal_rows)
t2_pretty = pd.DataFrame({"Model": t2.Model})
for k in calib_metrics(P_te_raw, yte):
    t2_pretty[k] = [fmt_ci(r[k], *r[k + " CI"], 4) for _, r in t2.iterrows()]
savetab(t2_pretty, "T2_calibration.csv")
R["table2"] = t2_pretty.to_dict(orient="records")
# in-distribution check (calibration slice itself is optimistic; report Oct–Dec 2023 for reference only)
R["calib_on_cal_slice_iso"] = calib_metrics(P_cal_iso, cal_df.y.values)
R["calib_on_cal_slice_raw"] = calib_metrics(P_cal_raw, cal_df.y.values)

# reliability diagrams with bin counts
def rel_table(yb, pb, n_bins=10):
    bins = np.minimum((pb * n_bins).astype(int), n_bins - 1); rows = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0: continue
        lo, hi = wilson(int(yb[m].sum()), int(m.sum()))
        rows.append({"bin": f"[{b / n_bins:.1f},{(b + 1) / n_bins:.1f})", "n": int(m.sum()),
                     "mean_pred": float(pb[m].mean()), "obs_freq": float(yb[m].mean()), "obs_lo": lo, "obs_hi": hi})
    return pd.DataFrame(rows)

rel = []
for nm, P in [("Uncalibrated", P_te_raw), ("Isotonic", P_te_iso)]:
    a = rel_table((yte == SUL).astype(float), P[:, SUL]); a["Model"] = nm; a["Target"] = "Sulphide vs rest"; rel.append(a)
    b = rel_table((P.argmax(1) == yte).astype(float), P.max(1)); b["Model"] = nm; b["Target"] = "Top-label"; rel.append(b)
rel = pd.concat(rel); savetab(rel.round(8), "T_reliability_bins.csv")

fig, axes = plt.subplots(2, 2, figsize=FS("f1", (7.2, 5.6)), gridspec_kw={"height_ratios": [3, 1.2]}, sharex="col")
for j, tgt in enumerate(["Sulphide vs rest", "Top-label"]):
    ax = axes[0, j]; ax.plot([0, 1], [0, 1], ls="--", color=C_GREY, lw=1)
    for nm, col, off in [("Uncalibrated", C_ORANGE, -0.012), ("Isotonic", C_BLUE, 0.012)]:
        s = rel[(rel.Model == nm) & (rel.Target == tgt)]; small = s[s.n < 5]; s = s[s.n >= 5]
        ax.errorbar(s.mean_pred + off, s.obs_freq, yerr=[s.obs_freq - s.obs_lo, s.obs_hi - s.obs_freq],
                    fmt="o-", color=col, label={"Uncalibrated": "M-base (uncalibrated)", "Isotonic": "M-cal (isotonic)"}[nm], capsize=2, lw=1.6, ms=5)
        ax.plot(small.mean_pred + off, small.obs_freq, "x", color=col, ms=6)
    ax.set_title(f"{tgt}"); ax.set_ylabel("Observed frequency"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axb = axes[1, j]; w = 0.04
    for nm, col, off in [("Uncalibrated", C_ORANGE, -w / 2), ("Isotonic", C_BLUE, w / 2)]:
        s = rel[(rel.Model == nm) & (rel.Target == tgt)]
        axb.bar(s.mean_pred + off, s.n, width=w, color=col, edgecolor="white", linewidth=0.5)
    axb.set_yscale("log"); axb.set_ylabel("Bin count (log)"); axb.set_xlabel("Mean predicted probability")
h_, l_ = axes[0, 0].get_legend_handles_labels()
plt.tight_layout(rect=(0, 0.06, 1, 1))
fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
plt.savefig(os.path.join(FIG, "Fig1_reliability_with_counts.png"), bbox_inches="tight"); plt.close()
print("calibration done", round(time.time() - T0), "s")

# %% [markdown]
# ## 6. Selective prediction (Reviewer 1, comment 2).  τ chosen on the calibration slice (Oct–Dec 2023)
# to give 70% coverage there, then applied unchanged to 2024.

# %%
tau_iso = float(np.quantile(P_cal_iso.max(1), 0.30))
tau_raw = float(np.quantile(P_cal_raw.max(1), 0.30))
R["tau"] = {"isotonic": tau_iso, "uncalibrated": tau_raw}

def op_point(P, tau, y):
    keep = P.max(1) >= tau; p = P.argmax(1)
    return {"coverage": float(keep.mean()), "risk": float((p[keep] != y[keep]).mean()) if keep.any() else np.nan}

sel_rows = []; aurc_boot = {}
for nm, P, tau in [("Uncalibrated", P_te_raw, tau_raw), ("Isotonic", P_te_iso, tau_iso)]:
    cov, risk, aurc = rc_curve(yte, P)
    bv = boot(lambda i: [rc_curve(yte[i], P[i])[2], *op_point(P[i], tau, yte[i]).values()], blocks_te, seed=5)
    aurc_boot[nm] = bv[:, 0]
    op = op_point(P, tau, yte)
    # risk at exactly 70% test coverage (oracle, for comparison only)
    k70 = int(0.70 * len(yte)); r70 = float(risk[k70 - 1]); tau70 = float(np.quantile(P.max(1), 0.30))
    sel_rows.append({"Model": nm, "tau (from cal. slice)": tau, "Test coverage": op["coverage"],
                     "Test coverage CI": ci(bv[:, 1]), "Test risk": op["risk"], "Test risk CI": ci(bv[:, 2]),
                     "AURC": aurc, "AURC CI": ci(bv[:, 0]), "Risk @ 70% test coverage (oracle)": r70, "Oracle tau (70% test coverage)": tau70,
                     "Full-coverage error": float((P.argmax(1) != yte).mean())})
d_aurc = aurc_boot["Isotonic"] - aurc_boot["Uncalibrated"]
R["aurc_diff"] = {"point": sel_rows[1]["AURC"] - sel_rows[0]["AURC"], "ci": ci(d_aurc)}
t_sel = pd.DataFrame(sel_rows)
t_sel_pretty = pd.DataFrame({
    "Model": t_sel.Model, "τ (cal. slice, 70%)": t_sel["tau (from cal. slice)"].round(8),
    "Coverage (2024)": [fmt_ci(r["Test coverage"], *r["Test coverage CI"]) for _, r in t_sel.iterrows()],
    "Risk (2024)": [fmt_ci(r["Test risk"], *r["Test risk CI"], 4) for _, r in t_sel.iterrows()],
    "AURC": [fmt_ci(r["AURC"], *r["AURC CI"], 4) for _, r in t_sel.iterrows()],
    "Risk @70% (oracle τ)": t_sel["Risk @ 70% test coverage (oracle)"].round(8),
    "Oracle τ for 70%": t_sel["Oracle tau (70% test coverage)"].round(8),
    "Error @100%": t_sel["Full-coverage error"].round(8)})
savetab(t_sel_pretty, "T3_selective_prediction.csv"); R["table3"] = t_sel_pretty.to_dict(orient="records")

# class-wise risk and deferral at the operating point
cw = []
for nm, P, tau in [("Uncalibrated", P_te_raw, tau_raw), ("Isotonic", P_te_iso, tau_iso)]:
    keep = P.max(1) >= tau; p = P.argmax(1)
    for k, c in enumerate(CLASSES):
        t_ = yte == k; pr = p == k
        cw.append({"Model": nm, "Class": c, "n true": int(t_.sum()),
                   "Deferral rate (true class)": float((~keep[t_]).mean()),
                   "Accepted n (true class)": int((keep & t_).sum()),
                   "Error among accepted (true class)": float((p[keep & t_] != k).mean()) if (keep & t_).any() else np.nan,
                   "Accepted n (pred class)": int((keep & pr).sum()),
                   "Error among accepted (pred class)": float((yte[keep & pr] != k).mean()) if (keep & pr).any() else np.nan})
t_cw = pd.DataFrame(cw); savetab(t_cw.round(8), "T4_classwise_selective.csv"); R["table4"] = t_cw.round(8).to_dict(orient="records")
# Sulphide recall among all true Sulphide (auto-accepted & correct) vs deferred
_keep = P_te_iso.max(1) >= tau_iso
R["deferred_share_correct_iso"] = float((P_te_iso.argmax(1)[~_keep] == yte[~_keep]).mean())
R["sulphide_auto_correct_iso"] = float(((P_te_iso.max(1) >= tau_iso) & (P_te_iso.argmax(1) == SUL) & (yte == SUL)).sum() / (yte == SUL).sum())

plt.figure(figsize=FS("f2", (4.6, 3.4)))
for nm, P, col, tau in [("Uncalibrated", P_te_raw, C_ORANGE, tau_raw), ("Isotonic", P_te_iso, C_BLUE, tau_iso)]:
    cov, risk, aurc = rc_curve(yte, P)
    m = cov >= 0.02
    plt.plot(cov[m], risk[m], color=col, label=f"{ {'Uncalibrated': 'M-base', 'Isotonic': 'M-cal'}[nm] } (AURC {aurc:.4f})")
    op = op_point(P, tau, yte); plt.plot(op["coverage"], op["risk"], "o", color=col, ms=7, mec="white", mew=1.5)
plt.axvline(0.70, color=C_GREY, ls=":", lw=1)
plt.xlabel("Coverage (fraction auto-decided)"); plt.ylabel("Risk (error among accepted)")
plt.legend(loc="upper left"); savefig("Fig2_risk_coverage.png")
print("selective done", round(time.time() - T0), "s")

# %% [markdown]
# ## 7. Monthly performance, Wilson intervals, label-shift monitoring (Reviewer 1, comments 3 & 5)
# Deployed model = M-cal (M-base + isotonic), no retraining.

# %%
def psi_cont(ref, cur, bins=10):
    ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 50: return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref); a = np.histogram(cur, edges)[0] / len(cur)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def psi_cat(e, a):
    e = np.clip(np.asarray(e, float), 1e-6, None); a = np.clip(np.asarray(a, float), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

# Label-shift estimators: BBSE (Lipton et al. 2018) using the calibration-slice confusion matrix,
# and EM prior adaptation (Saerens et al. 2002) on calibrated probabilities.
C_cal = confmat(cal_df.y.values, P_cal_iso.argmax(1)).astype(float)
C_joint = C_cal / C_cal.sum()                      # P(yhat=i, y=j)?  rows = true, cols = pred
pi_src = np.bincount(cal_df.y.values, minlength=3) / len(cal_df)

def bbse(pred_dist):
    # mu_hat = C^T w,  C[j,i] = P(y=j, yhat=i)  -> solve for importance weights w_j = q(y=j)/p(y=j)
    A = C_joint.T  # rows: yhat, cols: y
    w, *_ = np.linalg.lstsq(A, pred_dist, rcond=None); w = np.clip(w, 0, None)
    q = w * pi_src; return q / q.sum()

def em_prior(P, prior_src, iters=200):
    q = prior_src.copy()
    for _ in range(iters):
        Pq = P * (q / prior_src); Pq /= Pq.sum(1, keepdims=True); qn = Pq.mean(0)
        if np.abs(qn - q).max() < 1e-7: q = qn; break
        q = qn
    return q, Pq

ref23 = d23  # PSI reference: the 2023 training data of the deployed model
mon_rows = []; psi_rows = []
pred_te = P_te_iso.argmax(1)
for m in range(1, 13):
    idx = np.where(test.month.values == m)[0]; y_m = yte[idx]; p_m = pred_te[idx]; P_m = P_te_iso[idx]
    ns = int((y_m == SUL).sum()); ks = int(((y_m == SUL) & (p_m == SUL)).sum()); lo, hi = wilson(ks, ns)
    pdist = np.bincount(p_m, minlength=3) / len(idx); tdist = np.bincount(y_m, minlength=3) / len(idx)
    q_bbse = bbse(pdist); q_em, P_adj = em_prior(P_m, pi_src)
    row = {"Month": f"2024-{m:02d}", "n": len(idx), "Sulphide n": ns, "Macro-F1": macro_f1(y_m, p_m),
           "Balanced Acc": bal_acc(y_m, p_m), "Sulphide recall": ks / ns if ns else np.nan,
           "Sul recall Wilson lo": lo, "Sul recall Wilson hi": hi,
           "True Sulphide share": tdist[SUL], "Predicted Sulphide share": pdist[SUL],
           "BBSE Sulphide share": q_bbse[SUL], "EM Sulphide share": q_em[SUL],
           "True Oxide share": tdist[1], "Predicted Oxide share": pdist[1],
           "Pred-class PSI (vs cal. slice)": psi_cat(np.bincount(P_cal_iso.argmax(1), minlength=3) / len(cal_df), pdist),
           "Macro-F1 after EM prior adj.": macro_f1(y_m, P_adj.argmax(1)),
           "Sul recall after EM prior adj.": recall_k(y_m, P_adj.argmax(1))}
    cur = test.iloc[idx]
    fpsi = {c: psi_cont(ref23[c].values, cur[c].values) for c in NUM}
    row["Max feature PSI"] = max(fpsi.values()); row["Max-PSI feature"] = max(fpsi, key=fpsi.get)
    row["Missing-rate shift (max abs, pp)"] = float(100 * max(abs(cur[c].isna().mean() - ref23[c].isna().mean()) for c in NUM))
    mon_rows.append(row)
    for c in NUM: psi_rows.append({"Month": f"2024-{m:02d}", "Feature": c, "PSI": fpsi[c]})
monthly = pd.DataFrame(mon_rows); savetab(monthly.round(8), "T5_monthly_M-cal.csv")
psi_tab = pd.DataFrame(psi_rows).pivot(index="Month", columns="Feature", values="PSI").reset_index()
savetab(psi_tab.round(8), "T_psi_feature_month.csv")
R["monthly"] = monthly.round(8).to_dict(orient="records")
R["pooled_test_M-cal"] = {"Macro-F1": macro_f1(yte, pred_te), "Balanced Acc": bal_acc(yte, pred_te), "Sulphide recall": recall_k(yte, pred_te)}
# March confusion matrix
mar = test.month.values == 3
R["march_cm"] = confmat(yte[mar], pred_te[mar]).tolist()
R["full_cm_M-cal"] = confmat(yte, pred_te).tolist()
R["full_cm_M-full"] = confmat(yte, P_full.argmax(1)).tolist()
# is March worst after accounting for uncertainty?  bootstrap rank of March macro-F1
R["psi_corr_with_f1"] = float(np.corrcoef(monthly["Max feature PSI"], monthly["Macro-F1"])[0, 1])
R["predshare_corr_trueshare"] = float(np.corrcoef(monthly["Predicted Sulphide share"], monthly["True Sulphide share"])[0, 1])

# EM prior adaptation pooled over 2024 (deployable: needs no labels)
q_em_all, P_adj_all = em_prior(P_te_iso, pi_src)
R["em_pooled"] = {"q_sulphide": float(q_em_all[SUL]), "Macro-F1": macro_f1(yte, P_adj_all.argmax(1)),
                  "Balanced Acc": bal_acc(yte, P_adj_all.argmax(1)), "Sulphide recall": recall_k(yte, P_adj_all.argmax(1)),
                  "Sul NLL": nll_bin((yte == SUL).astype(float), P_adj_all[:, SUL]),
                  "Sul ECE@10": ece_bin((yte == SUL).astype(float), P_adj_all[:, SUL])}

fig, axes = plt.subplots(3, 1, figsize=FS("f3", (6.4, 7.0)), sharex=True)
x = np.arange(12); lab = [f"{m:02d}" for m in range(1, 13)]
ax = axes[0]; ax.plot(x, monthly["Macro-F1"], "o-", color=C_BLUE); ax.set_ylabel("Macro-F1"); ax.set_title("(a) Monthly Macro-F1, M-cal, no retraining")
ax = axes[1]
ax.errorbar(x, monthly["Sulphide recall"], yerr=[monthly["Sulphide recall"] - monthly["Sul recall Wilson lo"],
            monthly["Sul recall Wilson hi"] - monthly["Sulphide recall"]], fmt="o-", color=C_ORANGE, capsize=3)
for i, n in enumerate(monthly["Sulphide n"]): ax.annotate(f"n={n}", (x[i], 0.02), ha="center", fontsize=SM, color="#52514e")
ax.set_ylabel("Sulphide recall\n(95% Wilson)"); ax.set_ylim(0, 1.05); ax.set_title("(b) Monthly Sulphide recall")
ax = axes[2]
ax.plot(x, 100 * monthly["True Sulphide share"], "o-", color="#0b0b0b", label="True (needs labels)")
ax.plot(x, 100 * monthly["Predicted Sulphide share"], "s-", color=C_BLUE, label="Predicted (label-free)")
ax.plot(x, 100 * monthly["EM Sulphide share"], "^-", color=C_AQUA, label="EM estimate (label-free)")
ax.axhline(100 * pi_src[SUL], color=C_GREY, ls="--", lw=1, label="Calibration slice (Oct–Dec 2023)")
ax.set_ylabel("Sulphide share (%)"); ax.set_title("(c) Label-shift monitor"); ax.legend(fontsize=SM, ncol=2)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlabel("Month of 2024")
savefig("Fig3_monthly_drift_labelshift.png")

plt.figure(figsize=FS("f4", (6.4, 3.0)))
for i, c in enumerate(NUM):
    plt.plot(x, psi_tab[c], "o-", lw=1.4, ms=4, label=c, color=[C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAG, C_VIOLET][i])
plt.axhline(0.10, color=C_GREY, ls=":", lw=1); plt.axhline(0.25, color=C_GREY, ls="--", lw=1)
plt.xticks(x, lab); plt.xlabel("Month of 2024"); plt.ylabel("PSI vs 2023 (10 quantile bins)")
plt.legend(ncol=3, fontsize=SM); savefig("Fig4_feature_psi.png")
print("monthly done", round(time.time() - T0), "s")

# %% [markdown]
# ## 8. Calendar-feature ablation and pit hold-out (spatio-temporal)

# %%
abl = []
FEATS_NOTIME = NUM + MISS + CAT
for nm, feats in [("All features (M-cal)", FEATS), ("Without calendar features", FEATS_NOTIME),
                  ("Without missingness indicators", NUM + TIME + CAT)]:
    dep = Deployed(d23, feats=feats); P = dep.proba(test); p = P.argmax(1)
    abl.append({"Variant": nm, "Macro-F1": macro_f1(yte, p), "Balanced Acc": bal_acc(yte, p),
                "Sulphide recall": recall_k(yte, p), "March Macro-F1": macro_f1(yte[mar], p[mar]),
                "March Sulphide recall": recall_k(yte[mar], p[mar]),
                "Sul NLL": nll_bin((yte == SUL).astype(float), P[:, SUL])})
t_abl = pd.DataFrame(abl); savetab(t_abl.round(8), "T_ablation.csv"); R["ablation"] = t_abl.round(8).to_dict(orient="records")

# Fe sanity: treat physically impossible Fe (>72.4 wt%, above pure magnetite) as missing
ore_fe = ore.copy(); bad = ore_fe.fe_pct > 72.4
ore_fe.loc[bad, "fe_pct"] = np.nan; ore_fe["fe_pct_is_missing"] = ore_fe.fe_pct.isna().astype(int)
dep_fe = Deployed(ore_fe[ore_fe.year == 2023]); te_fe = ore_fe[ore_fe.year == 2024].reset_index(drop=True)
p_fe = dep_fe.predict(te_fe)
R["fe_sensitivity"] = {"Macro-F1": macro_f1(yte, p_fe), "Balanced Acc": bal_acc(yte, p_fe), "Sulphide recall": recall_k(yte, p_fe),
                       "n_set_missing": int(bad.sum())}

pit_rows = []
for hp in ["PitA", "PitB", "PitC"]:
    for proto, tr, te in [("Random-time (original: all years pooled)", ore[ore.pit != hp], ore[ore.pit == hp]),
                          ("Spatio-temporal (train 2023 other pits → test 2024 held-out pit)",
                           ore[(ore.pit != hp) & (ore.year == 2023)], ore[(ore.pit == hp) & (ore.year == 2024)])]:
        m_ = fit_lgbm(tr); p = m_.predict_proba(te[FEATS]).argmax(1); yy = te.y.values
        pit_rows.append({"Protocol": proto, "Held-out pit": hp, "n test": len(te), "Macro-F1": macro_f1(yy, p),
                         "Balanced Acc": bal_acc(yy, p), "Sulphide recall": recall_k(yy, p)})
t_pit = pd.DataFrame(pit_rows); savetab(t_pit.round(8), "T_pit_holdout.csv"); R["pit"] = t_pit.round(8).to_dict(orient="records")
print("ablation/pit done", round(time.time() - T0), "s")

# %% [markdown]
# ## 9. Retraining policies with label delay (Reviewer 1, comment 4)
# Assumptions: labels (assays + logged ore type) for month t become available at the end of month t+Δ.
# Primary Δ = 1 month; sensitivity Δ = 0.  Every retrain uses the identical recipe (Deployed):
# base on all labelled data except the last 3 months, isotonic on the last 3 months.
# Decisions for month m are taken at the start of month m using (i) labelled data with month index ≤ m−1−Δ and
# (ii) unlabelled features / predictions of month m−1.  Each model is evaluated only on months strictly after
# its training cut-off (asserted below).

# %%
def run_policy(kind, theta=None, delta=1):
    model = Deployed(d23); log = []; preds = np.empty(len(test), int); probs = np.empty((len(test), 3))
    for m in range(1, 13):
        mi = 12 + m - 1  # global month index of 2024-m
        avail = ore[ore.mi <= mi - 1 - delta]
        prev = ore[ore.mi == mi - 1]
        trig, sig = False, np.nan
        if kind == "P1":
            trig = True
        elif kind == "P2":
            sig = max(psi_cont(model.ref[c].values, prev[c].values) for c in NUM); trig = sig > theta
        elif kind == "P3":
            pd_prev = np.bincount(model.predict(prev), minlength=3) / len(prev)
            sig = psi_cat(model.cal_pred_dist, pd_prev); trig = sig > theta
        retrained = False
        if trig and avail.mi.max() > model.train_cutoff_mi:
            model = Deployed(avail); retrained = True
        idx = np.where(test.mi.values == mi)[0]
        assert model.train_cutoff_mi < mi, "model evaluated on a month inside its training window"
        probs[idx] = model.proba(test.iloc[idx]); preds[idx] = probs[idx].argmax(1)
        log.append({"month": m, "signal": sig, "triggered": bool(trig), "retrained": retrained,
                    "train_cutoff": f"{2023 + model.train_cutoff_mi // 12}-{model.train_cutoff_mi % 12 + 1:02d}"})
    return preds, probs, pd.DataFrame(log)

policies = [("P0 No retraining", "P0", None)] + [("P1 Monthly retraining", "P1", None)] + \
           [(f"P2 Feature-PSI trigger θ={t:.2f}", "P2", t) for t in [0.10, 0.15, 0.20, 0.25]] + \
           [(f"P3 Predicted-class PSI trigger θ={t:.2f}", "P3", t) for t in [0.05, 0.10, 0.25]]
import pickle
CACHE = os.path.join(OUT, "policy_cache.pkl")
if os.environ.get("USE_POLICY_CACHE") == "1" and os.path.exists(CACHE):
    pol_out = pickle.load(open(CACHE, "rb"))
else:
    pol_out = {}
    for delta in [1, 0]:
        for nm, k, t in policies:
            pol_out[(delta, nm)] = run_policy(k, t, delta)
        print("policies delta", delta, "done", round(time.time() - T0), "s")
    pickle.dump(pol_out, open(CACHE, "wb"))

def pooled_stats(p, y=None):
    y = yte if y is None else y
    return np.array([macro_f1(y, p), bal_acc(y, p), recall_k(y, p)])

def month_mean(p):
    return np.array([np.mean([macro_f1(yte[test.month.values == m], p[test.month.values == m]) for m in range(1, 13)])])

EQ_MARGIN = 0.01
prow = []
for delta in [1, 0]:
    p0 = pol_out[(delta, "P0 No retraining")][0]
    for nm, k, t in policies:
        p, P, log = pol_out[(delta, nm)]
        st = pooled_stats(p)
        bv = boot(lambda i: pooled_stats(p[i], yte[i]) - pooled_stats(p0[i], yte[i]), blocks_te, n=N_BOOT, seed=11)
        lo95, hi95 = np.quantile(bv, [0.025, 0.975], axis=0); lo90, hi90 = np.quantile(bv, [0.05, 0.95], axis=0)
        d = st - pooled_stats(p0)
        prow.append({"Label delay Δ (months)": delta, "Policy": nm, "Retrains (of 12)": int(log.retrained.sum()),
                     "Retrain months": ",".join(str(x) for x in log[log.retrained].month) or "—",
                     "Macro-F1": st[0], "Balanced Acc": st[1], "Sulphide recall": st[2],
                     "Mean monthly Macro-F1": month_mean(p)[0],
                     "ΔMacro-F1 vs P0": d[0], "ΔMacro-F1 95% CI": [float(lo95[0]), float(hi95[0])],
                     "ΔMacro-F1 90% CI": [float(lo90[0]), float(hi90[0])],
                     "ΔBalAcc vs P0": d[1], "ΔBalAcc 95% CI": [float(lo95[1]), float(hi95[1])],
                     "ΔSul recall vs P0": d[2], "ΔSul recall 95% CI": [float(lo95[2]), float(hi95[2])],
                     f"Equivalent to P0 (TOST, ±{EQ_MARGIN})": bool(lo90[0] > -EQ_MARGIN and hi90[0] < EQ_MARGIN) if nm != "P0 No retraining" else None})
t_pol = pd.DataFrame(prow)
t_pol_pretty = t_pol[["Label delay Δ (months)", "Policy", "Retrains (of 12)", "Retrain months"]].copy()
t_pol_pretty["Macro-F1"] = t_pol["Macro-F1"].round(8); t_pol_pretty["Balanced Acc"] = t_pol["Balanced Acc"].round(8)
t_pol_pretty["Sulphide recall"] = t_pol["Sulphide recall"].round(8)
t_pol_pretty["ΔMacro-F1 vs P0 [95% CI]"] = [fmt_ci(r["ΔMacro-F1 vs P0"], *r["ΔMacro-F1 95% CI"], 4) for _, r in t_pol.iterrows()]
t_pol_pretty["ΔBalAcc vs P0 [95% CI]"] = [fmt_ci(r["ΔBalAcc vs P0"], *r["ΔBalAcc 95% CI"], 4) for _, r in t_pol.iterrows()]
t_pol_pretty["ΔSul recall vs P0 [95% CI]"] = [fmt_ci(r["ΔSul recall vs P0"], *r["ΔSul recall 95% CI"], 4) for _, r in t_pol.iterrows()]
t_pol_pretty[f"Equivalent (TOST ±{EQ_MARGIN})"] = t_pol[f"Equivalent to P0 (TOST, ±{EQ_MARGIN})"]
savetab(t_pol_pretty, "T6_policies.csv"); R["table6"] = t_pol_pretty.to_dict(orient="records")
logs = []
for (delta, nm), (_, _, log) in pol_out.items():
    l = log.copy(); l.insert(0, "Policy", nm); l.insert(0, "Δ", delta); logs.append(l)
savetab(pd.concat(logs).round(8), "T_policy_logs.csv")

plt.figure(figsize=FS("f5", (6.4, 3.2)))
for nm, col, mk in [("P0 No retraining", "#0b0b0b", "o"), ("P1 Monthly retraining", C_BLUE, "s"),
                    ("P2 Feature-PSI trigger θ=0.15", C_ORANGE, "^"), ("P3 Predicted-class PSI trigger θ=0.10", C_AQUA, "D")]:
    p = pol_out[(1, nm)][0]
    plt.plot(x, [macro_f1(yte[test.month.values == m], p[test.month.values == m]) for m in range(1, 13)],
             marker=mk, color=col, lw=1.6, label=nm)
plt.xticks(x, lab); plt.xlabel("Month of 2024"); plt.ylabel("Macro-F1")
plt.legend(fontsize=SM, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
savefig("Fig5_policies_monthly.png")
print("policies done", round(time.time() - T0), "s")

# %% [markdown]
# ## 10. Tonnage forecasting — leakage-free one-day-ahead design (Reviewer 2)
# Forecast origin: end of day t−1.  Only information available by then is used: tonnage up to t−1 and the
# ore-sample summaries of day t−1 and earlier.  No backward filling.  Walk-forward: expanding window,
# 30-day test blocks starting 2023-06-15; evaluation restricted to days with ore data (≤ 2024-12-31).

# %%
ton = ton.sort_values("date").reset_index(drop=True)
od = ore.copy()
share = pd.crosstab(od.date, od.ore_type_label, normalize="index"); share.columns = [f"share_{c.lower()}" for c in share.columns]
means = od.groupby("date")[NUM].mean()
pits = pd.crosstab(od.date, od.pit, normalize="index"); pits.columns = [f"pitshare_{c.lower()}" for c in pits.columns]
daily = share.join(means).join(pits)
ORE_D = list(daily.columns)
df = ton.set_index("date").join(daily, how="left")
df[ORE_D] = df[ORE_D].ffill()           # forward fill only (past information)
y = df.tonnage
F = pd.DataFrame(index=df.index)
F["lag1"] = y.shift(1); F["lag7"] = y.shift(7); F["lag14"] = y.shift(14)
F["roll7"] = y.shift(1).rolling(7).mean(); F["roll30"] = y.shift(1).rolling(30).mean()
F["dow"] = df.index.dayofweek; F["month"] = df.index.month
F["t"] = np.arange(len(F))
for c in ORE_D: F[f"{c}_lag1"] = df[c].shift(1)           # yesterday's ore summary (known at origin)
for c in ORE_D: F[f"{c}_same_day"] = df[c]                 # SAME-DAY ore summary: NOT known at origin (leaky)
F["y"] = y
F = F[(F.index <= "2024-12-31")].dropna(subset=["lag14", "roll30"])
LAGS = ["lag1", "lag7", "lag14", "roll7", "roll30", "dow", "month"]
ORE_L1 = [f"{c}_lag1" for c in ORE_D]; ORE_SD = [f"{c}_same_day" for c in ORE_D]

def lgbm_reg():
    return LGBMRegressor(n_estimators=900, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                         subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=SEED, n_jobs=2, verbose=-1)

def ets_forecast(train_y, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    m = ExponentialSmoothing(train_y.values, trend="add", damped_trend=True, seasonal=None).fit(optimized=True)
    return m

cuts = pd.date_range("2023-06-15", "2024-12-31", freq="30D")
models_fc = {
    "Naïve (t−1)": lambda tr, te: te.lag1.values,
    "Seasonal naïve (t−7)": lambda tr, te: te.lag7.values,
    "Rolling mean 7 d": lambda tr, te: te.roll7.values,
    "Rolling mean 30 d": lambda tr, te: te.roll30.values,
    "Ridge (lags + calendar)": lambda tr, te: Pipeline([("s", StandardScaler()), ("r", Ridge(1.0))]).fit(
        tr[LAGS + ["t"]], tr.y).predict(te[LAGS + ["t"]]),
    "LightGBM (lags + calendar)": lambda tr, te: lgbm_reg().fit(tr[LAGS], tr.y).predict(te[LAGS]),
    "LightGBM (lags + calendar + ore, t−1)": lambda tr, te: lgbm_reg().fit(tr[LAGS + ORE_L1], tr.y).predict(te[LAGS + ORE_L1]),
    "LightGBM residual-to-roll30 (lags + ore, t−1)": lambda tr, te: te.roll30.values + lgbm_reg().fit(
        tr[LAGS + ORE_L1], tr.y - tr.roll30).predict(te[LAGS + ORE_L1]),
    "[Leaky] LightGBM (lags + same-day ore)": lambda tr, te: lgbm_reg().fit(tr[LAGS + ORE_SD], tr.y).predict(te[LAGS + ORE_SD]),
    "[Original, leaky] LightGBM same-day ore, no lags": lambda tr, te: lgbm_reg().fit(tr[ORE_SD], tr.y).predict(te[ORE_SD]),
}

def ets_wrap(tr, te):
    # one-step-ahead ETS forecasts within the test block, parameters fitted on the training block,
    # states updated with each newly observed value (no refit) -> strictly ex-ante
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(tr.y.values, trend="add", damped_trend=True).fit(optimized=True)
    a, b, phi = fit.params["smoothing_level"], fit.params["smoothing_trend"], fit.params["damping_trend"]
    l, tr_ = fit.level[-1], fit.trend[-1]; out = []
    for v in te.y.values:
        f = l + phi * tr_; out.append(f)
        l_new = a * v + (1 - a) * (l + phi * tr_); tr_ = b * (l_new - l) + (1 - b) * phi * tr_; l = l_new
    return np.array(out)
models_fc = dict(list(models_fc.items())[:4] + [("ETS (damped additive trend)", ets_wrap)] + list(models_fc.items())[4:])

fc_rows = []; fc_pred = {k: pd.Series(np.nan, index=F.index) for k in models_fc}
for cut in cuts:
    tr = F[F.index < cut]; te = F[(F.index >= cut) & (F.index < cut + pd.Timedelta(days=30))]
    if len(te) < 7 or len(tr) < 120: continue
    for k, fn in models_fc.items():
        pr = fn(tr, te); fc_pred[k].loc[te.index] = pr
        e = te.y.values - pr
        fc_rows.append({"cut": str(cut.date()), "Model": k, "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
                        "MAPE%": 100 * np.mean(np.abs(e) / te.y.values), "n": len(te)})
fcw = pd.DataFrame(fc_rows); savetab(fcw.round(3), "T_forecast_windows.csv")
nwin = fcw.cut.nunique()
summ = fcw.groupby("Model", sort=False)[["MAE", "RMSE", "MAPE%"]].mean().reset_index()
# paired bootstrap over windows of MAPE difference vs best ex-ante simple baseline (Rolling mean 30 d) and vs Naïve
piv = fcw.pivot(index="cut", columns="Model", values="MAPE%")
def paired(a, b, n=5000, seed=7):
    d = (piv[a] - piv[b]).values; rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()); return d.mean(), np.quantile(bs, [0.025, 0.975]), max(p, 1 / n)
# Diebold–Mariano (squared error, h=1, HAC with lag 6) on daily errors pooled across windows
ev = F.loc[fc_pred["Naïve (t−1)"].dropna().index]
def dm(a, b):
    ea = ev.y - fc_pred[a].loc[ev.index]; eb = ev.y - fc_pred[b].loc[ev.index]
    d = (np.abs(ea) - np.abs(eb)).values; n = len(d); dbar = d.mean(); L = 6
    g = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(L + 1)]
    var = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    stat = dbar / np.sqrt(var / n); return float(stat), float(2 * (1 - norm.cdf(abs(stat))))
ref_b = "Rolling mean 30 d"
summ["ΔMAPE vs rolling-30 [95% CI]"] = ""; summ["p (paired bootstrap, windows)"] = np.nan; summ["DM stat (|e|) vs rolling-30"] = np.nan; summ["DM p"] = np.nan
for i, r in summ.iterrows():
    if r.Model == ref_b: continue
    dmean, (lo, hi), p = paired(r.Model, ref_b)
    summ.loc[i, "ΔMAPE vs rolling-30 [95% CI]"] = fmt_ci(dmean, lo, hi, 2); summ.loc[i, "p (paired bootstrap, windows)"] = p
    s, pdm = dm(r.Model, ref_b); summ.loc[i, "DM stat (|e|) vs rolling-30"] = s; summ.loc[i, "DM p"] = pdm
summ.insert(1, "Ex-ante (no leakage)", [not k.startswith("[") for k in summ.Model])
savetab(summ.round(8), "T7_forecasting.csv"); R["table7"] = summ.round(8).to_dict(orient="records"); R["n_windows"] = int(nwin)
R["forecast_eval_days"] = int(len(ev)); R["forecast_eval_start"] = str(ev.index.min().date()); R["forecast_eval_end"] = str(ev.index.max().date())
# reproduce original lag-vs-nolag comparison under the corrected design (for the response letter)
dm_, ci_, p_ = paired("LightGBM (lags + calendar + ore, t−1)", "Naïve (t−1)")
R["lgbm_vs_naive"] = {"diff": dm_, "ci": list(ci_), "p": p_}

# importance: gain, split, SHAP, permutation on the final hold-out (last 60 days)
cut_imp = F.index.max() - pd.Timedelta(days=60)
tr, te = F[F.index < cut_imp], F[F.index >= cut_imp]
FI = LAGS + ORE_L1
mreg = lgbm_reg().fit(tr[FI], tr.y)
gain = mreg.booster_.feature_importance("gain"); split = mreg.booster_.feature_importance("split")
import shap
sv = shap.TreeExplainer(mreg).shap_values(te[FI]); shap_mean = np.abs(sv).mean(0)
from sklearn.inspection import permutation_importance
pim = permutation_importance(mreg, te[FI], te.y, n_repeats=30, random_state=SEED, scoring="neg_mean_absolute_error")
imp = pd.DataFrame({"Feature": FI, "Split count": split, "Gain (% of total)": 100 * gain / gain.sum(),
                    "Mean |SHAP| (t)": shap_mean, "Permutation ΔMAE (t)": pim.importances_mean,
                    "Permutation ΔMAE sd": pim.importances_std}).sort_values("Gain (% of total)", ascending=False)
savetab(imp.round(6), "T8_forecast_importance.csv"); R["importance"] = imp.round(6).to_dict(orient="records")
# Correlation of same-day and lagged Fe with tonnage (physical plausibility check)
R["corr_tonnage"] = {c: float(F["y"].corr(F[c])) for c in ["fe_pct_same_day", "fe_pct_lag1", "share_sulphide_same_day",
                                                           "share_oxide_same_day", "lag1", "roll30", "t"]}
# original split-count importance reproduced with same-day features (to explain the Fe artefact)
morig = lgbm_reg().fit(tr[LAGS + ORE_SD], tr.y)
R["orig_style_importance_top5_split"] = pd.Series(morig.booster_.feature_importance("split"), index=LAGS + ORE_SD).sort_values(ascending=False).head(5).to_dict()
# null check: same model on shuffled Fe (split counts for a pure-noise column)
trn = tr.copy(); trn["noise"] = np.random.default_rng(0).normal(size=len(trn))
mnoise = lgbm_reg().fit(trn[LAGS + ORE_L1 + ["noise"]], trn.y)
R["noise_split_rank"] = int((pd.Series(mnoise.booster_.feature_importance("split"), index=LAGS + ORE_L1 + ["noise"]).rank(ascending=False))["noise"])
R["noise_split_count"] = int(pd.Series(mnoise.booster_.feature_importance("split"), index=LAGS + ORE_L1 + ["noise"])["noise"])

plt.figure(figsize=FS("f6", (6.4, 3.2)))
show = [("Naïve (t−1)", "#0b0b0b", "o"), ("Rolling mean 30 d", C_GREY, "s"), ("ETS (damped additive trend)", C_AQUA, "D"),
        ("LightGBM (lags + calendar + ore, t−1)", C_BLUE, "^"), ("[Leaky] LightGBM (lags + same-day ore)", C_ORANGE, "v")]
for k, col, mk in show:
    s = piv[k]; plt.plot(range(len(s)), s.values, marker=mk, color=col, lw=1.4, ms=4, label=k)
plt.xticks(range(len(piv)), [c[:7] for c in piv.index], rotation=60, fontsize=SM)
plt.ylabel("MAPE (%) per 30-day window"); plt.xlabel("Window start"); plt.legend(fontsize=SM)
savefig("Fig6_forecast_windows.png")

top = imp.head(10).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=FS("f7", (7.2, 3.4)), sharey=True)
axes[0].barh(top.Feature, top["Gain (% of total)"], color=C_BLUE, height=0.6); axes[0].set_xlabel("Gain importance (% of total)")
axes[1].barh(top.Feature, top["Permutation ΔMAE (t)"], xerr=top["Permutation ΔMAE sd"], color=C_ORANGE, height=0.6)
axes[1].axvline(0, color=C_GREY, lw=1); axes[1].set_xlabel("Permutation ΔMAE on hold-out (t)")
savefig("Fig7_forecast_importance.png")
print("forecasting done", round(time.time() - T0), "s")

# %% [markdown]
# ## 11. Illustrative cost analysis grounded in the observed confusions (Reviewer 2)
# Relative cost units (Oxide→Oxide = 0).  Values are ILLUSTRATIVE and must be replaced by a matrix agreed with
# processing staff; sensitivity to the key entry (true Sulphide predicted Oxide/Mixed) is reported.

# %%
def cost_matrix(c_sul_miss=10.0):
    # rows = true (Mixed, Oxide, Sulphide), cols = predicted
    return np.array([[0, 2, 3],
                     [2, 0, 4],
                     [c_sul_miss * 0.6, c_sul_miss, 0]], float)
review_cost = 1.0
cost_rows = []
for csm in [5.0, 10.0, 20.0]:
    Cm = cost_matrix(csm)
    for nm, P, tau in [("M-cal, no deferral", P_te_iso, 0.0), ("M-cal + deferral at τ", P_te_iso, tau_iso),
                       ("M-cal + cost-sensitive (Bayes) decision", P_te_iso, None)]:
        if tau is None:
            p = (P @ Cm).argmin(1); keep = np.ones(len(p), bool)
        else:
            p = P.argmax(1); keep = P.max(1) >= tau
        cost = np.where(keep, Cm[yte, p], review_cost)
        cost_rows.append({"Sulphide-miss cost": csm, "Decision rule": nm, "Mean cost / sample": cost.mean(),
                          "Deferred %": 100 * (~keep).mean(), "Sulphide recall (auto)": recall_k(yte, np.where(keep, p, -1))})
t_cost = pd.DataFrame(cost_rows); savetab(t_cost.round(8), "T_cost_illustrative.csv"); R["cost"] = t_cost.round(8).to_dict(orient="records")

# %% [markdown]
# ## 12. Provenance (Reviewer 1, comment 3): which model produced which table/figure

# %%
prov = pd.DataFrame([
    ("M-full", "LightGBM CFG, fitted on all 2023 (n=%d); uncalibrated" % len(d23), "Table 3; same configuration used for pit hold-out (Table 8)"),
    ("M-base", "LightGBM CFG, fitted on Jan–Sep 2023 (n=%d); uncalibrated" % len(fit_df), "Tables 3–5; Figs. 1–2"),
    ("M-cal", "M-base (frozen) + one-vs-rest isotonic fitted on Oct–Dec 2023 (n=%d), renormalised" % len(cal_df),
     "Tables 3–7; Figs. 1–4; initial model of every policy (Table 9)"),
    ("M-ts", "M-base (frozen) + temperature scaling fitted on Oct–Dec 2023 (T=%.3f)" % tmp_.t, "extra check, not reported in the paper"),
    ("Policy models", "Deployed recipe re-applied to all labelled data available at each decision time", "Table 9; Fig. 5; Supplementary Table S2"),
    ("Forecasters", "LightGBMRegressor(900, 0.05, 31 leaves, min_child_samples=30)", "Table 10; Fig. 6"),
], columns=["Model ID", "Definition", "Used in"])
savetab(prov, "T_provenance.csv")

R["runtime_s"] = round(time.time() - T0)
import lightgbm, sklearn, statsmodels
R["versions"] = {"python": __import__("sys").version.split()[0], "lightgbm": lightgbm.__version__,
                 "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__, "shap": shap.__version__}
def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return _clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(_clean(R), open(os.path.join(OUT, "results.json"), "w"), indent=1, ensure_ascii=False, default=str)
print("ALL DONE in", R["runtime_s"], "s")
