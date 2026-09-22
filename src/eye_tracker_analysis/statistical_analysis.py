import os
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.sandbox.stats.multicomp import multipletests

warnings.filterwarnings("ignore")

# CONFIGURATION
DATA_ROOT   = Path(r"C:\Users\User\Desktop\Python\MRProcessing\data")
RESULTS_90  = DATA_ROOT / "high_90hz" / "RESULTS_90hz" / "summary" / "summary_all_results.csv"
RESULTS_60  = DATA_ROOT / "mid_60hz"  / "RESULTS_60hz" / "summary" / "summary_all_results.csv"
OUTPUT_DIR  = DATA_ROOT / "RESULTS" / "statistical_comparison_FINAL"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task Groupings
TASKS_2D = ["PlaneClose", "PlaneMiddle", "PlaneFar"]
TASKS_3D = ["ObjectCone(1)", "ObjectCone(3)", "UD0411(83)", "PlaneWSW"]

METRICS = {
    "ang_err_mean":          "Angular Error (deg)",
    "precision_rms_3d_mean": "Precision RMS 3D (deg)",
    "precision_rms_2d_mean": "Precision RMS 2D (deg)",
    "dist_2d_mean":          "2D Distance Error",
    "pos_vel_mean":          "Head Pos Velocity (m/s)",
    "ang_vel_mean":          "Head Ang Velocity (deg/s)",
}

# HELPERS
def sig_stars(p):
    if pd.isna(p): return "N/A"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    var1, var2 = g1.var(), g2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

def rank_biserial_r(z_stat, n):
    return abs(z_stat) / np.sqrt(n) if n > 0 else 0.0

def get_better_info(metric_col, med1, med2, name1, name2, p_corr):
    """
    Determines the winner, absolute gain, and relative % improvement.
    For eye tracking accuracy/precision and head movement, LOWER is better.
    """
    if pd.isna(p_corr) or p_corr >= 0.05:
        return "No sig. difference (p>=0.05)", np.nan, np.nan
    
    lower_is_better = ['ang_err', 'precision', 'dist_2d', 'pos_vel', 'ang_vel']
    is_lower_better = any(m in metric_col for m in lower_is_better)
    
    if is_lower_better:
        if med1 < med2:
            winner, loser_med, abs_gain = name1, med2, med2 - med1
        elif med2 < med1:
            winner, loser_med, abs_gain = name2, med1, med1 - med2
        else:
            return "Equal", 0, 0
    else:
        if med1 > med2:
            winner, loser_med, abs_gain = name1, med2, med1 - med2
        elif med2 > med1:
            winner, loser_med, abs_gain = name2, med1, med2 - med1
        else:
            return "Equal", 0, 0
            
    rel_imp = (abs_gain / loser_med) * 100 if loser_med != 0 else np.nan
    return winner, round(abs_gain, 4), round(rel_imp, 2)

def load_and_tag(path, freq):
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    df["frequency"] = freq
    df["task_type"] = df["analysis_folder"].apply(
        lambda f: "2D" if f in TASKS_2D else ("3D" if f in TASKS_3D else "Other"))
    return df

def merge_data(d90, d60):
    combined = pd.concat([d90, d60], ignore_index=True)
    combined = combined[combined["task_type"].isin(["2D", "3D"])].copy()
    ids = ["session", "analysis_folder", "frequency", "task_type"]
    vals = [c for c in combined.columns if c not in ids + ["type"]]
    return combined.groupby(ids, dropna=False).agg({c: "first" for c in vals}).reset_index()

# STATISTICAL ENGINE
def run_analysis(df):
    rows = []
    metrics_list = list(METRICS.keys())
    
    print("Running Literature-Backed Non-Parametric Tests...")
    
    for col, label in METRICS.items():
        if col not in df.columns: continue
        
        # 1. BETWEEN FREQUENCY (90Hz vs 60Hz)
        g90 = df.loc[df["frequency"] == "90Hz", col].dropna()
        g60 = df.loc[df["frequency"] == "60Hz", col].dropna()
        
        if len(g90) >= 3 and len(g60) >= 3:
            u_stat, p_val = sp_stats.mannwhitneyu(g90, g60, alternative='two-sided')
            n1, n2 = len(g90), len(g60)
            mu = n1 * n2 / 2
            sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (u_stat - mu) / sigma if sigma > 0 else 0
            
            p_corr = min(p_val * len(metrics_list), 1.0)
            med90, med60 = g90.median(), g60.median()
            winner, abs_gain, rel_imp = get_better_info(col, med90, med60, "90Hz", "60Hz", p_corr)
            
            rows.append({
                "Comparison": "90Hz vs 60Hz (All Tasks)",
                "Test": "Mann-Whitney U",
                "Metric": label,
                "N_90Hz": n1, "N_60Hz": n2,
                "Median_90Hz": round(med90, 4), "Median_60Hz": round(med60, 4),
                "Winner": winner,
                "Absolute_Gain": abs_gain,
                "Relative_Improvement_%": rel_imp,
                "Statistic (U)": u_stat, "Z_score": round(z, 4),
                "P_raw": p_val, "P_Bonferroni": p_corr,
                "Significance": sig_stars(p_corr),
                "Cohen_d": round(cohens_d(g90, g60), 4),
                "Effect_Size_r": round(rank_biserial_r(z, n1 + n2), 4),
                "Reason": "Awasthi (2023) - Non-parametric independent comparison"
            })

        # 2. WITHIN FREQUENCY: 2D vs 3D
        for freq in ["90Hz", "60Hz"]:
            df_f = df[df["frequency"] == freq]
            sess_2d = df_f[df_f["task_type"] == "2D"].groupby("session")[col].mean().dropna()
            sess_3d = df_f[df_f["task_type"] == "3D"].groupby("session")[col].mean().dropna()
            
            common_sessions = sess_2d.index.intersection(sess_3d.index)
            if len(common_sessions) >= 6:
                v2d = sess_2d.loc[common_sessions]
                v3d = sess_3d.loc[common_sessions]
                
                try:
                    stat, p_val = sp_stats.wilcoxon(v2d, v3d)
                    n = len(v2d)
                    T = stat
                    mu = n * (n + 1) / 4
                    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                    z = (T - mu) / sigma if sigma > 0 else 0
                    
                    p_corr = min(p_val * len(metrics_list), 1.0)
                    med2d, med3d = v2d.median(), v3d.median()
                    winner, abs_gain, rel_imp = get_better_info(col, med2d, med3d, "2D Tasks", "3D Tasks", p_corr)
                    
                    rows.append({
                        "Comparison": f"2D vs 3D Tasks ({freq})",
                        "Test": "Wilcoxon Signed-Rank",
                        "Metric": label,
                        "N_Pairs": n,
                        "Median_2D": round(med2d, 4), "Median_3D": round(med3d, 4),
                        "Winner": winner,
                        "Absolute_Gain": abs_gain,
                        "Relative_Improvement_%": rel_imp,
                        "Statistic (W)": stat, "Z_score": round(z, 4),
                        "P_raw": p_val, "P_Bonferroni": p_corr,
                        "Significance": sig_stars(p_corr),
                        "Cohen_d": round(cohens_d(v2d, v3d), 4),
                        "Effect_Size_r": round(rank_biserial_r(z, n), 4),
                        "Reason": "Kapp (2021) - Non-parametric paired comparison"
                    })
                except Exception: pass

        # 3. WITHIN FREQUENCY: Distance Post-Hoc
        for freq in ["90Hz", "60Hz"]:
            df_f = df[df["frequency"] == freq]
            planes = ["PlaneClose", "PlaneMiddle", "PlaneFar"]
            
            pivot = df_f[df_f["analysis_folder"].isin(planes)].pivot(
                index="session", columns="analysis_folder", values=col
            ).dropna()
            
            if len(pivot) >= 6 and all(p in pivot.columns for p in planes):
                data_arrays = [pivot[p].values for p in planes]
                
                try:
                    f_stat, p_fried = sp_stats.friedmanchisquare(*data_arrays)
                    p_fried_corr = min(p_fried * len(metrics_list), 1.0)
                    
                    rows.append({
                        "Comparison": f"Distance Effect ({freq})",
                        "Test": "Friedman Test (Omnibus)",
                        "Metric": label,
                        "N_Sessions": len(pivot),
                        "Statistic (Chi2)": round(f_stat, 4),
                        "P_raw": p_fried, "P_Bonferroni": p_fried_corr,
                        "Significance": sig_stars(p_fried_corr),
                        "Winner": "N/A (Omnibus)",
                        "Absolute_Gain": np.nan, "Relative_Improvement_%": np.nan,
                        "Reason": "Kapp (2021) - Non-parametric repeated measures ANOVA"
                    })
                    
                    if p_fried < 0.05:
                        pairs = list(combinations(planes, 2))
                        p_vals_posthoc = []
                        for p1, p2 in pairs:
                            _, p_w = sp_stats.wilcoxon(pivot[p1], pivot[p2])
                            p_vals_posthoc.append(p_w)
                            
                        reject, p_corr_posthoc, _, _ = multipletests(p_vals_posthoc, alpha=0.05, method='bonferroni')
                        
                        for i, (p1, p2) in enumerate(pairs):
                            med1, med2 = pivot[p1].median(), pivot[p2].median()
                            winner, abs_gain, rel_imp = get_better_info(col, med1, med2, p1, p2, p_corr_posthoc[i])
                            
                            rows.append({
                                "Comparison": f"Post-Hoc: {p1} vs {p2} ({freq})",
                                "Test": "Wilcoxon (Post-Hoc)",
                                "Metric": label,
                                "N_Pairs": len(pivot),
                                f"Median_{p1}": round(med1, 4), f"Median_{p2}": round(med2, 4),
                                "Winner": winner,
                                "Absolute_Gain": abs_gain,
                                "Relative_Improvement_%": rel_imp,
                                "P_raw": p_vals_posthoc[i],
                                "P_Bonferroni": p_corr_posthoc[i],
                                "Significance": sig_stars(p_corr_posthoc[i]),
                                "Cohen_d": round(cohens_d(pivot[p1], pivot[p2]), 4),
                                "Reason": "Kapp (2021) - Pairwise distance comparison"
                            })
                except Exception: pass

    return pd.DataFrame(rows)

# MAIN EXECUTION
def main():
    df_90 = load_and_tag(RESULTS_90, "90Hz")
    df_60 = load_and_tag(RESULTS_60, "60Hz")
    
    if df_90.empty and df_60.empty:
        print("ERROR: No data found."); sys.exit(1)
        
    df = merge_data(df_90, df_60)
    print(f"Loaded {len(df)} valid session-task combinations.")
    
    df_results = run_analysis(df)
    
    if df_results.empty:
        print("No statistical results generated.")
        return
        
    # Define exact column order for Excel readability
    cols = [
        "Comparison", "Test", "Metric", "Significance", 
        "Winner", "Absolute_Gain", "Relative_Improvement_%",
        "P_Bonferroni", "P_raw", "Cohen_d", "Effect_Size_r", "Reason"
    ]
    
    # Add dynamic median columns
    median_cols = [c for c in df_results.columns if c.startswith("Median_") or c.startswith("N_")]
    final_cols = cols[:4] + median_cols + cols[4:]
    
    final_cols = [c for c in final_cols if c in df_results.columns]
    df_out = df_results[final_cols].sort_values(by=["Metric", "Comparison"])
    
    out_path = OUTPUT_DIR / "Literature_Backed_Stats_Results.csv"
    df_out.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS! Saved clean, Excel-ready table to:\n{out_path}")
    print("New columns added: 'Winner', 'Absolute_Gain', 'Relative_Improvement_%'")

if __name__ == "__main__":
    main()