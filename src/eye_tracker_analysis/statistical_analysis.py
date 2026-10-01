# # # import os
# # # import sys
# # # import warnings
# # # from pathlib import Path
# # # from itertools import combinations

# # # import numpy as np
# # # import pandas as pd
# # # from scipy import stats as sp_stats
# # # from statsmodels.sandbox.stats.multicomp import multipletests

# # # warnings.filterwarnings("ignore")

# # # # CONFIGURATION
# # # DATA_ROOT   = Path(r"C:\Users\User\Desktop\Python\MRProcessing\data")
# # # RESULTS_90  = DATA_ROOT / "high_90hz" / "RESULTS_90hz" / "summary" / "summary_all_results.csv"
# # # RESULTS_60  = DATA_ROOT / "mid_60hz"  / "RESULTS_60hz" / "summary" / "summary_all_results.csv"
# # # OUTPUT_DIR  = DATA_ROOT / "RESULTS" / "statistical_comparison_FINAL"
# # # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # # # Task Groupings
# # # TASKS_2D = ["PlaneClose", "PlaneMiddle", "PlaneFar"]
# # # TASKS_3D = ["ObjectCone(1)", "ObjectCone(3)", "UD0411(83)", "PlaneWSW"]

# # # METRICS = {
# # #     "ang_err_mean":          "Angular Error (deg)",
# # #     "precision_rms_3d_mean": "Precision RMS 3D (deg)",
# # #     "precision_rms_2d_mean": "Precision RMS 2D (deg)",
# # #     "dist_2d_mean":          "2D Distance Error",
# # #     "pos_vel_mean":          "Head Pos Velocity (m/s)",
# # #     "ang_vel_mean":          "Head Ang Velocity (deg/s)",
# # # }

# # # # HELPERS
# # # def sig_stars(p):
# # #     if pd.isna(p): return "N/A"
# # #     if p < 0.001: return "***"
# # #     if p < 0.01:  return "**"
# # #     if p < 0.05:  return "*"
# # #     return "ns"

# # # def cohens_d(g1, g2):
# # #     n1, n2 = len(g1), len(g2)
# # #     if n1 < 2 or n2 < 2: return np.nan
# # #     var1, var2 = g1.var(), g2.var()
# # #     pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
# # #     return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

# # # def rank_biserial_r(z_stat, n):
# # #     return abs(z_stat) / np.sqrt(n) if n > 0 else 0.0

# # # def get_better_info(metric_col, med1, med2, name1, name2, p_corr):
# # #     """
# # #     Determines the winner, absolute gain, and relative % improvement.
# # #     For eye tracking accuracy/precision and head movement, LOWER is better.
# # #     """
# # #     if pd.isna(p_corr) or p_corr >= 0.05:
# # #         return "No sig. difference (p>=0.05)", np.nan, np.nan
    
# # #     lower_is_better = ['ang_err', 'precision', 'dist_2d', 'pos_vel', 'ang_vel']
# # #     is_lower_better = any(m in metric_col for m in lower_is_better)
    
# # #     if is_lower_better:
# # #         if med1 < med2:
# # #             winner, loser_med, abs_gain = name1, med2, med2 - med1
# # #         elif med2 < med1:
# # #             winner, loser_med, abs_gain = name2, med1, med1 - med2
# # #         else:
# # #             return "Equal", 0, 0
# # #     else:
# # #         if med1 > med2:
# # #             winner, loser_med, abs_gain = name1, med2, med1 - med2
# # #         elif med2 > med1:
# # #             winner, loser_med, abs_gain = name2, med1, med2 - med1
# # #         else:
# # #             return "Equal", 0, 0
            
# # #     rel_imp = (abs_gain / loser_med) * 100 if loser_med != 0 else np.nan
# # #     return winner, round(abs_gain, 4), round(rel_imp, 2)

# # # def load_and_tag(path, freq):
# # #     if not path.exists(): return pd.DataFrame()
# # #     df = pd.read_csv(path)
# # #     df["frequency"] = freq
# # #     df["task_type"] = df["analysis_folder"].apply(
# # #         lambda f: "2D" if f in TASKS_2D else ("3D" if f in TASKS_3D else "Other"))
# # #     return df

# # # def merge_data(d90, d60):
# # #     combined = pd.concat([d90, d60], ignore_index=True)
# # #     combined = combined[combined["task_type"].isin(["2D", "3D"])].copy()
# # #     ids = ["session", "analysis_folder", "frequency", "task_type"]
# # #     vals = [c for c in combined.columns if c not in ids + ["type"]]
# # #     return combined.groupby(ids, dropna=False).agg({c: "first" for c in vals}).reset_index()

# # # # STATISTICAL ENGINE
# # # def run_analysis(df):
# # #     rows = []
# # #     metrics_list = list(METRICS.keys())
    
# # #     print("Running Literature-Backed Non-Parametric Tests...")
    
# # #     for col, label in METRICS.items():
# # #         if col not in df.columns: continue
        
# # #         # 1. BETWEEN FREQUENCY (90Hz vs 60Hz)
# # #         g90 = df.loc[df["frequency"] == "90Hz", col].dropna()
# # #         g60 = df.loc[df["frequency"] == "60Hz", col].dropna()
        
# # #         if len(g90) >= 3 and len(g60) >= 3:
# # #             u_stat, p_val = sp_stats.mannwhitneyu(g90, g60, alternative='two-sided')
# # #             n1, n2 = len(g90), len(g60)
# # #             mu = n1 * n2 / 2
# # #             sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
# # #             z = (u_stat - mu) / sigma if sigma > 0 else 0
            
# # #             p_corr = min(p_val * len(metrics_list), 1.0)
# # #             med90, med60 = g90.median(), g60.median()
# # #             winner, abs_gain, rel_imp = get_better_info(col, med90, med60, "90Hz", "60Hz", p_corr)
            
# # #             rows.append({
# # #                 "Comparison": "90Hz vs 60Hz (All Tasks)",
# # #                 "Test": "Mann-Whitney U",
# # #                 "Metric": label,
# # #                 "N_90Hz": n1, "N_60Hz": n2,
# # #                 "Median_90Hz": round(med90, 4), "Median_60Hz": round(med60, 4),
# # #                 "Winner": winner,
# # #                 "Absolute_Gain": abs_gain,
# # #                 "Relative_Improvement_%": rel_imp,
# # #                 "Statistic (U)": u_stat, "Z_score": round(z, 4),
# # #                 "P_raw": p_val, "P_Bonferroni": p_corr,
# # #                 "Significance": sig_stars(p_corr),
# # #                 "Cohen_d": round(cohens_d(g90, g60), 4),
# # #                 "Effect_Size_r": round(rank_biserial_r(z, n1 + n2), 4),
# # #                 "Reason": "Awasthi (2023) - Non-parametric independent comparison"
# # #             })

# # #         # 2. WITHIN FREQUENCY: 2D vs 3D
# # #         for freq in ["90Hz", "60Hz"]:
# # #             df_f = df[df["frequency"] == freq]
# # #             sess_2d = df_f[df_f["task_type"] == "2D"].groupby("session")[col].mean().dropna()
# # #             sess_3d = df_f[df_f["task_type"] == "3D"].groupby("session")[col].mean().dropna()
            
# # #             common_sessions = sess_2d.index.intersection(sess_3d.index)
# # #             if len(common_sessions) >= 6:
# # #                 v2d = sess_2d.loc[common_sessions]
# # #                 v3d = sess_3d.loc[common_sessions]
                
# # #                 try:
# # #                     stat, p_val = sp_stats.wilcoxon(v2d, v3d)
# # #                     n = len(v2d)
# # #                     T = stat
# # #                     mu = n * (n + 1) / 4
# # #                     sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
# # #                     z = (T - mu) / sigma if sigma > 0 else 0
                    
# # #                     p_corr = min(p_val * len(metrics_list), 1.0)
# # #                     med2d, med3d = v2d.median(), v3d.median()
# # #                     winner, abs_gain, rel_imp = get_better_info(col, med2d, med3d, "2D Tasks", "3D Tasks", p_corr)
                    
# # #                     rows.append({
# # #                         "Comparison": f"2D vs 3D Tasks ({freq})",
# # #                         "Test": "Wilcoxon Signed-Rank",
# # #                         "Metric": label,
# # #                         "N_Pairs": n,
# # #                         "Median_2D": round(med2d, 4), "Median_3D": round(med3d, 4),
# # #                         "Winner": winner,
# # #                         "Absolute_Gain": abs_gain,
# # #                         "Relative_Improvement_%": rel_imp,
# # #                         "Statistic (W)": stat, "Z_score": round(z, 4),
# # #                         "P_raw": p_val, "P_Bonferroni": p_corr,
# # #                         "Significance": sig_stars(p_corr),
# # #                         "Cohen_d": round(cohens_d(v2d, v3d), 4),
# # #                         "Effect_Size_r": round(rank_biserial_r(z, n), 4),
# # #                         "Reason": "Kapp (2021) - Non-parametric paired comparison"
# # #                     })
# # #                 except Exception: pass

# # #         # 3. WITHIN FREQUENCY: Distance Post-Hoc
# # #         for freq in ["90Hz", "60Hz"]:
# # #             df_f = df[df["frequency"] == freq]
# # #             planes = ["PlaneClose", "PlaneMiddle", "PlaneFar"]
            
# # #             pivot = df_f[df_f["analysis_folder"].isin(planes)].pivot(
# # #                 index="session", columns="analysis_folder", values=col
# # #             ).dropna()
            
# # #             if len(pivot) >= 6 and all(p in pivot.columns for p in planes):
# # #                 data_arrays = [pivot[p].values for p in planes]
                
# # #                 try:
# # #                     f_stat, p_fried = sp_stats.friedmanchisquare(*data_arrays)
# # #                     p_fried_corr = min(p_fried * len(metrics_list), 1.0)
                    
# # #                     rows.append({
# # #                         "Comparison": f"Distance Effect ({freq})",
# # #                         "Test": "Friedman Test (Omnibus)",
# # #                         "Metric": label,
# # #                         "N_Sessions": len(pivot),
# # #                         "Statistic (Chi2)": round(f_stat, 4),
# # #                         "P_raw": p_fried, "P_Bonferroni": p_fried_corr,
# # #                         "Significance": sig_stars(p_fried_corr),
# # #                         "Winner": "N/A (Omnibus)",
# # #                         "Absolute_Gain": np.nan, "Relative_Improvement_%": np.nan,
# # #                         "Reason": "Kapp (2021) - Non-parametric repeated measures ANOVA"
# # #                     })
                    
# # #                     if p_fried < 0.05:
# # #                         pairs = list(combinations(planes, 2))
# # #                         p_vals_posthoc = []
# # #                         for p1, p2 in pairs:
# # #                             _, p_w = sp_stats.wilcoxon(pivot[p1], pivot[p2])
# # #                             p_vals_posthoc.append(p_w)
                            
# # #                         reject, p_corr_posthoc, _, _ = multipletests(p_vals_posthoc, alpha=0.05, method='bonferroni')
                        
# # #                         for i, (p1, p2) in enumerate(pairs):
# # #                             med1, med2 = pivot[p1].median(), pivot[p2].median()
# # #                             winner, abs_gain, rel_imp = get_better_info(col, med1, med2, p1, p2, p_corr_posthoc[i])
                            
# # #                             rows.append({
# # #                                 "Comparison": f"Post-Hoc: {p1} vs {p2} ({freq})",
# # #                                 "Test": "Wilcoxon (Post-Hoc)",
# # #                                 "Metric": label,
# # #                                 "N_Pairs": len(pivot),
# # #                                 f"Median_{p1}": round(med1, 4), f"Median_{p2}": round(med2, 4),
# # #                                 "Winner": winner,
# # #                                 "Absolute_Gain": abs_gain,
# # #                                 "Relative_Improvement_%": rel_imp,
# # #                                 "P_raw": p_vals_posthoc[i],
# # #                                 "P_Bonferroni": p_corr_posthoc[i],
# # #                                 "Significance": sig_stars(p_corr_posthoc[i]),
# # #                                 "Cohen_d": round(cohens_d(pivot[p1], pivot[p2]), 4),
# # #                                 "Reason": "Kapp (2021) - Pairwise distance comparison"
# # #                             })
# # #                 except Exception: pass

# # #     return pd.DataFrame(rows)

# # # # MAIN EXECUTION
# # # def main():
# # #     df_90 = load_and_tag(RESULTS_90, "90Hz")
# # #     df_60 = load_and_tag(RESULTS_60, "60Hz")
    
# # #     if df_90.empty and df_60.empty:
# # #         print("ERROR: No data found."); sys.exit(1)
        
# # #     df = merge_data(df_90, df_60)
# # #     print(f"Loaded {len(df)} valid session-task combinations.")
    
# # #     df_results = run_analysis(df)
    
# # #     if df_results.empty:
# # #         print("No statistical results generated.")
# # #         return
        
# # #     # Define exact column order for Excel readability
# # #     cols = [
# # #         "Comparison", "Test", "Metric", "Significance", 
# # #         "Winner", "Absolute_Gain", "Relative_Improvement_%",
# # #         "P_Bonferroni", "P_raw", "Cohen_d", "Effect_Size_r", "Reason"
# # #     ]
    
# # #     # Add dynamic median columns
# # #     median_cols = [c for c in df_results.columns if c.startswith("Median_") or c.startswith("N_")]
# # #     final_cols = cols[:4] + median_cols + cols[4:]
    
# # #     final_cols = [c for c in final_cols if c in df_results.columns]
# # #     df_out = df_results[final_cols].sort_values(by=["Metric", "Comparison"])
    
# # #     out_path = OUTPUT_DIR / "Literature_Backed_Stats_Results.csv"
# # #     df_out.to_csv(out_path, index=False)
    
# # #     print(f"\nSUCCESS! Saved clean, Excel-ready table to:\n{out_path}")
# # #     print("New columns added: 'Winner', 'Absolute_Gain', 'Relative_Improvement_%'")

# # # if __name__ == "__main__":
# # #     main()


# # import os, sys, warnings
# # from pathlib import Path
# # from itertools import combinations

# # import numpy as np
# # import pandas as pd
# # from scipy import stats as sp_stats
# # from statsmodels.sandbox.stats.multicomp import multipletests

# # warnings.filterwarnings("ignore")

# # # ────────────────────────────────────────────────────────────
# # # CONFIGURATION
# # # ────────────────────────────────────────────────────────────
# # DATA_ROOT   = Path(r"C:\Users\User\Desktop\Python\MRProcessing\data")
# # RESULTS_90  = DATA_ROOT / "high_90hz" / "RESULTS_90hz" / "summary" / "summary_all_results.csv"
# # RESULTS_60  = DATA_ROOT / "mid_60hz"  / "RESULTS_60hz" / "summary" / "summary_all_results.csv"
# # OUTPUT_DIR  = DATA_ROOT / "RESULTS" / "statistical_comparison_FINAL"
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # TASKS_2D = ["PlaneClose", "PlaneMiddle", "PlaneFar"]
# # TASKS_3D = ["ObjectCone(1)", "ObjectCone(3)", "UD0411(83)", "PlaneWSW"]
# # ALL_TASKS = TASKS_2D + TASKS_3D

# # # ── Metrics split by task scope ──
# # METRICS_ALL = {
# #     "ang_err_mean":          "Angular Error (deg)",
# #     "precision_rms_3d_mean": "Precision RMS 3D (deg)",
# #     "precision_rms_2d_mean": "Precision RMS 2D (deg)",
# #     "dist_2d_mean":          "2D Distance Error (m)",
# # }

# # METRICS_3D_ONLY = {
# #     "pos_vel_mean":          "Head Positional Velocity (m/s)",
# #     "ang_vel_mean":          "Head Angular Velocity (deg/s)",
# #     "eye_gaze_dist_mean":    "Viewing Distance (m)",
# # }

# # ALL_METRICS = {**METRICS_ALL, **METRICS_3D_ONLY}
# # ALPHA = 0.05

# # # ────────────────────────────────────────────────────────────
# # # HELPERS
# # # ────────────────────────────────────────────────────────────
# # def sig_stars(p):
# #     if pd.isna(p): return "N/A"
# #     if p < 0.001: return "***"
# #     if p < 0.01:  return "**"
# #     if p < 0.05:  return "*"
# #     return "ns"

# # def cohens_d(g1, g2):
# #     n1, n2 = len(g1), len(g2)
# #     if n1 < 2 or n2 < 2: return np.nan
# #     var1, var2 = g1.var(), g2.var()
# #     pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
# #     return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0

# # def rank_biserial_r(z, n):
# #     return abs(z) / np.sqrt(n) if n > 0 else 0.0

# # def get_better_info(metric_col, med1, med2, name1, name2, p_corr):
# #     if pd.isna(p_corr) or p_corr >= 0.05:
# #         return "No sig. difference (p>=0.05)", np.nan, np.nan
# #     lower_better = ['ang_err','precision','dist_2d','pos_vel','ang_vel','eye_gaze']
# #     is_lower = any(m in metric_col for m in lower_better)
# #     if is_lower:
# #         if med1 < med2:
# #             winner, loser_med, gain = name1, med2, med2 - med1
# #         elif med2 < med1:
# #             winner, loser_med, gain = name2, med1, med1 - med2
# #         else:
# #             return "Equal", 0, 0
# #     else:
# #         if med1 > med2:
# #             winner, loser_med, gain = name1, med2, med1 - med2
# #         elif med2 > med1:
# #             winner, loser_med, gain = name2, med1, med2 - med1
# #         else:
# #             return "Equal", 0, 0
# #     rel = (gain / loser_med) * 100 if loser_med != 0 else np.nan
# #     return winner, round(gain, 4), round(rel, 2)

# # def load_and_tag(path, freq):
# #     if not path.exists(): return pd.DataFrame()
# #     df = pd.read_csv(path)
# #     df["frequency"] = freq
# #     df["task_type"] = df["analysis_folder"].apply(
# #         lambda f: "2D" if f in TASKS_2D else ("3D" if f in TASKS_3D else "Other"))
# #     return df

# # def merge_data(d90, d60):
# #     combined = pd.concat([d90, d60], ignore_index=True)
# #     combined = combined[combined["task_type"].isin(["2D","3D"])].copy()
# #     ids  = ["session","analysis_folder","frequency","task_type"]
# #     vals = [c for c in combined.columns if c not in ids + ["type"]]
# #     return combined.groupby(ids, dropna=False).agg({c:"first" for c in vals}).reset_index()

# # # ────────────────────────────────────────────────────────────
# # # WILCOXON POST-HOC ENGINE (reusable)
# # # ────────────────────────────────────────────────────────────
# # def friedman_wilcoxon_posthoc(pivot, task_names, col, label, comparison_name,
# #                               freq, rows, n_metric_correction):
# #     """Run Friedman omnibus + Bonferroni-corrected Wilcoxon post-hoc."""
# #     data_arrays = [pivot[t].values for t in task_names]
# #     try:
# #         chi2, p_fried = sp_stats.friedmanchisquare(*data_arrays)
# #     except Exception:
# #         return
# #     p_fried_corr = min(p_fried * n_metric_correction, 1.0)

# #     medians = {t: pivot[t].median() for t in task_names}
# #     med_str = "; ".join([f"{t}: {medians[t]:.4f}" for t in task_names])

# #     rows.append({
# #         "Comparison": f"{comparison_name} ({freq})",
# #         "Test": "Friedman (Omnibus)",
# #         "Metric": label,
# #         "N_Sessions": len(pivot),
# #         "Tasks_Compared": ", ".join(task_names),
# #         "Medians": med_str,
# #         "Statistic_Chi2": round(chi2, 4),
# #         "P_raw": round(p_fried, 6),
# #         "P_Bonferroni": round(p_fried_corr, 6),
# #         "Significance": sig_stars(p_fried_corr),
# #         "Winner": "N/A (Omnibus)",
# #         "Absolute_Gain": np.nan,
# #         "Relative_Improvement_%": np.nan,
# #         "Reason": "Kapp (2021) – Friedman + Wilcoxon post-hoc"
# #     })

# #     if p_fried_corr >= 0.05:
# #         return

# #     pairs = list(combinations(task_names, 2))
# #     p_vals = []
# #     for t1, t2 in pairs:
# #         try:
# #             _, pw = sp_stats.wilcoxon(pivot[t1], pivot[t2])
# #         except Exception:
# #             pw = np.nan
# #         p_vals.append(pw)

# #     reject, p_corr, _, _ = multipletests(
# #         [p if not np.isnan(p) else 1.0 for p in p_vals],
# #         alpha=ALPHA, method='bonferroni')

# #     for i, (t1, t2) in enumerate(pairs):
# #         med1, med2 = pivot[t1].median(), pivot[t2].median()
# #         winner, gain, rel = get_better_info(col, med1, med2, t1, t2, p_corr[i])

# #         # Z-score for effect size r
# #         try:
# #             stat_w, _ = sp_stats.wilcoxon(pivot[t1], pivot[t2])
# #             n = len(pivot)
# #             mu = n*(n+1)/4
# #             sigma = np.sqrt(n*(n+1)*(2*n+1)/24)
# #             z = (stat_w - mu) / sigma if sigma > 0 else 0
# #         except Exception:
# #             z = np.nan

# #         rows.append({
# #             "Comparison": f"Post-Hoc: {t1} vs {t2} ({freq})",
# #             "Test": "Wilcoxon Signed-Rank (Post-Hoc)",
# #             "Metric": label,
# #             "N_Pairs": len(pivot),
# #             "Tasks_Compared": f"{t1} vs {t2}",
# #             f"Median_{t1}": round(med1, 4),
# #             f"Median_{t2}": round(med2, 4),
# #             "Winner": winner,
# #             "Absolute_Gain": gain,
# #             "Relative_Improvement_%": rel,
# #             "Z_score": round(z, 4) if not np.isnan(z) else np.nan,
# #             "P_raw": round(p_vals[i], 6) if not np.isnan(p_vals[i]) else np.nan,
# #             "P_Bonferroni": round(p_corr[i], 6),
# #             "Significance": sig_stars(p_corr[i]),
# #             "Cohen_d": round(cohens_d(pivot[t1], pivot[t2]), 4),
# #             "Effect_Size_r": round(rank_biserial_r(z, len(pivot)), 4)
# #                          if not np.isnan(z) else np.nan,
# #             "Reason": "Kapp (2021) – Bonferroni-corrected pairwise"
# #         })

# # # ────────────────────────────────────────────────────────────
# # # MAIN ANALYSIS
# # # ────────────────────────────────────────────────────────────
# # def run_analysis(df):
# #     rows = []
# #     n_all  = len(ALL_METRICS)
# #     n_3d   = len(METRICS_3D_ONLY)

# #     # ================================================================
# #     # 1. BETWEEN FREQUENCY: 90 Hz vs 60 Hz  (Mann-Whitney U)
# #     # ================================================================
# #     print("=" * 70)
# #     print("1. BETWEEN FREQUENCY (90 Hz vs 60 Hz)")
# #     print("=" * 70)

# #     for col, label in ALL_METRICS.items():
# #         if col not in df.columns: continue
# #         g90 = df.loc[df["frequency"]=="90Hz", col].dropna()
# #         g60 = df.loc[df["frequency"]=="60Hz", col].dropna()
# #         if len(g90) < 3 or len(g60) < 3: continue

# #         u, p = sp_stats.mannwhitneyu(g90, g60, alternative='two-sided')
# #         n1, n2 = len(g90), len(g60)
# #         mu = n1*n2/2
# #         sigma = np.sqrt(n1*n2*(n1+n2+1)/12)
# #         z = (u - mu)/sigma if sigma > 0 else 0
# #         pc = min(p * n_all, 1.0)
# #         med90, med60 = g90.median(), g60.median()
# #         winner, gain, rel = get_better_info(col, med90, med60, "90Hz", "60Hz", pc)

# #         rows.append({
# #             "Comparison": "90Hz vs 60Hz (All Tasks)",
# #             "Test": "Mann-Whitney U",
# #             "Metric": label,
# #             "N_90Hz": n1, "N_60Hz": n2,
# #             "Median_90Hz": round(med90,4), "Median_60Hz": round(med60,4),
# #             "Winner": winner,
# #             "Absolute_Gain": gain,
# #             "Relative_Improvement_%": rel,
# #             "Statistic_U": u, "Z_score": round(z,4),
# #             "P_raw": p, "P_Bonferroni": pc,
# #             "Significance": sig_stars(pc),
# #             "Cohen_d": round(cohens_d(g90, g60), 4),
# #             "Effect_Size_r": round(rank_biserial_r(z, n1+n2), 4),
# #             "Reason": "Awasthi (2023) – Bonferroni-corrected MWU"
# #         })
# #         print(f"  {label:40s} p_c={pc:.4f} {sig_stars(pc)}  Winner: {winner}")

# #     # ================================================================
# #     # 2. WITHIN FREQUENCY: ALL TASKS – Median comparison
# #     #    (Friedman + Wilcoxon post-hoc)
# #     #    Metrics: METRICS_ALL
# #     # ================================================================
# #     print("\n" + "=" * 70)
# #     print("2. WITHIN FREQUENCY – ALL TASKS (Median Comparison)")
# #     print("=" * 70)

# #     for freq in ["90Hz", "60Hz"]:
# #         df_f = df[df["frequency"] == freq]
# #         if df_f.empty: continue

# #         for col, label in METRICS_ALL.items():
# #             if col not in df_f.columns: continue

# #             pivot = df_f[df_f["analysis_folder"].isin(ALL_TASKS)].pivot(
# #                 index="session", columns="analysis_folder", values=col
# #             ).dropna()

# #             valid_tasks = [t for t in ALL_TASKS if t in pivot.columns]
# #             if len(valid_tasks) < 3 or len(pivot) < 6:
# #                 continue

# #             print(f"\n  [{freq}] {label} across {len(valid_tasks)} tasks:")
# #             friedman_wilcoxon_posthoc(
# #                 pivot, valid_tasks, col, label,
# #                 "All Tasks Median Comparison", freq, rows, len(METRICS_ALL))

# #     # ================================================================
# #     # 3. WITHIN FREQUENCY: 2D vs 3D  (Wilcoxon)
# #     # ================================================================
# #     print("\n" + "=" * 70)
# #     print("3. WITHIN FREQUENCY – 2D vs 3D")
# #     print("=" * 70)

# #     for freq in ["90Hz", "60Hz"]:
# #         df_f = df[df["frequency"] == freq]
# #         if df_f.empty: continue

# #         for col, label in METRICS_ALL.items():
# #             if col not in df_f.columns: continue
# #             s2d = df_f[df_f["task_type"]=="2D"].groupby("session")[col].mean().dropna()
# #             s3d = df_f[df_f["task_type"]=="3D"].groupby("session")[col].mean().dropna()
# #             common = s2d.index.intersection(s3d.index)
# #             if len(common) < 6: continue
# #             v2d, v3d = s2d.loc[common], s3d.loc[common]

# #             try:
# #                 stat, p = sp_stats.wilcoxon(v2d, v3d)
# #             except Exception:
# #                 continue
# #             n = len(common)
# #             mu = n*(n+1)/4
# #             sigma = np.sqrt(n*(n+1)*(2*n+1)/24)
# #             z = (stat - mu)/sigma if sigma > 0 else 0
# #             pc = min(p * len(METRICS_ALL), 1.0)
# #             med2d, med3d = v2d.median(), v3d.median()
# #             winner, gain, rel = get_better_info(col, med2d, med3d,
# #                                                 "2D Tasks", "3D Tasks", pc)

# #             rows.append({
# #                 "Comparison": f"2D vs 3D Tasks ({freq})",
# #                 "Test": "Wilcoxon Signed-Rank",
# #                 "Metric": label,
# #                 "N_Pairs": n,
# #                 "Median_2D": round(med2d,4), "Median_3D": round(med3d,4),
# #                 "Winner": winner,
# #                 "Absolute_Gain": gain,
# #                 "Relative_Improvement_%": rel,
# #                 "Statistic_W": stat, "Z_score": round(z,4),
# #                 "P_raw": p, "P_Bonferroni": pc,
# #                 "Significance": sig_stars(pc),
# #                 "Cohen_d": round(cohens_d(v2d, v3d), 4),
# #                 "Effect_Size_r": round(rank_biserial_r(z, n), 4),
# #                 "Reason": "Kapp (2021) – Paired non-parametric"
# #             })
# #             print(f"  [{freq}] {label:35s} p_c={pc:.4f} {sig_stars(pc)}  Winner: {winner}")

# #     # ================================================================
# #     # 4. WITHIN FREQUENCY: 3D TASKS ONLY
# #     #    (Velocity + Viewing Distance)
# #     #    Friedman + Wilcoxon post-hoc
# #     # ================================================================
# #     print("\n" + "=" * 70)
# #     print("4. WITHIN FREQUENCY – 3D TASKS ONLY "
# #           "(Velocity + Viewing Distance)")
# #     print("=" * 70)

# #     for freq in ["90Hz", "60Hz"]:
# #         df_f = df[df["frequency"] == freq]
# #         if df_f.empty: continue

# #         for col, label in METRICS_3D_ONLY.items():
# #             if col not in df_f.columns: continue

# #             pivot = df_f[df_f["analysis_folder"].isin(TASKS_3D)].pivot(
# #                 index="session", columns="analysis_folder", values=col
# #             ).dropna()

# #             valid_tasks = [t for t in TASKS_3D if t in pivot.columns]
# #             if len(valid_tasks) < 3 or len(pivot) < 6:
# #                 continue

# #             print(f"\n  [{freq}] {label} across {len(valid_tasks)} 3D tasks:")
# #             friedman_wilcoxon_posthoc(
# #                 pivot, valid_tasks, col, label,
# #                 "3D Tasks Only", freq, rows, len(METRICS_3D_ONLY))

# #     # ================================================================
# #     # 5. WITHIN FREQUENCY: DISTANCE EFFECT (2D planes)
# #     #    Friedman + Wilcoxon post-hoc
# #     # ================================================================
# #     print("\n" + "=" * 70)
# #     print("5. WITHIN FREQUENCY – DISTANCE EFFECT (PlaneClose/Mid/Far)")
# #     print("=" * 70)

# #     for freq in ["90Hz", "60Hz"]:
# #         df_f = df[df["frequency"] == freq]
# #         if df_f.empty: continue

# #         for col, label in METRICS_ALL.items():
# #             if col not in df_f.columns: continue

# #             pivot = df_f[df_f["analysis_folder"].isin(TASKS_2D)].pivot(
# #                 index="session", columns="analysis_folder", values=col
# #             ).dropna()

# #             valid = [t for t in TASKS_2D if t in pivot.columns]
# #             if len(valid) < 3 or len(pivot) < 6:
# #                 continue

# #             print(f"\n  [{freq}] {label} across planes:")
# #             friedman_wilcoxon_posthoc(
# #                 pivot, valid, col, label,
# #                 "Distance Effect", freq, rows, len(METRICS_ALL))

# #     return pd.DataFrame(rows)

# # # ────────────────────────────────────────────────────────────
# # # MAIN
# # # ────────────────────────────────────────────────────────────
# # def main():
# #     df_90 = load_and_tag(RESULTS_90, "90Hz")
# #     df_60 = load_and_tag(RESULTS_60, "60Hz")

# #     if df_90.empty and df_60.empty:
# #         print("ERROR: No data found."); sys.exit(1)

# #     df = merge_data(df_90, df_60)
# #     print(f"Loaded {len(df)} valid session-task rows.\n")

# #     df_res = run_analysis(df)

# #     if df_res.empty:
# #         print("No results generated."); return

# #     # ── Column ordering ──
# #     fixed = ["Comparison","Test","Metric","Significance",
# #              "Winner","Absolute_Gain","Relative_Improvement_%",
# #              "P_Bonferroni","P_raw",
# #              "Cohen_d","Effect_Size_r","Reason"]
# #     median_cols = [c for c in df_res.columns
# #                    if c.startswith("Median_") or c.startswith("N_")
# #                    or c == "Tasks_Compared" or c == "Medians"
# #                    or c.startswith("Statistic") or c == "Z_score"]
# #     final_cols = fixed[:5] + median_cols + fixed[5:]
# #     final_cols = [c for c in final_cols if c in df_res.columns]
# #     # remove duplicates preserving order
# #     seen = set(); ordered = []
# #     for c in final_cols:
# #         if c not in seen:
# #             seen.add(c); ordered.append(c)

# #     df_out = df_res[ordered].sort_values(by=["Metric","Comparison"])
# #     out_path = OUTPUT_DIR / "Literature_Backed_Stats_Results.csv"
# #     df_out.to_csv(out_path, index=False)

# #     sig = df_out["Significance"].isin(["*","**","***"])
# #     print(f"\n{'='*70}")
# #     print(f"Saved: {out_path}")
# #     print(f"Total rows: {len(df_out)}")
# #     print(f"Significant (corrected): {sig.sum()} / {len(df_out)}")
# #     print(f"{'='*70}")

# # if __name__ == "__main__":
# #     main()




# import os, sys, warnings
# from pathlib import Path
# from itertools import combinations

# import numpy as np
# import pandas as pd
# from scipy import stats as sp_stats
# from statsmodels.sandbox.stats.multicomp import multipletests

# warnings.filterwarnings("ignore")

# # ────────────────────────────────────────────────────────────
# # CONFIGURATION
# # ────────────────────────────────────────────────────────────
# DATA_ROOT   = Path(r"C:\Users\User\Desktop\Python\MRProcessing\data")
# RESULTS_90  = DATA_ROOT / "high_90hz" / "RESULTS" / "summary" / "summary_all_results.csv"
# RESULTS_60  = DATA_ROOT / "mid_60hz"  / "RESULTS" / "summary" / "summary_all_results.csv"
# OUTPUT_DIR  = DATA_ROOT / "RESULTS" / "statistical_comparison_FINAL"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TASKS_2D  = ["PlaneClose", "PlaneMiddle", "PlaneFar"]
# TASKS_3D  = ["ObjectCone(1)", "ObjectCone(3)", "UD0411(83)", "PlaneWSW"]
# ALL_TASKS = TASKS_2D + TASKS_3D

# # ════════════════════════════════════════════════════════════
# # METRICS BY HYPOTHESIS
# # ════════════════════════════════════════════════════════════

# # H1 & H2: Spatial metrics
# # Kapp (ARETT) §4.4: "mean gaze sample" → use MEAN
# # precision_rms_2d_mean REMOVED (not used)
# METRICS_SPATIAL = {
#     "ang_err_mean":          "Angular Error (deg)",
#     "precision_rms_3d_mean": "Precision RMS 3D (deg)",
#     "dist_2d_mean":          "2D Distance Error (m)",
# }

# # H3: Viewing behaviour (3D tasks only) → use MEAN
# METRICS_3D_BEHAVIOUR = {
#     "pos_vel_mean":          "Head Positional Velocity (m/s)",
#     "ang_vel_mean":          "Head Angular Velocity (deg/s)",
#     "eye_gaze_dist_mean":    "Viewing Distance (m)",
# }

# # Temporal: ONLY ONE metric (ISI = 1/freq, adding both is redundant)
# # Aziz §3.4: median is robust to dropped-sample outliers
# METRICS_TEMPORAL = {
#     "isi_median":            "ISI Median (ms)",
# }

# ALPHA = 0.05


# # ────────────────────────────────────────────────────────────
# # HELPERS
# # ────────────────────────────────────────────────────────────
# def sig_stars(p):
#     if pd.isna(p): return "N/A"
#     if p < 0.001: return "***"
#     if p < 0.01:  return "**"
#     if p < 0.05:  return "*"
#     return "ns"


# def cohens_d(g1, g2):
#     n1, n2 = len(g1), len(g2)
#     if n1 < 2 or n2 < 2:
#         return np.nan
#     var1, var2 = g1.var(), g2.var()
#     pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
#     return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0


# def rank_biserial_r(z, n):
#     return abs(z) / np.sqrt(n) if n > 0 else 0.0


# def summarize_group(series, use_mean=True):
#     """Return correct central tendency for a group."""
#     return series.mean() if use_mean else series.median()


# def stat_prefix(use_mean):
#     """Column name prefix."""
#     return "Mean" if use_mean else "Median"


# def get_better_info(metric_col, val1, val2, name1, name2, p_corr):
#     """
#     Determines winner, absolute gain, relative % improvement.
#     LOWER is better: error, precision, distance, velocity, ISI
#     HIGHER is better: frequency (not used)
#     """
#     if pd.isna(p_corr) or p_corr >= 0.05:
#         return "No sig. difference (p>=0.05)", np.nan, np.nan

#     lower_better = [
#         'ang_err', 'precision', 'dist_2d',
#         'pos_vel', 'ang_vel', 'eye_gaze',
#         'isi'
#     ]
#     is_lower = any(m in metric_col for m in lower_better)

#     if is_lower:
#         if val1 < val2:
#             winner, loser_val, gain = name1, val2, val2 - val1
#         elif val2 < val1:
#             winner, loser_val, gain = name2, val1, val1 - val2
#         else:
#             return "Equal", 0, 0
#     else:
#         if val1 > val2:
#             winner, loser_val, gain = name1, val2, val1 - val2
#         elif val2 > val1:
#             winner, loser_val, gain = name2, val1, val2 - val1
#         else:
#             return "Equal", 0, 0

#     rel = (gain / loser_val) * 100 if loser_val != 0 else np.nan
#     return winner, round(gain, 4), round(rel, 2)


# def load_and_tag(path, freq):
#     if not path.exists():
#         return pd.DataFrame()
#     df = pd.read_csv(path)
#     df["frequency"] = freq
#     df["task_type"] = df["analysis_folder"].apply(
#         lambda f: "2D" if f in TASKS_2D
#         else ("3D" if f in TASKS_3D else "Other"))
#     return df


# def merge_data(d90, d60):
#     combined = pd.concat([d90, d60], ignore_index=True)
#     combined = combined[combined["task_type"].isin(["2D", "3D"])].copy()
#     ids  = ["session", "analysis_folder", "frequency", "task_type"]
#     vals = [c for c in combined.columns if c not in ids + ["type"]]
#     return (combined
#             .groupby(ids, dropna=False)
#             .agg({c: "first" for c in vals})
#             .reset_index())


# # ────────────────────────────────────────────────────────────
# # REUSABLE: FRIEDMAN + WILCOXON POST-HOC
# # ────────────────────────────────────────────────────────────
# def friedman_wilcoxon_posthoc(pivot, task_names, col, label,
#                               comparison_name, freq, rows,
#                               n_metric_correction, hypothesis,
#                               use_mean=True):
#     """
#     Friedman omnibus + Bonferroni-corrected Wilcoxon post-hoc.
#     use_mean=True  → report MEAN   (Kapp §4.4: spatial + H3 behaviour)
#     use_mean=False → report MEDIAN (temporal ISI)
#     """
#     prefix = stat_prefix(use_mean)

#     data_arrays = [pivot[t].values for t in task_names]
#     try:
#         chi2, p_fried = sp_stats.friedmanchisquare(*data_arrays)
#     except Exception:
#         return

#     p_fried_corr = min(p_fried * n_metric_correction, 1.0)

#     central = {t: summarize_group(pivot[t], use_mean) for t in task_names}
#     central_str = "; ".join([f"{t}: {central[t]:.4f}" for t in task_names])

#     rows.append({
#         "Hypothesis": hypothesis,
#         "Comparison": f"{comparison_name} ({freq})",
#         "Test": "Friedman (Omnibus)",
#         "Metric": label,
#         "Summary_Stat": prefix,
#         "N_Sessions": len(pivot),
#         "Tasks_Compared": ", ".join(task_names),
#         "Central_Values": central_str,
#         "Statistic_Chi2": round(chi2, 4),
#         "P_raw": round(p_fried, 6),
#         "P_Bonferroni": round(p_fried_corr, 6),
#         "Significance": sig_stars(p_fried_corr),
#         "Winner": "N/A (Omnibus)",
#         "Absolute_Gain": np.nan,
#         "Relative_Improvement_%": np.nan,
#         "Reason": "Kapp (2021) – Friedman + Wilcoxon post-hoc"
#     })

#     if p_fried_corr >= 0.05:
#         return

#     pairs  = list(combinations(task_names, 2))
#     p_vals = []
#     for t1, t2 in pairs:
#         try:
#             _, pw = sp_stats.wilcoxon(pivot[t1], pivot[t2])
#         except Exception:
#             pw = np.nan
#         p_vals.append(pw)

#     reject, p_corr, _, _ = multipletests(
#         [p if not np.isnan(p) else 1.0 for p in p_vals],
#         alpha=ALPHA, method='bonferroni')

#     for i, (t1, t2) in enumerate(pairs):
#         val1 = summarize_group(pivot[t1], use_mean)
#         val2 = summarize_group(pivot[t2], use_mean)
#         winner, gain, rel = get_better_info(
#             col, val1, val2, t1, t2, p_corr[i])

#         try:
#             stat_w, _ = sp_stats.wilcoxon(pivot[t1], pivot[t2])
#             n = len(pivot)
#             mu    = n * (n + 1) / 4
#             sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
#             z = (stat_w - mu) / sigma if sigma > 0 else 0
#         except Exception:
#             z = np.nan

#         rows.append({
#             "Hypothesis": hypothesis,
#             "Comparison": f"Post-Hoc: {t1} vs {t2} ({freq})",
#             "Test": "Wilcoxon Signed-Rank (Post-Hoc)",
#             "Metric": label,
#             "Summary_Stat": prefix,
#             "N_Pairs": len(pivot),
#             "Tasks_Compared": f"{t1} vs {t2}",
#             f"{prefix}_{t1}": round(val1, 4),
#             f"{prefix}_{t2}": round(val2, 4),
#             "Winner": winner,
#             "Absolute_Gain": gain,
#             "Relative_Improvement_%": rel,
#             "Z_score": round(z, 4) if not np.isnan(z) else np.nan,
#             "P_raw": (round(p_vals[i], 6)
#                       if not np.isnan(p_vals[i]) else np.nan),
#             "P_Bonferroni": round(p_corr[i], 6),
#             "Significance": sig_stars(p_corr[i]),
#             "Cohen_d": round(cohens_d(pivot[t1], pivot[t2]), 4),
#             "Effect_Size_r": (round(rank_biserial_r(z, len(pivot)), 4)
#                               if not np.isnan(z) else np.nan),
#             "Reason": "Kapp (2021) – Bonferroni-corrected pairwise"
#         })


# # ────────────────────────────────────────────────────────────
# # REUSABLE: MANN-WHITNEY U (between frequency)
# # ────────────────────────────────────────────────────────────
# def mannwhitney_between_freq(df, col, label, rows,
#                              n_correction, hypothesis,
#                              task_filter=None,
#                              use_mean=True):
#     """
#     Mann-Whitney U: 90Hz vs 60Hz.
#     use_mean=True  → MEAN   (Kapp §4.4: spatial + H3)
#     use_mean=False → MEDIAN (temporal ISI)
#     """
#     prefix = stat_prefix(use_mean)

#     if task_filter is not None:
#         df_use = df[df["analysis_folder"].isin(task_filter)]
#     else:
#         df_use = df

#     g90 = df_use.loc[df_use["frequency"] == "90Hz", col].dropna()
#     g60 = df_use.loc[df_use["frequency"] == "60Hz", col].dropna()
#     if len(g90) < 3 or len(g60) < 3:
#         return

#     u, p = sp_stats.mannwhitneyu(g90, g60, alternative='two-sided')
#     n1, n2 = len(g90), len(g60)
#     mu    = n1 * n2 / 2
#     sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
#     z = (u - mu) / sigma if sigma > 0 else 0
#     pc = min(p * n_correction, 1.0)

#     val90 = summarize_group(g90, use_mean)
#     val60 = summarize_group(g60, use_mean)
#     winner, gain, rel = get_better_info(
#         col, val90, val60, "90Hz", "60Hz", pc)

#     scope = "All Tasks" if task_filter is None else "3D Tasks"
#     rows.append({
#         "Hypothesis": hypothesis,
#         "Comparison": f"90Hz vs 60Hz ({scope})",
#         "Test": "Mann-Whitney U",
#         "Metric": label,
#         "Summary_Stat": prefix,
#         "N_90Hz": n1,
#         "N_60Hz": n2,
#         f"{prefix}_90Hz": round(val90, 4),
#         f"{prefix}_60Hz": round(val60, 4),
#         "Winner": winner,
#         "Absolute_Gain": gain,
#         "Relative_Improvement_%": rel,
#         "Statistic_U": u,
#         "Z_score": round(z, 4),
#         "P_raw": p,
#         "P_Bonferroni": pc,
#         "Significance": sig_stars(pc),
#         "Cohen_d": round(cohens_d(g90, g60), 4),
#         "Effect_Size_r": round(rank_biserial_r(z, n1 + n2), 4),
#         "Reason": "Awasthi (2023) – Bonferroni-corrected MWU"
#     })
#     print(f"  {label:40s} p_c={pc:.4f} "
#           f"{sig_stars(pc)}  Winner: {winner}")


# # ────────────────────────────────────────────────────────────
# # MAIN ANALYSIS
# # ────────────────────────────────────────────────────────────
# def run_analysis(df):
#     rows = []
#     n_spatial  = len(METRICS_SPATIAL)        # 3
#     n_3d_beh   = len(METRICS_3D_BEHAVIOUR)   # 3
#     n_temporal = len(METRICS_TEMPORAL)       # 1

#     # ============================================================
#     # H2: Sampling rate affects spatial accuracy/precision
#     #     90Hz vs 60Hz  |  use_mean=True (Kapp §4.4)
#     # ============================================================
#     print("=" * 70)
#     print("H2: BETWEEN FREQUENCY – Spatial Metrics (90Hz vs 60Hz)")
#     print("    Summary: MEAN (Kapp 2021 §4.4)")
#     print("=" * 70)

#     for col, label in METRICS_SPATIAL.items():
#         if col not in df.columns:
#             continue
#         mannwhitney_between_freq(
#             df, col, label, rows,
#             n_correction=n_spatial,
#             hypothesis="H2",
#             use_mean=True)              # ← MEAN per Kapp §4.4

#     # ============================================================
#     # H1: Stimuli affect accuracy/precision
#     #     Within 2D, within 3D, between 2D vs 3D
#     #     use_mean=True (Kapp §4.4)
#     # ============================================================
#     print("\n" + "=" * 70)
#     print("H1: WITHIN FREQUENCY – Spatial Metrics")
#     print("    Summary: MEAN (Kapp 2021 §4.4)")
#     print("=" * 70)

#     for freq in ["90Hz", "60Hz"]:
#         df_f = df[df["frequency"] == freq]
#         if df_f.empty:
#             continue

#         for col, label in METRICS_SPATIAL.items():
#             if col not in df_f.columns:
#                 continue

#             # ── H1a: Within 2D (Distance Effect) ──
#             pivot_2d = (df_f[df_f["analysis_folder"].isin(TASKS_2D)]
#                         .pivot(index="session",
#                                columns="analysis_folder",
#                                values=col).dropna())
#             valid_2d = [t for t in TASKS_2D if t in pivot_2d.columns]
#             if len(valid_2d) >= 3 and len(pivot_2d) >= 6:
#                 print(f"\n  [{freq}] H1a: {label} within 2D:")
#                 friedman_wilcoxon_posthoc(
#                     pivot_2d, valid_2d, col, label,
#                     "H1a: Within 2D (Distance)", freq, rows,
#                     n_spatial, "H1",
#                     use_mean=True)      # ← MEAN per Kapp §4.4

#             # ── H1b: Within 3D ──
#             pivot_3d = (df_f[df_f["analysis_folder"].isin(TASKS_3D)]
#                         .pivot(index="session",
#                                columns="analysis_folder",
#                                values=col).dropna())
#             valid_3d = [t for t in TASKS_3D if t in pivot_3d.columns]
#             if len(valid_3d) >= 3 and len(pivot_3d) >= 6:
#                 print(f"\n  [{freq}] H1b: {label} within 3D:")
#                 friedman_wilcoxon_posthoc(
#                     pivot_3d, valid_3d, col, label,
#                     "H1b: Within 3D", freq, rows,
#                     n_spatial, "H1",
#                     use_mean=True)      # ← MEAN per Kapp §4.4

#             # ── H1c: Between 2D vs 3D (Wilcoxon) ──
#             s2d = (df_f[df_f["task_type"] == "2D"]
#                    .groupby("session")[col].mean().dropna())
#             s3d = (df_f[df_f["task_type"] == "3D"]
#                    .groupby("session")[col].mean().dropna())
#             common = s2d.index.intersection(s3d.index)
#             if len(common) >= 6:
#                 v2d, v3d = s2d.loc[common], s3d.loc[common]
#                 try:
#                     stat, p = sp_stats.wilcoxon(v2d, v3d)
#                 except Exception:
#                     continue
#                 n     = len(common)
#                 mu    = n * (n + 1) / 4
#                 sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
#                 z = (stat - mu) / sigma if sigma > 0 else 0
#                 pc = min(p * n_spatial, 1.0)

#                 val2d = summarize_group(v2d, use_mean=True)
#                 val3d = summarize_group(v3d, use_mean=True)
#                 winner, gain, rel = get_better_info(
#                     col, val2d, val3d,
#                     "2D Tasks", "3D Tasks", pc)

#                 rows.append({
#                     "Hypothesis": "H1",
#                     "Comparison": f"H1c: 2D vs 3D ({freq})",
#                     "Test": "Wilcoxon Signed-Rank",
#                     "Metric": label,
#                     "Summary_Stat": "Mean",
#                     "N_Pairs": n,
#                     "Mean_2D": round(val2d, 4),
#                     "Mean_3D": round(val3d, 4),
#                     "Winner": winner,
#                     "Absolute_Gain": gain,
#                     "Relative_Improvement_%": rel,
#                     "Statistic_W": stat,
#                     "Z_score": round(z, 4),
#                     "P_raw": p,
#                     "P_Bonferroni": pc,
#                     "Significance": sig_stars(pc),
#                     "Cohen_d": round(cohens_d(v2d, v3d), 4),
#                     "Effect_Size_r": round(rank_biserial_r(z, n), 4),
#                     "Reason": "Kapp (2021) – Paired non-parametric"
#                 })
#                 print(f"  [{freq}] H1c: {label:30s} "
#                       f"p_c={pc:.4f} {sig_stars(pc)}  "
#                       f"Winner: {winner}")

#     # ============================================================
#     # H3: Viewing behaviour differs across 3D tasks
#     #     use_mean=True (behavioural magnitude)
#     # ============================================================
#     print("\n" + "=" * 70)
#     print("H3: VIEWING BEHAVIOUR – 3D Tasks Only")
#     print("    Summary: MEAN")
#     print("=" * 70)

#     # H3a: Between frequency (3D tasks only)
#     print("\n  H3a: 90Hz vs 60Hz (3D tasks only):")
#     for col, label in METRICS_3D_BEHAVIOUR.items():
#         if col not in df.columns:
#             continue
#         mannwhitney_between_freq(
#             df, col, label, rows,
#             n_correction=n_3d_beh,
#             hypothesis="H3",
#             task_filter=TASKS_3D,
#             use_mean=True)              # ← MEAN

#     # H3b: Within each frequency across 3D tasks
#     for freq in ["90Hz", "60Hz"]:
#         df_f = df[df["frequency"] == freq]
#         if df_f.empty:
#             continue

#         for col, label in METRICS_3D_BEHAVIOUR.items():
#             if col not in df_f.columns:
#                 continue
#             pivot = (df_f[df_f["analysis_folder"].isin(TASKS_3D)]
#                      .pivot(index="session",
#                             columns="analysis_folder",
#                             values=col).dropna())
#             valid = [t for t in TASKS_3D if t in pivot.columns]
#             if len(valid) < 3 or len(pivot) < 6:
#                 continue
#             print(f"\n  [{freq}] H3b: {label} across 3D tasks:")
#             friedman_wilcoxon_posthoc(
#                 pivot, valid, col, label,
#                 "H3b: 3D Tasks Behaviour", freq, rows,
#                 n_3d_beh, "H3",
#                 use_mean=True)          # ← MEAN

#     # ============================================================
#     # TEMPORAL VALIDATION: ISI
#     #     use_mean=False → MEDIAN
#     #     Aziz & Komogortsev (2022) §3.4
#     # ============================================================
#     print("\n" + "=" * 70)
#     print("TEMPORAL VALIDATION: ISI (All Tasks)")
#     print("    Summary: MEDIAN (Aziz 2022 §3.4)")
#     print("=" * 70)

#     # Between frequency
#     for col, label in METRICS_TEMPORAL.items():
#         if col not in df.columns:
#             continue
#         mannwhitney_between_freq(
#             df, col, label, rows,
#             n_correction=n_temporal,
#             hypothesis="Temporal",
#             use_mean=False)             # ← MEDIAN

#     # Within frequency across all tasks
#     for freq in ["90Hz", "60Hz"]:
#         df_f = df[df["frequency"] == freq]
#         if df_f.empty:
#             continue
#         for col, label in METRICS_TEMPORAL.items():
#             if col not in df_f.columns:
#                 continue
#             pivot = (df_f[df_f["analysis_folder"].isin(ALL_TASKS)]
#                      .pivot(index="session",
#                             columns="analysis_folder",
#                             values=col).dropna())
#             valid = [t for t in ALL_TASKS if t in pivot.columns]
#             if len(valid) < 3 or len(pivot) < 6:
#                 continue
#             print(f"\n  [{freq}] Temporal: {label} "
#                   f"across {len(valid)} tasks:")
#             friedman_wilcoxon_posthoc(
#                 pivot, valid, col, label,
#                 "Temporal: ISI (All Tasks)", freq, rows,
#                 n_temporal, "Temporal",
#                 use_mean=False)         # ← MEDIAN

#     return pd.DataFrame(rows)


# # ────────────────────────────────────────────────────────────
# # MAIN
# # ────────────────────────────────────────────────────────────
# def main():
#     df_90 = load_and_tag(RESULTS_90, "90Hz")
#     df_60 = load_and_tag(RESULTS_60, "60Hz")

#     if df_90.empty and df_60.empty:
#         print("ERROR: No data found.")
#         sys.exit(1)

#     df = merge_data(df_90, df_60)
#     print(f"Loaded {len(df)} valid session-task rows.\n")

#     df_res = run_analysis(df)

#     if df_res.empty:
#         print("No results generated.")
#         return

#     # ── Column ordering for Excel ──
#     fixed = [
#         "Hypothesis", "Comparison", "Test", "Metric",
#         "Summary_Stat",
#         "Significance", "Winner",
#         "Absolute_Gain", "Relative_Improvement_%",
#         "P_Bonferroni", "P_raw",
#         "Cohen_d", "Effect_Size_r", "Reason"
#     ]
#     stat_cols = [
#         c for c in df_res.columns
#         if c.startswith("Median_") or c.startswith("Mean_")
#         or c.startswith("N_")
#         or c == "Tasks_Compared" or c == "Central_Values"
#         or c.startswith("Statistic") or c == "Z_score"
#     ]
#     final_cols = fixed[:5] + stat_cols + fixed[5:]
#     final_cols = [c for c in final_cols if c in df_res.columns]

#     seen = set()
#     ordered = []
#     for c in final_cols:
#         if c not in seen:
#             seen.add(c)
#             ordered.append(c)

#     df_out = (df_res[ordered]
#               .sort_values(by=["Hypothesis", "Metric", "Comparison"]))

#     out_path = OUTPUT_DIR / "Literature_Backed_Stats_Results.csv"
#     df_out.to_csv(out_path, index=False)

#     sig = df_out["Significance"].isin(["*", "**", "***"])
#     print(f"\n{'=' * 70}")
#     print(f"Saved: {out_path}")
#     print(f"Total rows: {len(df_out)}")
#     print(f"Significant (corrected): "
#           f"{sig.sum()} / {len(df_out)}")
#     print(f"{'=' * 70}")


# if __name__ == "__main__":
#     main()









import os, sys, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.sandbox.stats.multicomp import multipletests

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────
DATA_ROOT   = Path(r"C:\Users\User\Desktop\Python\MRProcessing\data")
RESULTS_90  = DATA_ROOT / "high_90hz" / "RESULTS" / "summary" / "summary_all_results.csv"
RESULTS_60  = DATA_ROOT / "mid_60hz"  / "RESULTS" / "summary" / "summary_all_results.csv"
OUTPUT_DIR  = DATA_ROOT / "RESULTS" / "statistical_comparison_FINAL"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
# TASK GROUPINGS (CRITICAL FIX)
# UD0411(83) is FREE VIEWING -> NO accuracy, NO precision
# PlaneWSW is TRACKING -> NO precision
# ════════════════════════════════════════════════════════════
TASKS_2D = ["PlaneClose", "PlaneMiddle", "PlaneFar"]

TASKS_3D_SACCADE  = ["ObjectCone(1)", "ObjectCone(3)"] # Has Accuracy + Precision
TASKS_3D_TRACKING = ["PlaneWSW"]                       # Has Accuracy, NO Precision
TASKS_3D_FREEVIEW = ["UD0411(83)"]                     # NO Accuracy, NO Precision

TASKS_3D_ALL = TASKS_3D_SACCADE + TASKS_3D_TRACKING + TASKS_3D_FREEVIEW
TASKS_3D_WITH_ACCURACY = TASKS_3D_SACCADE + TASKS_3D_TRACKING
TASKS_3D_WITH_PRECISION = TASKS_3D_SACCADE  # Only 2 tasks!

ALL_TASKS = TASKS_2D + TASKS_3D_ALL

# ════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════
METRICS_SPATIAL = {
    "ang_err_mean":          "Angular Error (deg)",
    "precision_rms_3d_mean": "Precision RMS 3D (deg)",
    "dist_2d_mean":          "2D Distance Error (m)",
}

METRICS_3D_BEHAVIOUR = {
    "pos_vel_mean":          "Head Positional Velocity (m/s)",
    "ang_vel_mean":          "Head Angular Velocity (deg/s)",
    "eye_gaze_dist_mean":    "Viewing Distance (m)",
}

METRICS_TEMPORAL = {
    "isi_median":            "ISI Median (ms)",
}

ALPHA = 0.05


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
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
    pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0

def rank_biserial_r(z, n):
    return abs(z) / np.sqrt(n) if n > 0 else 0.0

def summarize_group(series, use_mean=True):
    return series.mean() if use_mean else series.median()

def stat_prefix(use_mean):
    return "Mean" if use_mean else "Median"

def get_better_info(metric_col, val1, val2, name1, name2, p_corr):
    if pd.isna(p_corr) or p_corr >= 0.05:
        return "No sig. difference (p>=0.05)", np.nan, np.nan
    lower_better = ['ang_err', 'precision', 'dist_2d', 'pos_vel', 'ang_vel', 'eye_gaze', 'isi']
    is_lower = any(m in metric_col for m in lower_better)
    if is_lower:
        if val1 < val2: winner, loser_val, gain = name1, val2, val2 - val1
        elif val2 < val1: winner, loser_val, gain = name2, val1, val1 - val2
        else: return "Equal", 0, 0
    else:
        if val1 > val2: winner, loser_val, gain = name1, val2, val1 - val2
        elif val2 > val1: winner, loser_val, gain = name2, val1, val2 - val1
        else: return "Equal", 0, 0
    rel = (gain / loser_val) * 100 if loser_val != 0 else np.nan
    return winner, round(gain, 4), round(rel, 2)

def load_and_tag(path, freq):
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path)
    df["frequency"] = freq
    # Assign task types
    def get_type(f):
        if f in TASKS_2D: return "2D"
        if f in TASKS_3D_ALL: return "3D"
        return "Other"
    df["task_type"] = df["analysis_folder"].apply(get_type)
    return df

def merge_data(d90, d60):
    combined = pd.concat([d90, d60], ignore_index=True)
    combined = combined[combined["task_type"].isin(["2D", "3D"])].copy()
    ids = ["session", "analysis_folder", "frequency", "task_type"]
    if "type" in combined.columns:
        combined = combined.drop(columns=["type"])
    # .first() skips NaN and finds first valid value per column
    return combined.groupby(ids, dropna=False).first().reset_index()


# ────────────────────────────────────────────────────────────
# REUSABLE TESTS
# ────────────────────────────────────────────────────────────
def friedman_wilcoxon_posthoc(pivot, task_names, col, label, comparison_name, freq, rows, n_metric_correction, hypothesis, use_mean=True):
    prefix = stat_prefix(use_mean)
    data_arrays = [pivot[t].values for t in task_names]
    try:
        chi2, p_fried = sp_stats.friedmanchisquare(*data_arrays)
    except Exception as e:
        print(f"    [!] Friedman failed: {e}")
        return

    p_fried_corr = min(p_fried * n_metric_correction, 1.0)
    print(f"    Omnibus: Chi2={chi2:.3f}, p_raw={p_fried:.4f}, p_corr={p_fried_corr:.4f} {sig_stars(p_fried_corr)}")

    central = {t: summarize_group(pivot[t], use_mean) for t in task_names}
    central_str = "; ".join([f"{t}: {central[t]:.4f}" for t in task_names])

    rows.append({
        "Hypothesis": hypothesis, "Comparison": f"{comparison_name} ({freq})",
        "Test": "Friedman (Omnibus)", "Metric": label, "Summary_Stat": prefix,
        "N_Sessions": len(pivot), "Tasks_Compared": ", ".join(task_names),
        "Central_Values": central_str, "Statistic_Chi2": round(chi2, 4),
        "P_raw": round(p_fried, 6), "P_Bonferroni": round(p_fried_corr, 6),
        "Significance": sig_stars(p_fried_corr), "Winner": "N/A (Omnibus)",
        "Absolute_Gain": np.nan, "Relative_Improvement_%": np.nan,
        "Reason": "Kapp (2021) – Friedman + Wilcoxon post-hoc"
    })

    if p_fried_corr >= 0.05: return

    pairs  = list(combinations(task_names, 2))
    p_vals = []
    for t1, t2 in pairs:
        try: _, pw = sp_stats.wilcoxon(pivot[t1], pivot[t2])
        except Exception: pw = np.nan
        p_vals.append(pw)

    reject, p_corr, _, _ = multipletests([p if not np.isnan(p) else 1.0 for p in p_vals], alpha=ALPHA, method='bonferroni')

    for i, (t1, t2) in enumerate(pairs):
        val1 = summarize_group(pivot[t1], use_mean)
        val2 = summarize_group(pivot[t2], use_mean)
        winner, gain, rel = get_better_info(col, val1, val2, t1, t2, p_corr[i])

        try:
            stat_w, _ = sp_stats.wilcoxon(pivot[t1], pivot[t2])
            n = len(pivot)
            mu = n * (n + 1) / 4
            sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z = (stat_w - mu) / sigma if sigma > 0 else 0
        except Exception: z = np.nan

        rows.append({
            "Hypothesis": hypothesis, "Comparison": f"Post-Hoc: {t1} vs {t2} ({freq})",
            "Test": "Wilcoxon Signed-Rank (Post-Hoc)", "Metric": label, "Summary_Stat": prefix,
            "N_Pairs": len(pivot), "Tasks_Compared": f"{t1} vs {t2}",
            f"{prefix}_{t1}": round(val1, 4), f"{prefix}_{t2}": round(val2, 4),
            "Winner": winner, "Absolute_Gain": gain, "Relative_Improvement_%": rel,
            "Z_score": round(z, 4) if not np.isnan(z) else np.nan,
            "P_raw": round(p_vals[i], 6) if not np.isnan(p_vals[i]) else np.nan,
            "P_Bonferroni": round(p_corr[i], 6), "Significance": sig_stars(p_corr[i]),
            "Cohen_d": round(cohens_d(pivot[t1], pivot[t2]), 4),
            "Effect_Size_r": round(rank_biserial_r(z, len(pivot)), 4) if not np.isnan(z) else np.nan,
            "Reason": "Kapp (2021) – Bonferroni-corrected pairwise"
        })
        print(f"    Post-Hoc: {t1} vs {t2}: p_c={p_corr[i]:.4f} {sig_stars(p_corr[i])} Winner: {winner}")


def wilcoxon_two_tasks(pivot, t1, t2, col, label, comparison_name, freq, rows, n_metric_correction, hypothesis, use_mean=True):
    """Fallback for when only 2 tasks are available (e.g. Precision in 3D)"""
    prefix = stat_prefix(use_mean)
    v1, v2 = pivot[t1], pivot[t2]
    try:
        stat, p = sp_stats.wilcoxon(v1, v2)
    except Exception as e:
        print(f"    [!] Wilcoxon failed: {e}")
        return
        
    n = len(v1)
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (stat - mu) / sigma if sigma > 0 else 0
    p_corr = min(p * n_metric_correction, 1.0)
    
    val1 = summarize_group(v1, use_mean)
    val2 = summarize_group(v2, use_mean)
    winner, gain, rel = get_better_info(col, val1, val2, t1, t2, p_corr)
    
    print(f"    Wilcoxon (2 tasks): p_raw={p:.4f}, p_corr={p_corr:.4f} {sig_stars(p_corr)} Winner: {winner}")
    
    rows.append({
        "Hypothesis": hypothesis, "Comparison": f"{comparison_name} ({freq})",
        "Test": "Wilcoxon Signed-Rank (2 tasks)", "Metric": label, "Summary_Stat": prefix,
        "N_Pairs": n, "Tasks_Compared": f"{t1} vs {t2}",
        f"{prefix}_{t1}": round(val1, 4), f"{prefix}_{t2}": round(val2, 4),
        "Winner": winner, "Absolute_Gain": gain, "Relative_Improvement_%": rel,
        "Statistic_W": stat, "Z_score": round(z, 4),
        "P_raw": round(p, 6), "P_Bonferroni": round(p_corr, 6),
        "Significance": sig_stars(p_corr), "Cohen_d": round(cohens_d(v1, v2), 4),
        "Effect_Size_r": round(rank_biserial_r(z, n), 4),
        "Reason": "Kapp (2021) – Wilcoxon (only 2 tasks available)"
    })


def mannwhitney_between_freq(df, col, label, rows, n_correction, hypothesis, task_filter=None, use_mean=True):
    prefix = stat_prefix(use_mean)
    df_use = df[df["analysis_folder"].isin(task_filter)] if task_filter else df
    g90 = df_use.loc[df_use["frequency"] == "90Hz", col].dropna()
    g60 = df_use.loc[df_use["frequency"] == "60Hz", col].dropna()
    if len(g90) < 3 or len(g60) < 3: return

    u, p = sp_stats.mannwhitneyu(g90, g60, alternative='two-sided')
    n1, n2 = len(g90), len(g60)
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sigma if sigma > 0 else 0
    pc = min(p * n_correction, 1.0)

    val90 = summarize_group(g90, use_mean)
    val60 = summarize_group(g60, use_mean)
    winner, gain, rel = get_better_info(col, val90, val60, "90Hz", "60Hz", pc)

    scope = "All Tasks" if not task_filter else "3D Tasks"
    rows.append({
        "Hypothesis": hypothesis, "Comparison": f"90Hz vs 60Hz ({scope})",
        "Test": "Mann-Whitney U", "Metric": label, "Summary_Stat": prefix,
        "N_90Hz": n1, "N_60Hz": n2,
        f"{prefix}_90Hz": round(val90, 4), f"{prefix}_60Hz": round(val60, 4),
        "Winner": winner, "Absolute_Gain": gain, "Relative_Improvement_%": rel,
        "Statistic_U": u, "Z_score": round(z, 4), "P_raw": p, "P_Bonferroni": pc,
        "Significance": sig_stars(pc), "Cohen_d": round(cohens_d(g90, g60), 4),
        "Effect_Size_r": round(rank_biserial_r(z, n1 + n2), 4),
        "Reason": "Awasthi (2023) – Bonferroni-corrected MWU"
    })
    print(f"  {label:40s} p_c={pc:.4f} {sig_stars(pc)}  Winner: {winner}")


# ────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ────────────────────────────────────────────────────────────
def run_analysis(df):
    rows = []
    n_spatial  = len(METRICS_SPATIAL)
    n_3d_beh   = len(METRICS_3D_BEHAVIOUR)
    n_temporal = len(METRICS_TEMPORAL)

    # H2: Between Frequency
    print("=" * 70 + "\nH2: BETWEEN FREQUENCY\n" + "=" * 70)
    for col, label in METRICS_SPATIAL.items():
        if col in df.columns:
            # Filter tasks based on metric availability
            if "precision" in col: tasks_filt = TASKS_2D + TASKS_3D_WITH_PRECISION
            else: tasks_filt = TASKS_2D + TASKS_3D_WITH_ACCURACY
            
            mannwhitney_between_freq(df, col, label, rows, n_spatial, "H2", task_filter=tasks_filt, use_mean=True)

    # H1: Within Frequency
    print("\n" + "=" * 70 + "\nH1: WITHIN FREQUENCY\n" + "=" * 70)
    for freq in ["90Hz", "60Hz"]:
        df_f = df[df["frequency"] == freq]
        if df_f.empty: continue

        for col, label in METRICS_SPATIAL.items():
            if col not in df_f.columns: continue

            # H1a: Within 2D
            pivot_2d = df_f[df_f["analysis_folder"].isin(TASKS_2D)].pivot(index="session", columns="analysis_folder", values=col)
            valid_2d = [t for t in TASKS_2D if t in pivot_2d.columns and pivot_2d[t].notna().sum() >= 3]
            if len(valid_2d) >= 3:
                pivot_2d_clean = pivot_2d[valid_2d].dropna()
                if len(pivot_2d_clean) >= 3:
                    print(f"\n  [{freq}] H1a: {label} within 2D:")
                    friedman_wilcoxon_posthoc(pivot_2d_clean, valid_2d, col, label, "H1a: Within 2D", freq, rows, n_spatial, "H1", use_mean=True)

            # H1b: Within 3D (Dynamic Task Selection)
            if "precision" in col:
                tasks_3d = TASKS_3D_WITH_PRECISION  # Only Saccade (2 tasks)
            else:
                tasks_3d = TASKS_3D_WITH_ACCURACY   # Saccade + Tracking (3 tasks)
                
            pivot_3d = df_f[df_f["analysis_folder"].isin(tasks_3d)].pivot(index="session", columns="analysis_folder", values=col)
            valid_3d = [t for t in tasks_3d if t in pivot_3d.columns and pivot_3d[t].notna().sum() >= 3]
            
            if len(valid_3d) >= 3:
                pivot_3d_clean = pivot_3d[valid_3d].dropna()
                if len(pivot_3d_clean) >= 3:
                    print(f"\n  [{freq}] H1b: {label} within 3D:")
                    friedman_wilcoxon_posthoc(pivot_3d_clean, valid_3d, col, label, "H1b: Within 3D", freq, rows, n_spatial, "H1", use_mean=True)
            elif len(valid_3d) == 2:
                pivot_3d_clean = pivot_3d[valid_3d].dropna()
                if len(pivot_3d_clean) >= 3:
                    print(f"\n  [{freq}] H1b: {label} within 3D (2 tasks available):")
                    wilcoxon_two_tasks(pivot_3d_clean, valid_3d[0], valid_3d[1], col, label, "H1b: Within 3D", freq, rows, n_spatial, "H1", use_mean=True)

            # H1c: 2D vs 3D
            s2d = df_f[df_f["analysis_folder"].isin(TASKS_2D)].groupby("session")[col].mean().dropna()
            s3d = df_f[df_f["analysis_folder"].isin(tasks_3d)].groupby("session")[col].mean().dropna() # Uses dynamic tasks_3d
            common = s2d.index.intersection(s3d.index)
            if len(common) >= 3:
                v2d, v3d = s2d.loc[common], s3d.loc[common]
                try: stat, p = sp_stats.wilcoxon(v2d, v3d)
                except Exception: continue
                n = len(common)
                mu = n * (n + 1) / 4
                sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                z = (stat - mu) / sigma if sigma > 0 else 0
                pc = min(p * n_spatial, 1.0)
                val2d, val3d = summarize_group(v2d, True), summarize_group(v3d, True)
                winner, gain, rel = get_better_info(col, val2d, val3d, "2D Tasks", "3D Tasks", pc)
                rows.append({
                    "Hypothesis": "H1", "Comparison": f"H1c: 2D vs 3D ({freq})",
                    "Test": "Wilcoxon Signed-Rank", "Metric": label, "Summary_Stat": "Mean",
                    "N_Pairs": n, "Mean_2D": round(val2d, 4), "Mean_3D": round(val3d, 4),
                    "Winner": winner, "Absolute_Gain": gain, "Relative_Improvement_%": rel,
                    "Statistic_W": stat, "Z_score": round(z, 4), "P_raw": p, "P_Bonferroni": pc,
                    "Significance": sig_stars(pc), "Cohen_d": round(cohens_d(v2d, v3d), 4),
                    "Effect_Size_r": round(rank_biserial_r(z, n), 4),
                    "Reason": "Kapp (2021) – Paired non-parametric"
                })
                print(f"  [{freq}] H1c: {label:30s} p_c={pc:.4f} {sig_stars(pc)}  Winner: {winner}")

    # H3: Viewing Behaviour (ALL 3D tasks)
    print("\n" + "=" * 70 + "\nH3: VIEWING BEHAVIOUR (3D Tasks)\n" + "=" * 70)
    for col, label in METRICS_3D_BEHAVIOUR.items():
        if col in df.columns:
            mannwhitney_between_freq(df, col, label, rows, n_3d_beh, "H3", task_filter=TASKS_3D_ALL, use_mean=True)

    for freq in ["90Hz", "60Hz"]:
        df_f = df[df["frequency"] == freq]
        if df_f.empty: continue
        for col, label in METRICS_3D_BEHAVIOUR.items():
            if col not in df_f.columns: continue
            pivot = df_f[df_f["analysis_folder"].isin(TASKS_3D_ALL)].pivot(index="session", columns="analysis_folder", values=col)
            valid = [t for t in TASKS_3D_ALL if t in pivot.columns and pivot[t].notna().sum() >= 3]
            if len(valid) >= 3:
                pivot_clean = pivot[valid].dropna()
                if len(pivot_clean) >= 3:
                    print(f"\n  [{freq}] H3b: {label} across 3D tasks:")
                    friedman_wilcoxon_posthoc(pivot_clean, valid, col, label, "H3b: 3D Behaviour", freq, rows, n_3d_beh, "H3", use_mean=True)

    # Temporal: ISI (ALL tasks)
    print("\n" + "=" * 70 + "\nTEMPORAL VALIDATION: ISI\n" + "=" * 70)
    for col, label in METRICS_TEMPORAL.items():
        if col in df.columns:
            mannwhitney_between_freq(df, col, label, rows, n_temporal, "Temporal", use_mean=False)

    for freq in ["90Hz", "60Hz"]:
        df_f = df[df["frequency"] == freq]
        if df_f.empty: continue
        for col, label in METRICS_TEMPORAL.items():
            if col not in df_f.columns: continue
            pivot = df_f[df_f["analysis_folder"].isin(ALL_TASKS)].pivot(index="session", columns="analysis_folder", values=col)
            valid = [t for t in ALL_TASKS if t in pivot.columns and pivot[t].notna().sum() >= 3]
            if len(valid) >= 3:
                pivot_clean = pivot[valid].dropna()
                if len(pivot_clean) >= 3:
                    print(f"\n  [{freq}] Temporal: {label}:")
                    friedman_wilcoxon_posthoc(pivot_clean, valid, col, label, "Temporal: ISI", freq, rows, n_temporal, "Temporal", use_mean=False)

    return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    df_90 = load_and_tag(RESULTS_90, "90Hz")
    df_60 = load_and_tag(RESULTS_60, "60Hz")
    if df_90.empty and df_60.empty:
        print("ERROR: No data found."); sys.exit(1)

    df = merge_data(df_90, df_60)
    print(f"Loaded {len(df)} valid session-task rows.\n")

    df_res = run_analysis(df)
    if df_res.empty:
        print("No results generated."); return

    fixed = ["Hypothesis", "Comparison", "Test", "Metric", "Summary_Stat", "Significance", "Winner",
             "Absolute_Gain", "Relative_Improvement_%", "P_Bonferroni", "P_raw", "Cohen_d", "Effect_Size_r", "Reason"]
    stat_cols = [c for c in df_res.columns if c.startswith("Median_") or c.startswith("Mean_") or c.startswith("N_")
                 or c == "Tasks_Compared" or c == "Central_Values" or c.startswith("Statistic") or c == "Z_score"]
    
    final_cols = [c for c in fixed[:5] + stat_cols + fixed[5:] if c in df_res.columns]
    seen, ordered = set(), []
    for c in final_cols:
        if c not in seen: seen.add(c); ordered.append(c)

    df_out = df_res[ordered].sort_values(by=["Hypothesis", "Metric", "Comparison"])
    out_path = OUTPUT_DIR / "Literature_Backed_Stats_Results.csv"
    df_out.to_csv(out_path, index=False)

    sig = df_out["Significance"].isin(["*", "**", "***"])
    print(f"\n{'='*70}\nSaved: {out_path}\nTotal rows: {len(df_out)}\nSignificant: {sig.sum()} / {len(df_out)}\n{'='*70}")

if __name__ == "__main__":
    main()