# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""The paper's Table 5 and Table S3 as LaTeX, from nested_sweep.py's nested_sweep_results.json.

  table5_nested_summary.tex   Table 5: in-sample versus nested read-outs of the reference
                              detector (the configuration selected by the confirmation
                              objective), at the confirmation and at the detection date.
  table_s3_nested_folds.tex   Table S3: the configuration selected in each fold and its
                              effect on the held-out MU.

MU numbers follow the paper (paper_mu_numbers.py).
Usage: uv run python 07_detector/make_nested_tables.py [path/to/nested_sweep_results.json]
Env: DATA_DIR (default ./data); OUT_DIR (default DATA_DIR/runs/nested_sweep, where
     nested_sweep.py writes; also the default location of the results file)."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_mu_numbers import MU_NUMBER                       # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else DATA_DIR / "runs" / "nested_sweep"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "nested_sweep_results.json"


def day(v):
    if v is None:
        return "--"
    v = int(round(v))
    return f"\\phantom{{+}}{v}" if v == 0 else f"{v:+d}"


def num(v, nd=2):                                  # thresholds with 2 decimals, sigma with 1
    return f"{v:.{nd}f}"


def sd(v):                                         # signed day, "0" for zero
    return "0" if round(v) == 0 else f"{v:+.0f}"


if __name__ == "__main__":
    R = json.loads(RES.read_text(encoding="utf-8"))
    op = R["operating_points"]["confirmation"]; ref = R["insample_cfg"]["confirmation"]; N = R["n_mu"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Table S3: theta_(-k) per fold and its effect on the held-out MU ----
    rows = sorted(op["per_mu"], key=lambda m: MU_NUMBER[m["mu"]])
    L = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering",
         r"\caption{Nested leave-one-MU-out evaluation of the detector grid search. For each MU (Table~\ref{tab:mu_metadata} numbering), the grid search of the main text was repeated on the other 19 MUs with the same objective and tie-break, and the winning configuration $\theta_{(-k)}$ was applied unchanged to the held-out MU. Columns give the seven parameters of $\theta_{(-k)}$, whether it coincides with the reference configuration selected on all 20 MUs, the detection and confirmation lags of the held-out MU under $\theta_{(-k)}$ and under the reference, and the number of pre-onset observations at which $\theta_{(-k)}$ fires over the number of pre-onset observations in the record.}",
         r"\label{tab:nested_folds}", r"\footnotesize", r"\renewcommand{\arraystretch}{0.82}",
         r"\begin{tabular}{l rrrrrrr l rr rr r}", r"\toprule",
         r"& \multicolumn{7}{c}{$\theta_{(-k)}$: configuration selected on the other 19 MUs} & & \multicolumn{2}{c}{Held-out MU, $\theta_{(-k)}$} & \multicolumn{2}{c}{Held-out MU, reference} & \\",
         r"\cmidrule(lr){2-8} \cmidrule(lr){10-11} \cmidrule(lr){12-13}",
         r"MU & $f$ & $p$ & $T_E$ & $L_g$ & $\sigma$ & $T_G$ & $K$ & Same as ref. & Det. & Conf. & Det. & Conf. & Pre-onset firing \\", r"\midrule"]
    for m in rows:
        c = m["cfg"]; n = m["nested"]; s = m["insample"]
        L.append(f"{MU_NUMBER[m['mu']]} & {c['f']} & {c['p']} & {num(c['TE'])} & {c['L']} & {num(c['sigma'], 1)} & {num(c['TG'])} & {c['K']} & "
                 f"{'yes' if m['same_as_insample'] else 'no'} & {day(n['detection'])} & {day(n['confirmation'])} & {day(s['detection'])} & {day(s['confirmation'])} & "
                 f"{n['pre_onset_firing_obs']}/{n['n_pre_onset_obs']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"", r"\vspace{3pt}", r"\scriptsize",
          f"Notes. Reference configuration (all 20 MUs): $f={ref['f']}$, $p={ref['p']}$, $T_E={num(ref['TE'])}$, $L_g={ref['L']}$, $\\sigma={num(ref['sigma'], 1)}$, $T_G={num(ref['TG'])}$, $K={ref['K']}$; "
          f"{op['stability']['folds_identical_to_insample']} of the {N} folds re-select it. Lags in days relative to the onset $t_{{\\mathrm{{on}}}}$; negative values are early (false-positive) firings; ``--'' marks a MU at which the configuration never fires within the record. "
          r"The same objective applies in every fold: fewest early false positives, then fewest missed MUs, then the lowest median and mean confirmation lag, with remaining ties resolved by the enumeration order of the grid. The regressor's out-of-fold score series are the same in every fold; only the detector is nested.",
          r"\end{table}", r"\end{landscape}"]
    (OUT_DIR / "table_s3_nested_folds.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- Table 5: in-sample versus nested, both read-out dates ----
    A = op["aggregate"]

    def line(label, a):
        b = a["bootstrap"]
        return (f"{label} & {a['det']}/{N} & {a['fp']} & {a['miss']} & ${sd(a['median'])}$ [{sd(b['median_lag_ci95'][0])}, {sd(b['median_lag_ci95'][1])}] & ${sd(a['mean'])}$ [{sd(b['mean_lag_ci95'][0])}, {sd(b['mean_lag_ci95'][1])}] \\\\")
    S = [r"\begin{table}[t]", r"\centering",
         r"\caption{Detector performance over the 20 MUs with the parameters selected on all 20 MUs (in-sample, the reference configuration of the main text) and with the parameters selected in a nested leave-one-MU-out design (each MU read under the configuration selected on the other 19; Table~S3). Lags in days from the onset, with 95\% intervals from a bootstrap over MUs (10{,}000 resamples).}",
         r"\label{tab:nested}", r"\footnotesize", r"\begin{tabular}{llcccll}", r"\toprule",
         r"Read-out date & Parameters & On time & Early FP & Missed & Median lag [95\% CI] & Mean lag [95\% CI] \\", r"\midrule",
         line("Confirmation & in-sample", A["insample_confirmation"]), line(" & nested", A["nested_confirmation"]),
         line("Detection & in-sample", A["insample_detection"]), line(" & nested", A["nested_detection"]),
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR / "table5_nested_summary.tex").write_text("\n".join(S) + "\n", encoding="utf-8")

    st = op["stability"]
    print(f"wrote {OUT_DIR / 'table5_nested_summary.tex'} and {OUT_DIR / 'table_s3_nested_folds.tex'} | "
          f"folds re-selecting the reference: {st['folds_identical_to_insample']}/{N}; distinct configurations: {st['distinct_cfgs']}")
    for k in ("insample_confirmation", "nested_confirmation", "insample_detection", "nested_detection"):
        a = A[k]; b = a["bootstrap"]
        print(f"  {k:<22} on time {a['det']}/{N}  FP {a['fp']}  miss {a['miss']}  median {a['median']:+.1f} "
              f"[{b['median_lag_ci95'][0]:+.1f}, {b['median_lag_ci95'][1]:+.1f}]  mean {a['mean']:+.1f} "
              f"[{b['mean_lag_ci95'][0]:+.1f}, {b['mean_lag_ci95'][1]:+.1f}]")
