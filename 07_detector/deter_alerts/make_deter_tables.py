# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""DETER alerts inside each analysed period: the paper's Table S2 and the counts quoted in its text.

An alert counts for a period when its polygon intersects the site's box (deter_survey.py) and
its view date falls inside the period. Three groups of periods:
  * the 20 monitoring units: the record the models read, from the lookback window of the first
    observation of the monitoring period (79 observations earlier) to the view date that closes it;
  * the 77 logging-free candidates drawn as training negatives, and
  * the 51 targets of the negative test (07_detector/negative_test/select_targets.py):
    for both, the lookback window of the target (79 observations) plus the target.
A view date trails the disturbance it reports, so the counts approximate what each period holds.

Writes, under OUT_DIR:
  deter_alerts_by_period.csv           Table S2 as data: alerts per class for every period, with
                                       subtotals per group and the total;
  table_s2_deter_alerts.tex            Table S2 as LaTeX;
  deter_alerts_mu_records.csv          every alert inside the 20 monitoring-unit records;
  deter_alerts_target_lookbacks.csv    every alert inside the scored stretches of the 51 targets
                                       (flagging those in the certified 20-observation stretch);
and prints the summary counts quoted in the paper's Results. In the two listings,
on_own_view_date flags alerts sharing the view date of the alert the site was built from; that
alert cannot be matched by identifier, since DETER reassigns identifiers between downloads.

Usage: uv run python 07_detector/deter_alerts/make_deter_tables.py    (CPU; seconds)
Env: DATA_DIR (default ./data), ZARRS, ANNOTATIONS, VDJSON; OUT_DIR (default
     DATA_DIR/runs/deter_alerts); SURVEY (default OUT_DIR/deter_survey.csv, as deter_survey.py
     writes it; the record's copy, detector-evaluation/deter_alerts/deter_survey.csv, reproduces
     the paper without the shapefile); TARGETS (default DATA_DIR/runs/negative_test/targets.csv)."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_mu_numbers import MU_NUMBER                          # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
ZARRS = Path(os.environ["ZARRS"]) if os.environ.get("ZARRS") else DATA_DIR / "labelled-dataset"
ANNOTATIONS = Path(os.environ["ANNOTATIONS"]) if os.environ.get("ANNOTATIONS") else DATA_DIR / "annotations.json"
VDJSON = Path(os.environ["VDJSON"]) if os.environ.get("VDJSON") else DATA_DIR / "view_dates.json"
OUT_DIR = Path(os.environ["OUT_DIR"]) if os.environ.get("OUT_DIR") else DATA_DIR / "runs" / "deter_alerts"
SURVEY = Path(os.environ["SURVEY"]) if os.environ.get("SURVEY") else OUT_DIR / "deter_survey.csv"
TARGETS = Path(os.environ["TARGETS"]) if os.environ.get("TARGETS") else DATA_DIR / "runs" / "negative_test" / "targets.csv"
LW = 79                                                         # lookback observations before a target (L - 1)
CERTIFIED = 20                                                  # observations read logging-free before a target
CLASSES = ["CS_GEOMETRICO", "CS_DESORDENADO", "DEGRADACAO", "DESMATAMENTO_CR", "DESMATAMENTO_VEG", "CICATRIZ_DE_QUEIMADA", "MINERACAO"]
GROUPS = {"mu_records": "The 20 monitoring units: record = lookback window of the first monitored observation + monitored year",
          "training_negatives": "The 77 negative samples used in training: lookback window (79 observations) + target observation",
          "negative_test_targets": "The 51 certified targets never used in training (negative test, Table 3 of the main text): lookback window + target observation"}
_zc = {}


def zdates(aoi):
    """ISO dates of a site's observations, in store order."""
    if aoi not in _zc:
        _zc[aoi] = [str(np.datetime_as_string(x, unit="D")) for x in xr.open_zarr(ZARRS / f"{aoi}.zarr", consolidated=False)["time"].values]
    return _zc[aoi]


def iso(d8):
    return f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"


def target_index(aoi, d8, k):
    same = [i for i, d in enumerate(zdates(aoi)) if d == iso(d8)]
    return same[min(k, len(same)) - 1]


def count(sv, site, start, end):
    a = sv[(sv.site == site) & (sv.view_date >= start) & (sv.view_date <= end)]
    c = {cl: int((a.classname == cl).sum()) for cl in CLASSES}; c["total"] = len(a)
    return c


def period_rows(sv, vd, first_obs, pool, targets):
    rows = []
    for site in sorted(MU_NUMBER, key=MU_NUMBER.get):
        zt = zdates(site); i0 = zt.index(first_obs[site]); start = zt[max(0, i0 - LW)]; end = vd[site]
        rows.append(dict(group="mu_records", period=f"MU{MU_NUMBER[site]}", site=site, start=start, end=end,
                         n_obs=len([d for d in zt if start <= d <= end]), **count(sv, site, start, end)))
    for grp, items in (("training_negatives", sorted(c for c in pool if c not in targets)), ("negative_test_targets", sorted(targets))):
        for aoi, d8, k in items:
            zt = zdates(aoi); t = target_index(aoi, d8, k); start = zt[max(0, t - LW)]
            rows.append(dict(group=grp, period=f"{aoi.split('_')[0]} @ {iso(d8)} (k{k})", site=aoi, start=start, end=iso(d8),
                             n_obs=t - max(0, t - LW) + 1, **count(sv, aoi, start, iso(d8))))
    df = pd.DataFrame(rows); out = []
    for grp, sub in df.groupby("group", sort=False):
        out.append(sub)
        tot = {cl: int(sub[cl].sum()) for cl in CLASSES}; tot["total"] = int(sub.total.sum())
        out.append(pd.DataFrame([dict(group=grp, period="SUBTOTAL", site=f"{sub.site.nunique()} sites, {len(sub)} periods, {int((sub.total > 0).sum())} with alerts",
                                      start=sub.start.min(), end=sub.end.max(), n_obs=int(sub.n_obs.sum()), **tot)]))
    tot = {cl: int(df[cl].sum()) for cl in CLASSES}; tot["total"] = int(df.total.sum())
    out.append(pd.DataFrame([dict(group="all", period="TOTAL", site=f"{df.site.nunique()} sites, {len(df)} periods, {int((df.total > 0).sum())} with alerts",
                                  start=df.start.min(), end=df.end.max(), n_obs=int(df.n_obs.sum()), **tot)]))
    return pd.concat(out, ignore_index=True)


def latex(table):
    def label(r):
        if r["group"] == "mu_records":
            return "MU$_{%s}$" % r["period"][2:]
        site, d = r["period"].split(" @ "); d, k = d.split(" (k"); k = k.rstrip(")")
        return f"{site}, {d}" + (f" ({k})" if k != "1" else "")

    def bold(r, name):
        return (r"\textbf{%s} & \multicolumn{2}{l}{\textbf{%s}} & \textbf{%s} & " % (name, r["site"], r["n_obs"])
                + " & ".join(r"\textbf{%s}" % r[c] for c in CLASSES) + r" & \textbf{%s} \\" % r["total"])
    out = [r"""\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{0.9}\scriptsize
\begin{longtable}{l l l r rrrrrrr r}
\caption{DETER alerts of every class whose polygon intersects the site's 1,920~m box and whose view date falls inside each analysed period, from the public DETER Amaz\^onia record (download of 15 September 2026). Three groups of periods, each analysed by the same rule: the records of the 20 monitoring units (main text, Sec.~\ref{sec:separability}), the 77 negative samples that trained the regressor and the 51 interpreter-certified targets held out of training (main text, Table~\ref{tab:negtest}); for the last two, the period is the 79-observation lookback window plus the target observation. Obs.: observations in the period. Classes: CS-G, geometric selective logging; CS-D, disordered selective logging; DEG, degradation; CR, clear-cut deforestation; VEG, deforestation with vegetation; FIRE, burn scar; MIN, mining. A view date trails the disturbance it reports, so the counts approximate what each period contains; whether an event fell inside the period was not verified against the imagery. Subtotals in bold; in the subtotal rows the second column gives the number of sites, periods and periods with at least one alert.}
\label{tab:deter_alerts}\\
\toprule
Period & Start & End & Obs. & CS-G & CS-D & DEG & CR & VEG & FIRE & MIN & Total \\
\midrule
\endfirsthead
\multicolumn{12}{l}{\footnotesize Table~\thetable, continued.}\\
\toprule
Period & Start & End & Obs. & CS-G & CS-D & DEG & CR & VEG & FIRE & MIN & Total \\
\midrule
\endhead
\midrule \multicolumn{12}{r}{\footnotesize continued on the next page}\\
\endfoot
\bottomrule
\endlastfoot"""]
    cur = None
    for _, r in table.iterrows():
        if r["period"] == "TOTAL":
            out += [r"\midrule", bold(r, "All periods")]; continue
        if r["period"] == "SUBTOTAL":
            out += [r"\midrule", bold(r, "Subtotal"), r"\midrule"]; continue
        if r["group"] != cur:
            cur = r["group"]; out += [r"\multicolumn{12}{l}{\textit{%s}}\\" % GROUPS[cur], r"\midrule"]
        out.append(f"{label(r)} & {r['start']} & {r['end']} & {r['n_obs']} & " + " & ".join(str(r[c]) for c in CLASSES) + f" & {r['total']} \\\\")
    out.append(r"\end{longtable}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ann = json.loads(ANNOTATIONS.read_text(encoding="utf-8")); vd = json.loads(VDJSON.read_text(encoding="utf-8"))
    first_obs = {s: iso(d.replace("-", "")) for s, d in ann["first_observation"].items()}
    onset = {s: iso(min(d.replace("-", "") for d, _ in v)) for s, v in ann["logging_activity"].items()}
    pool = [(r, d.replace("-", ""), int(k)) for r, d, k in ann["oversample_pool"]]
    targets = {(r["aoi_id"], r["date"], int(r["k"])) for r in csv.DictReader(open(TARGETS, encoding="utf-8"))}
    sv = pd.read_csv(SURVEY, dtype={"site": str, "view_date": str, "classname": str})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = period_rows(sv, vd, first_obs, pool, targets)
    table.to_csv(OUT_DIR / "deter_alerts_by_period.csv", index=False, lineterminator="\n")
    (OUT_DIR / "table_s2_deter_alerts.tex").write_text(latex(table), encoding="utf-8")

    # every alert inside the 20 monitoring-unit records
    mu_rows = []
    for site in sorted(MU_NUMBER, key=MU_NUMBER.get):
        zt = zdates(site); start = zt[max(0, zt.index(first_obs[site]) - LW)]
        a = sv[(sv.site == site) & (sv.view_date >= start) & (sv.view_date <= vd[site])]
        for _, al in a.iterrows():
            mu_rows.append(dict(mu=f"MU{MU_NUMBER[site]}", site=site, classname=al.classname, view_date=al.view_date,
                                where="lookback before the monitoring period" if al.view_date < first_obs[site] else "monitoring period",
                                days_from_onset=(pd.Timestamp(al.view_date) - pd.Timestamp(onset[site])).days,
                                on_own_view_date="yes" if al.view_date == vd[site] else "", box_share=f"{al.frac_box:.1%}", area_km2=round(al.area_km2, 2)))
    mu = pd.DataFrame(mu_rows); mu.to_csv(OUT_DIR / "deter_alerts_mu_records.csv", index=False, lineterminator="\n")

    # every alert inside the scored stretch of each negative-test target (79 before .. target)
    tg_rows = []; n_with = n_in_cert = 0
    for aoi, d8, k in sorted(targets):
        zt = zdates(aoi); t = target_index(aoi, d8, k); w0 = zt[max(LW, t - LW)]; c0 = zt[t - (CERTIFIED - 1)]
        a = sv[(sv.site == aoi) & (sv.view_date >= w0) & (sv.view_date <= iso(d8))]
        n_with += len(a) > 0; n_in_cert += bool((a.view_date >= c0).any())
        for _, al in a.iterrows():
            tg_rows.append(dict(target=f"{aoi.split('_')[0]} @ {iso(d8)} (k{k})", site=aoi, classname=al.classname, view_date=al.view_date,
                                days_to_target=(pd.Timestamp(al.view_date) - pd.Timestamp(iso(d8))).days,
                                in_certified_stretch="yes" if al.view_date >= c0 else "", on_own_view_date="yes" if al.view_date == vd[aoi] else "",
                                box_share=f"{al.frac_box:.1%}", area_km2=round(al.area_km2, 2)))
    tg = pd.DataFrame(tg_rows); tg.to_csv(OUT_DIR / "deter_alerts_target_lookbacks.csv", index=False, lineterminator="\n")

    cs = mu.classname.str.startswith("CS_"); other = mu[~cs]
    per_mu = mu.groupby("mu").size()
    own = mu[cs & (mu.on_own_view_date == "yes")].mu.nunique()   # units whose own selective-logging alert is in the record
    print(f"20 monitoring-unit records: {len(mu)} alerts, {int((per_mu >= 2).sum())} units with two or more; "
          f"{int(cs.sum())} of selective logging, {int(cs.sum()) - own} beyond the alert that defined each unit ({own} units); "
          f"{len(other)} of other classes on {other.mu.nunique()} units ({', '.join(sorted(other.mu.unique(), key=lambda m: int(m[2:])))}), "
          f"{int((other.days_from_onset < 0).sum())} with view dates before the onset; "
          f"largest box share of another class: {max(float(s.rstrip('%')) for s in other.box_share):.1f}%")
    tcs = tg.classname.str.startswith("CS_")
    print(f"51 negative-test targets: {len(tg)} alerts in the lookback windows of {n_with} targets "
          f"({int(tcs.sum())} of selective logging, {int((~tcs).sum())} of other classes); "
          f"{int((tg.in_certified_stretch == 'yes').sum())} alerts on {n_in_cert} targets inside the certified {CERTIFIED}-observation stretch")
    print(f"wrote deter_alerts_by_period.csv, table_s2_deter_alerts.tex, deter_alerts_mu_records.csv and deter_alerts_target_lookbacks.csv under {OUT_DIR}")
