#!/usr/bin/env python3
"""
Cross-document consistency check for the second revision.

Verifies that the canonical numbers (from the results JSONs) appear in the
manuscript, appendix, response letter, and tables, and flags stale numbers from
the prior version that must not appear in the revised text.
"""
import json
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
NB = Path(__file__).resolve().parents[4] / "notebooks" / "panelfm" / "revision2"
RES = PKG / "revision2" / "results"

STEM = "panelfm_medicalcare_"  # descriptive topic_venue prefix for all deliverables
DOCS = {
    "manuscript": NB / "docs" / f"{STEM}manuscript_bold.md",
    "appendix": NB / "docs" / f"{STEM}supplementary_appendix.md",
    "response": NB / "docs" / f"{STEM}response_to_reviewers.md",
}
TEXT = {k: p.read_text() for k, p in DOCS.items() if p.exists()}
ALLTEXT = "\n".join(TEXT.values())

m = json.loads((RES / "all_metrics_disjoint.json").read_text())
ci = json.loads((RES / "bootstrap_ci_disjoint.json").read_text())
sp = json.loads((RES / "split_info_disjoint.json").read_text())
st = json.loads((RES / "stratified_mae_disjoint.json").read_text())

problems, checks = [], []


def want(label, needle, where="any"):
    targets = TEXT if where == "any" else {where: TEXT[where]}
    found = any(needle in t for t in targets.values())
    checks.append((label, needle, found))
    if not found:
        problems.append(f"MISSING: {label} -> expected '{needle}'")


def forbid(label, needle, allow=()):
    # `allow` lists documents where the needle is legitimately permitted
    # (e.g., the response letter may cite the prior version's numbers to explain a correction).
    hits = [k for k, t in TEXT.items() if needle in t and k not in allow]
    checks.append((f"STALE {label}", needle, not hits))
    if hits:
        problems.append(f"STALE: {label} -> '{needle}' found in {hits}")


# --- Canonical cohort / split numbers ---
want("cohort members", "122,849")
want("patient-months", "2,392,363")
want("zero-cost months", "61.4%")
want("eligible members", "57,367")
want("train members", "40,158")
want("val members", "8,605")
want("test members", "8,604")

# --- Headline model numbers ---
want("chronos MAE", "448")
want("chronos MAE CI low", "409")
want("chronos MAE CI high", "491")
want("chronos R2cal", "−1.11")  # note: en-dash minus used in prose
want("chronos PR", "0.48")
want("best CS MAE (stacking)", "593")
want("MAE reduction ~24%", "24%")
want("CS R2 range low", "0.17")
want("CS R2 range high", "0.28")
want("gated hybrid MAE", "511")
want("gated hybrid MAE CI", "473")
want("gated hybrid R2", "0.19")
want("gated hybrid PR", "0.75")
want("calibration factor", "1.054")
want("stratified chronos zero", "$7")
want("concurrent R2 high", "0.80")

# --- Second-round additions: foundation-model panel, decision analysis, recalibration, temporal split ---
want("TimesFM lowest MAE", "$420")
want("panel PR range low", "0.47")
want("panel PR range high", "0.80")
want("TabPFN over-prediction", "1.64")
want("tweedie MAE row", "| Tweedie (mean-target) | CS | 543 |", where="appendix")
want("quantile-median PR", "0.57")
want("cost-captured recall range low", "0.58")
want("cost-captured recall range high", "0.68")
want("recalibration chronos before", "$1,272", where="appendix")
want("recalibration chronos after", "$1,513", where="appendix")
want("recalibration tabpfn before", "$2,151", where="appendix")
want("temporal chronos MAE", "$476", where="appendix")
want("Tweedie reference 23", "Zhou H, Qian W, Yang Y", where="manuscript")
want("TabPFN reference 24", "Hollmann N", where="manuscript")
want("DCA reference 25", "Vickers AJ, Elkin EB", where="manuscript")
want("CRPS reference 26", "Gneiting T, Raftery AE", where="manuscript")
want("supp table S11", "Table S11")
want("supp table S12", "Table S12")
want("supp figure S5", "Figure S5")

# Forbid the superseded subset-based gated-hybrid / chronos numbers (full-set canonical is 511 / −1.11)
forbid("subset gated hybrid $513", "$513")
# (removed) the −1.08 forbid collided with the legitimate XGBoost entry-cohort temporal R² (−1.08); chronos −1.11 is enforced by want() above.

# --- Numbers must match the JSON to one decimal / integer ---
def approx_in(val, txt, tol=0.6):
    # check an integer near val appears
    return str(int(round(val))) in txt

assert abs(m["chronos_zeroshot"]["mae"] - 448.0) < 1, "chronos MAE drift"
assert abs(m["hybrid_gated"]["mae"] - 510.9) < 1, "gated MAE drift"
assert sp["n_test_members"] == 8604, "test n drift"

# --- 12-month-horizon sensitivity (CY2024 -> CY2025); validated against all_metrics_12mo.json ---
if (RES / "all_metrics_12mo.json").exists():
    m12 = json.loads((RES / "all_metrics_12mo.json").read_text())
    sp12 = json.loads((RES / "split_info_12mo.json").read_text())
    assert abs(m12["chronos_zeroshot"]["mae"] - 531.0) < 1.5, "12mo chronos MAE drift"
    assert abs(m12["random_forest"]["r_squared_calibrated"] - 0.409) < 0.01, "12mo RF R2cal drift"
    assert sp12["n_test_members"] == 5025 and sp12["n_eligible_members"] == 33484, "12mo n drift"
    want("12mo chronos MAE+CI", "531 (486–580)", where="appendix")
    want("12mo RF R2cal+CI", "0.41 (0.34 to 0.48)", where="appendix")
    want("12mo eligible n", "33,484", where="appendix")
    want("12mo RF in Results", "random forest calibrated R² 0.41", where="manuscript")
    want("12mo Table S13 ref", "Table S13", where="appendix")

# --- Stale prior-version numbers that must NOT appear ---
forbid("old 76% reduction", "76%", allow=("response",))
forbid("old chronos MAE 324", "$324")
forbid("old hybrid MAE 288", "$288")
forbid("old two-part total MAE 1,342", "1,342")
forbid("old test n 64,141", "64,141")
forbid("old test n 64,047", "64,047")
forbid("old hybrid R2 0.96 PR", "0.96")  # old predictive ratio claim
forbid("old stratified 30,780", "30,780")
forbid("temporal-only 'patients typically appear'", "patients typically appear in both")

print("=== CONSISTENCY CHECK ===")
for label, needle, ok in checks:
    print(f"  [{'OK' if ok else 'XX'}] {label}: '{needle}'")
print()
if problems:
    print(f"{len(problems)} PROBLEM(S):")
    for p in problems:
        print("  -", p)
else:
    print("ALL CHECKS PASSED")
