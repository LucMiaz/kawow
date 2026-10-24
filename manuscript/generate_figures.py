"""generate_figures.py
=====================
Figures of the manuscript "A Vanishing Partitioning Gap" and its SI, for the
five benchmarked models (Crippen, Naef & Acree, combined Crippen+Naef & Acree
ridge, Random Forest, XGBoost).

Written to outputs/figures/ (pdf, png, svg):

  fig1_regulatory_space          Figure 1  (_clp: SI Figure S3)
  fig8_gap_zoom_models           Figure 2  (_clp: SI Figure S4)
  fig10_clinventory_gap_zoom     Figure 3  (_clp: SI Figure S5)
  fig9_arphale_gap_persistence   Figure 4  (_clp: SI Figure S6)
  fig7_yrandomization            SI Figure S1
  fig3_model_comparison          SI Figure S2

The figures that depend on the M/vM thresholds are drawn once per entry of
MOBILITY_CRITERIA: UBA criteria (no suffix) and EU CLP criteria ("_clp").
The thresholds themselves are taken from kawow.criteria.

Inputs: tests/out/benchmark_*_5model.csv (scripts/benchmark_5model.py),
cache/ (build_deployment_predictions.py and yrandomization.py) and
data/arphale_hale_2022_lser.csv. See README.md.

Usage:
    python generate_figures.py
"""

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import paths  # noqa: E402  (also puts the kawow repository on sys.path)
from kawow.criteria import CRITERIA as KAWOW_CRITERIA  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT = paths.FIGURES
BENCHMARK_METRICS_CSV = paths.BENCHMARK_METRICS_CSV
BENCHMARK_PRED_CSV = paths.BENCHMARK_PRED_CSV
YRAND_RESULTS_CSV = paths.CACHE / "yrandomization_results.csv"
ARPHALE_PRED_CSV = paths.CACHE / "arphale_predictions.csv"
ARPHALE_NAEF_EXP_CSV = paths.CACHE / "arphale_naef_experimental.csv"
CROSS_ENDPOINT_PRED_CSV = paths.CACHE / "cross_endpoint_predictions.csv"
CLINVENTORY_PRED_CSV = paths.CACHE / "clinventory_predictions_5model.csv"

# ── Style ─────────────────────────────────────────────────────────────────────
# Journal figure spec: every figure is exactly FIG_WIDTH_IN wide (180 mm);
# text sizes are snapped to one of four discrete values -- FS_FLOOR (6 pt) is
# an absolute minimum reserved for the densest, most crowded overlays only;
# FS_LABEL (8 pt) is the normal size for tick labels and in-plot annotations;
# FS_LABEL_HI (10 pt) is used for more prominent in-plot/subplot titles and
# doubles as the low end of the axis-title/legend band; FS_TITLE (12 pt) is
# the high end of that band, used for axis titles, legends, and figure-level
# (sup)titles. Nothing in this file should use a size outside this set.
FIG_WIDTH_MM = 180.0
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
FS_FLOOR    = 6
FS_LABEL    = 8
FS_LABEL_HI = 10
FS_TITLE    = 12

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        FS_LABEL,
    "axes.labelsize":   FS_LABEL_HI,
    "axes.titlesize":   FS_LABEL_HI,
    "legend.fontsize":  FS_LABEL_HI,
    "xtick.labelsize":  FS_LABEL,
    "ytick.labelsize":  FS_LABEL,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# ── Wes Anderson colour palettes ────────────────────────────────────────────
# From https://github.com/hneutr/wesanderson (mirrors the R wesanderson package).
# Every colour used in this file's figures is drawn from one of these five
# (each a single named sub-palette; Isle of Dogs and Darjeeling Limited each
# have a second, unused sub-palette; Moonrise Kingdom has two more unused).
CASTELLO_CAVALCANTI = ["#D8B70A", "#02401B", "#A2A475", "#81A88D", "#972D15"]
HOTEL_CHEVALIER     = ["#446455", "#FDD262", "#D3DDDC", "#C7B19C"]
ISLE_OF_DOGS_1      = ["#9986A5", "#79402E", "#CCBA72", "#0F0D0E", "#D9D0D3", "#8D8680"]
DARJEELING_1        = ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"]
MOONRISE_1          = ["#F3DF6C", "#CEAB07", "#D5D5D3", "#24281A"]


def _tint(hexcode, white_frac):
    """Blend a palette colour toward white -- used to derive pale background
    fills (region shading) from the saturated palette swatches above, since
    the palettes themselves are all bold/saturated."""
    hexcode = hexcode.lstrip("#")
    r, g, b = int(hexcode[0:2], 16), int(hexcode[2:4], 16), int(hexcode[4:6], 16)
    r = round(r + (255 - r) * white_frac)
    g = round(g + (255 - g) * white_frac)
    b = round(b + (255 - b) * white_frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken(hexcode, black_frac):
    """Blend a palette colour toward black -- used for boundary lines/labels
    drawn directly on the page, which need more contrast than the palette
    swatch itself (e.g. the pale Darjeeling sky-blue is too light to read as
    axis text) while staying in the same hue family."""
    hexcode = hexcode.lstrip("#")
    r, g, b = int(hexcode[0:2], 16), int(hexcode[2:4], 16), int(hexcode[4:6], 16)
    r = round(r * (1 - black_frac))
    g = round(g * (1 - black_frac))
    b = round(b * (1 - black_frac))
    return f"#{r:02x}{g:02x}{b:02x}"


# Region fill colours (light, distinguishable in greyscale) -- pale tints of
# the palette swatches below, since the region fills sit *under* the scatter
# points and must stay pale regardless of how saturated the source colour is.
C_M_ONLY  = _tint(DARJEELING_1[4], 0.65)    # sky-blue (5BBCD6), light
C_VM      = _tint(DARJEELING_1[4], 0.30)    # sky-blue (5BBCD6), more saturated
C_M_MARGIN = _tint(DARJEELING_1[4], 0.90)   # sky-blue (5BBCD6), very pale -- "Potential M/vM" screening margin
C_B_ONLY  = _tint(CASTELLO_CAVALCANTI[4], 0.78)  # brick-red (972D15), light -- B, terrestrial (Kelly) route only
C_VB      = _tint(CASTELLO_CAVALCANTI[4], 0.45)  # brick-red (972D15), more saturated -- vB, both routes agree
C_B_BOTH  = _tint(CASTELLO_CAVALCANTI[4], 0.60)  # brick-red (972D15), mid-tint -- B, both terrestrial and aquatic routes
C_B_AQUATIC  = "#CEA567"                     # gold -- B, aquatic (BCF) route only, not reachable via Kelly
C_VB_AQUATIC = _darken("#CEA567", 0.30)      # darker gold -- vB, aquatic (BCF) route only
C_BM      = _tint(ISLE_OF_DOGS_1[0], 0.65)  # purple-grey (9986A5), light
C_GAP     = _tint(CASTELLO_CAVALCANTI[0], 0.75)  # mustard (D8B70A), pale yellow
C_NON_BM  = _tint(MOONRISE_1[2], 0.55)      # pale grey (D5D5D3), near-neutral
C_EDGE    = MOONRISE_1[3]                    # near-black (24281A), boundary lines/text
C_BG      = _tint(MOONRISE_1[2], 0.85)      # pale grey (D5D5D3), panel background
C_MUTED   = MOONRISE_1[2]                    # pale grey (D5D5D3), generic light grey

# Boundary-line and label colours (fig1, fig8, fig9) -- darkened
# versions of the same swatches used for the region fills above, so a given
# hazard family (B/vB red, M/vM blue, B+M purple) reads as one consistent
# hue between its pale background fill and its boundary line/label text.
C_B_LINE   = CASTELLO_CAVALCANTI[4]          # brick-red (972D15), B/vB
C_VB_LINE  = _darken(CASTELLO_CAVALCANTI[4], 0.30)  # darker brick-red, vB
C_M_LINE   = _darken(DARJEELING_1[4], 0.45)  # dark sky-blue, M/vM
C_BM_LABEL = _darken(ISLE_OF_DOGS_1[0], 0.35)  # dark purple-grey, B+M overlap
C_GAP_LABEL = _darken(CASTELLO_CAVALCANTI[2], 0.20)  # dark sage, "gap" label
C_GAP_EDGE  = CASTELLO_CAVALCANTI[0]         # mustard (D8B70A), gap hatch edge

# "Into gap" / "of concern" red vs. "not in gap" / "unremarkable" blue, used
# consistently across fig7 (observed R2) and fig8 (predicted gap membership).
C_HIGHLIGHT = DARJEELING_1[0]  # red (FF0000)
C_CALM      = DARJEELING_1[4]  # sky-blue (5BBCD6)
# fig10 (C&L Inventory) uses its own "into gap" red, distinct from fig8's
# C_HIGHLIGHT (pure red) and from C_B_LINE (brick-red, already the B/vB
# boundary-line colour) -- a rust/maroon so the two "gap" figures aren't
# visually identical and this one doesn't collide with the B-line hue.
C_HIGHLIGHT_ALT = ISLE_OF_DOGS_1[1]  # rust/maroon (79402E)

# ── Regulatory thresholds ─────────────────────────────────────────────────────
# M/vM (via logKow proxy): Arp & Hale (2019, UBA TEXTE 126/2019) empirically-
# validated screening zones (their Fig. 9, neutral-compound branch), not a
# fixed Koc=c*Kow conversion. Not-M above M_KOW; "Potential M/vM" -- their own
# built-in screening margin -- between M_POTENTIAL_KOW and M_KOW; "M/vM"
# (mobile, unclear if very) between VM_KOW and M_POTENTIAL_KOW; confident vM
# below VM_KOW. The raw logKoc criteria (M<4.0, vM<3.0) are unchanged
# (Neumann & Schliebner 2019).
#
# The final EU CLP criteria (Delegated Regulation (EU) 2023/707) are stricter,
# logKoc<3.0 for M and <2.0 for vM (pH 4-9). No logKow screening scheme has
# been validated for these cutoffs, so the Arp & Hale zones are shifted down
# by the same 1.0 log-unit as the logKoc cutoffs: logKoc-logKow regressions
# have a slope of ~1 (Koc~0.35*Kow, Seth et al. 1999, gives logKow~3.5 for
# the CLP M cutoff as well). M_KOW, M_POTENTIAL_KOW, VM_KOW, GAP3_KOW_MIN and
# FIG_SUFFIX are set by use_mobility_criteria() from the kawow.criteria preset
# of the same name (both presets are defined in kawow, not here).
# All thresholds come from kawow.criteria (presets "uba" and "clp"); only
# the plotting settings below are specific to this script.
B_KOW  = KAWOW_CRITERIA["uba"].b_logkow_threshold
VB_KOW = KAWOW_CRITERIA["uba"].vb_logkow_threshold
B_KOA  = KAWOW_CRITERIA["uba"].b_logkoa_threshold
# "zoom" is the (kow_lim, kaw_lim) window of the gap close-ups (fig8, fig10):
# the CLP gap reaches down to logKaw=-2.5 at logKow=3.5, so its window is
# larger. "legend_loc" places the fig9 legend; None (CLP, where the gap
# molecules fill most of the panel) replaces it by counts on the bar panels.
MOBILITY_CRITERIA = {
    "uba": {"name": "UBA", "suffix": "", "title_tag": "",
            "zoom": ((3.0, 7.0), (-2.0, 2.0)), "legend_loc": "upper left"},
    "clp": {"name": "CLP", "suffix": "_clp",
            "title_tag": r"CLP mobility criteria (M: $\log K_{oc}<3$, vM: $\log K_{oc}<2$)",
            "zoom": ((2.5, 7.5), (-3.0, 2.0)), "legend_loc": None},
}

# Standard aquatic (fish) bioaccumulation, logKow-only, no logKoa requirement:
# BCF ~ lipid_fraction * Kow (I. Cousins, personal communication, 2026),
# lipid_fraction=0.05. B at BCF>2,000 -> logKow>4.602; vB at BCF>5,000 ->
# logKow>5.0 -- the same numeric value as VB_KOW above, by coincidence of two
# independent B/vB routes (Kelly's air-breather logKoa-based route above, and
# this Kow-only aquatic route), not a shared derivation. A chemical that
# fails Kelly's air-breather test (logKoa<6) can still be B/vB via this
# route regardless of logKoa -- see _draw_aquatic_bcf_overlay(). This is why
# the true "neither B nor M" gap is much narrower than a Kelly-only picture
# suggests: it ends where the aquatic route alone already makes a chemical
# B, not at VB_KOW (Ian Cousins, Slack 2026-04-28: "The gap disappears
# almost entirely when including aquatic bioaccumulation!").
B_KOW_AQUATIC = KAWOW_CRITERIA["uba"].b_aquatic_logkow_threshold   # ~4.602
GAP3_KOW_MAX = B_KOW_AQUATIC


def use_mobility_criteria(key):
    """Point the module-level M/vM thresholds, the lower gap bound and the
    output file suffix at one entry of MOBILITY_CRITERIA."""
    global MOBILITY, M_KOW, M_POTENTIAL_KOW, VM_KOW, GAP3_KOW_MIN, FIG_SUFFIX
    MOBILITY = MOBILITY_CRITERIA[key]
    crit = KAWOW_CRITERIA[key]
    M_KOW = crit.m_logkow_threshold
    M_POTENTIAL_KOW = crit.m_potential_logkow_threshold
    VM_KOW = crit.vm_logkow_threshold
    GAP3_KOW_MIN = M_KOW
    FIG_SUFFIX = MOBILITY["suffix"]


def _title_tag():
    """Extra title line naming the mobility criteria (empty for UBA, so the
    original figures are unchanged)."""
    return f"\n{MOBILITY['title_tag']}" if MOBILITY["title_tag"] else ""


use_mobility_criteria("uba")

# ── Model display metadata (5 models) ─────────────────────────────────────────
MODEL_ORDER = ["crippen", "naefacree", "naefacree_mixed", "naef_crippen_rf", "naef_crippen_xgb"]
MODEL_LABELS = {
    "crippen":          "Crippen",
    "naefacree":        "Naef & Acree",
    "naefacree_mixed":  "Crippen+Naef & Acree",
    "naef_crippen_rf":  "Random Forest",
    "naef_crippen_xgb": "XGBoost",
}
# Isle of Dogs, palette 1.
MODEL_COLORS = {
    "crippen":          ISLE_OF_DOGS_1[0],  # purple-grey
    "naefacree":        ISLE_OF_DOGS_1[1],  # brown
    "naefacree_mixed":  ISLE_OF_DOGS_1[2],  # khaki
    "naef_crippen_rf":  ISLE_OF_DOGS_1[5],  # taupe-grey
    "naef_crippen_xgb": ISLE_OF_DOGS_1[3],  # near-black
}

# Data-provenance colours (fig3): which endpoint(s) are genuinely measured for
# a given molecule, vs. filled in via cross-inference -- see
# load_benchmark_predictions_all() and build_deployment_predictions.py.
# Castello Cavalcanti, mainly the first two and last swatches.
CATEGORY_COLORS = {
    "both":         CASTELLO_CAVALCANTI[4],  # brick-red
    "logKow_only":  CASTELLO_CAVALCANTI[0],  # mustard
    "logKoa_only":  CASTELLO_CAVALCANTI[1],  # dark green
}
CATEGORY_LABELS = {
    "both":         "Both endpoints measured",
    "logKow_only":  r"Only $\log K_{ow}$ measured ($\log K_{oa}$ modelled)",
    "logKoa_only":  r"Only $\log K_{oa}$ measured ($\log K_{ow}$ modelled)",
}

# Persistence buckets (fig9): vP (worst) -> P -> Potential P -> Not P
# (best) -> no data. Darjeeling Limited 1 for the 4-step concern gradient,
# Moonrise Kingdom 1 for the neutral "no data" grey.
BUCKET_COLORS = {
    "vP":          DARJEELING_1[0],  # red
    "P":           DARJEELING_1[3],  # orange
    "Potential P": DARJEELING_1[2],  # mustard
    "Not P":       DARJEELING_1[4],  # sky-blue
    "no data":     MOONRISE_1[2],    # pale grey
}


def load_benchmark_metrics():
    df = pd.read_csv(BENCHMARK_METRICS_CSV)
    return df


def _read_benchmark_predictions():
    """benchmark_predictions_5model.csv as one 'pred_<model>' column per
    model. Since kawow ee2dc6d the file stores each endpoint's predictions
    in endpoint-suffixed columns (pred_<model>_kow / pred_<model>_koa, NaN on
    the other endpoint's rows); they are folded back into one column here."""
    df = pd.read_csv(BENCHMARK_PRED_CSV)
    for mkey in MODEL_ORDER:
        if f"pred_{mkey}" not in df.columns:
            df[f"pred_{mkey}"] = df[f"pred_{mkey}_kow"].fillna(df[f"pred_{mkey}_koa"])
    return df.drop(columns=[c for c in df.columns if c.endswith(("_kow", "_koa")) and c.startswith("pred_")])


def load_benchmark_predictions_all():
    """Return merged per-molecule dataframe, OUTER join: keeps every
    molecule with AT LEAST ONE of logKow/logKoa measured, not just the
    ~1,100 with both. For a molecule missing one endpoint's experimental
    value, that endpoint's model prediction is filled in from
    cross_endpoint_predictions.csv -- a FINAL model fit on the full
    opposite-endpoint dataset and applied to this molecule (same "final
    model, external inference" pattern already used for the Arp & Hale
    dataset; see build_deployment_predictions.py). Adds a 'category' column:
    'both' / 'logKow_only' / 'logKoa_only'."""
    df = _read_benchmark_predictions()
    s01 = df[df["dataset"] == "logKow"].drop_duplicates("inchikey").copy()
    s02 = df[df["dataset"] == "logKoa"].drop_duplicates("inchikey").copy()
    pred_cols = [c for c in df.columns if c.startswith("pred_")]

    m = pd.merge(
        s01[["inchikey", "exp_value"] + pred_cols].rename(columns={"exp_value": "logKow_exp"}),
        s02[["inchikey", "exp_value"] + pred_cols].rename(columns={"exp_value": "logKoa_exp"}),
        on="inchikey", suffixes=("_kow", "_koa"), how="outer",
    )
    m["category"] = np.where(
        m["logKow_exp"].notna() & m["logKoa_exp"].notna(), "both",
        np.where(m["logKow_exp"].notna(), "logKow_only", "logKoa_only"))
    m["logKaw_exp"] = m["logKow_exp"] - m["logKoa_exp"]

    cross = pd.read_csv(paths.require_cache(CROSS_ENDPOINT_PRED_CSV.name, "build_deployment_predictions.py"))
    cross_kow_only = (cross[cross["category"] == "logKow_only"].drop_duplicates("inchikey").set_index("inchikey")
                       if len(cross) else pd.DataFrame())
    cross_koa_only = (cross[cross["category"] == "logKoa_only"].drop_duplicates("inchikey").set_index("inchikey")
                       if len(cross) else pd.DataFrame())

    for mkey in MODEL_ORDER:
        kow_c = f"pred_{mkey}_kow"
        koa_c = f"pred_{mkey}_koa"
        m[f"{mkey}_logKow"] = m[kow_c]
        m[f"{mkey}_logKoa"] = m[koa_c]
        fill_col_koa = f"cross_pred_{mkey}_logKoa"
        if fill_col_koa in cross_kow_only.columns:
            m[f"{mkey}_logKoa"] = m[f"{mkey}_logKoa"].fillna(m["inchikey"].map(cross_kow_only[fill_col_koa]))
        fill_col_kow = f"cross_pred_{mkey}_logKow"
        if fill_col_kow in cross_koa_only.columns:
            m[f"{mkey}_logKow"] = m[f"{mkey}_logKow"].fillna(m["inchikey"].map(cross_koa_only[fill_col_kow]))
        m[f"{mkey}_logKaw"] = m[f"{mkey}_logKow"] - m[f"{mkey}_logKoa"]
    return m


def load_arphale_persistence():
    """Arp & Hale REACH dataset: logKow/logKoa + a simplified 4-way persistence
    bucket derived from the UBA PMT/vPvM screening category (not a strict
    formal P/vP determination -- see figure caption). Bucketing logic in
    _bucket_uba(), defined below.

    The 'logKow'/'logKoa' columns of data/arphale_hale_2022_lser.csv are UFZ
    linear-solvation-energy-relationship (LSER) estimates, not experimental
    values (see README.md, "Data")."""
    df = pd.read_csv(paths.ARPHALE_CSV)
    df = df.dropna(subset=["logKow", "logKoa"]).copy()
    df["p_bucket"] = df["vPvM_UBA"].apply(_bucket_uba)
    return df


def load_arphale_naef_experimental():
    """Arp & Hale REACH molecules with a genuine EXPERIMENTAL logKow (Naef
    S01) and logKoa (Naef S02) pair, matched by RDKit-recomputed InChIKey
    (built by build_deployment_predictions.py). This subset is much smaller
    (n=703) than the LSER-covered set (n=11,318) in
    load_arphale_persistence(), but not model-derived at all."""
    df = pd.read_csv(paths.require_cache(ARPHALE_NAEF_EXP_CSV.name, "build_deployment_predictions.py"))
    df["p_bucket"] = df["vPvM_UBA"].apply(_bucket_uba)
    return df


def load_arphale_model_predictions():
    """Predictions from final (non-CV) crippen/naefacree/naefacree_mixed models,
    fit on the full S01/S02 training data, applied to Arp & Hale SMILES
    (built by build_deployment_predictions.py). Adds the same p_bucket as
    load_arphale_persistence() (see _bucket_uba, defined below)."""
    df = pd.read_csv(paths.require_cache(ARPHALE_PRED_CSV.name, "build_deployment_predictions.py"))
    df["p_bucket"] = df["vPvM_UBA"].apply(_bucket_uba)
    return df


def load_yrandomization():
    results = pd.read_csv(YRAND_RESULTS_CSV)
    null_dists = {}
    for _, row in results.iterrows():
        key = (row["model"], row["endpoint"])
        null_csv = paths.CACHE / f"yrandomization_null_{row['model']}_{row['endpoint']}.csv"
        null_dists[key] = pd.read_csv(null_csv)["perm_r2"].values
    return results, null_dists


# ── Save helper ───────────────────────────────────────────────────────────────
def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = OUT / f"{stem}{FIG_SUFFIX}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else 150)
        print(f"  Saved {path.name}")
    plt.close(fig)


# ── (logKow, logKaw) helpers ─────────────────────────────────────────────────────

def _draw_regulatory_regions_kaw(ax, kow_lim, kaw_lim):
    kow_arr = np.linspace(kow_lim[0], kow_lim[1], 400)
    ax.set_facecolor(C_NON_BM)

    # M/vM side, two tiers for the main figures: confident vM below VM_KOW,
    # M from VM_KOW up to M_KOW (Not-M above M_KOW, unshaded). This matches
    # the two-threshold M/vM description used in the main text (M<~4.5,
    # vM<~2.5); Arp & Hale's own finer 4-tier screening scheme (adding a
    # "Potential M/vM" margin at logKow 3.5-4.5 and a separate "M/vM,
    # unclear if very" tier at 2.5-3.5) is not drawn, since a third,
    # unlabelled tier in this figure was confusing rather than informative.
    # All region fills below are fully opaque (alpha=1.0): every (x, y) point
    # belongs to exactly one joint hazard class, and with the zorder ordering
    # already used here, a higher-zorder region always represents the correct
    # *combined* classification for whatever it's drawn over -- so opacity
    # gives the correct colour everywhere, whereas alpha<1 let a lower region
    # show through and mix into a colour matching neither class. The iso-logKoa
    # reference lines are drawn last, above all region fills (see
    # _draw_iso_koa_lines), so they remain visible as thin lines on top.
    ax.axvspan(kow_lim[0], VM_KOW, color=C_VM, alpha=1.0, zorder=2)
    ax.axvspan(VM_KOW, M_KOW, color=C_M_ONLY, alpha=1.0, zorder=2)

    # Kelly (air-breather) B/vB, below the logKoa=6 line, split by whether the
    # aquatic BCF route (see _draw_aquatic_bcf_overlay) ALSO applies at this
    # logKow: terrestrial-only below B_KOW_AQUATIC, "both routes" from there
    # to VB_KOW. vB is always "both" once kow>=VB_KOW, since the aquatic vB
    # threshold is numerically identical to Kelly's -- see module-level note
    # on B_KOW_AQUATIC.
    kow_b = kow_arr[kow_arr >= M_KOW]
    b_top = kow_b - B_KOA
    ax.fill_between(kow_b, kaw_lim[0], b_top,
                    where=(kow_b >= B_KOW) & (kow_b < B_KOW_AQUATIC), color=C_B_ONLY, alpha=1.0, zorder=1)
    ax.fill_between(kow_b, kaw_lim[0], b_top,
                    where=(kow_b >= B_KOW_AQUATIC) & (kow_b < VB_KOW), color=C_B_BOTH, alpha=1.0, zorder=1)
    ax.fill_between(kow_b, kaw_lim[0], b_top,
                    where=(kow_b >= VB_KOW), color=C_VB, alpha=1.0, zorder=2)

    kow_bm = kow_arr[(kow_arr >= B_KOW) & (kow_arr <= M_KOW)]
    if len(kow_bm):
        ax.fill_between(kow_bm, kaw_lim[0], kow_bm - B_KOA, color=C_BM, alpha=1.0, zorder=3)
    kow_bvm = kow_arr[(kow_arr >= B_KOW) & (kow_arr <= VM_KOW)]
    if len(kow_bvm):
        ax.fill_between(kow_bvm, kaw_lim[0], kow_bvm - B_KOA, color=C_BM, alpha=1.0, zorder=4)

    kow_gap = kow_arr[(kow_arr >= GAP3_KOW_MIN) & (kow_arr <= GAP3_KOW_MAX)]
    ax.fill_between(kow_gap, kow_gap - B_KOA, kaw_lim[1],
                    color=C_GAP, alpha=1.0, zorder=5, hatch="///", edgecolor=C_GAP_EDGE, linewidth=0.5)

    _draw_aquatic_bcf_overlay(ax, kow_lim, kaw_lim)


def _draw_aquatic_bcf_overlay(ax, kow_lim, kaw_lim):
    """Standard aquatic (fish) BCF-from-Kow bioaccumulation, logKow-only, no
    logKoa needed: B at logKow>=B_KOW_AQUATIC (BCF>2,000), vB at
    logKow>=VB_KOW (BCF>5,000 -- coincidentally the same Kow value as
    Kelly's air-breather vB threshold). Drawn only ABOVE the logKoa=6 line,
    i.e. only where a chemical fails Kelly's air-breather B/vB test on
    logKoa but is still B/vB via this ordinary Kow-only route -- this is
    what closes the old, undifferentiated "neither" region into B/vB rather
    than leaving it unclassified (Ian Cousins, Slack 2026-04-28: "The gap
    disappears almost entirely when including aquatic bioaccumulation!").

    Plain solid fill in a gold shade (C_B_AQUATIC / C_VB_AQUATIC) distinct
    from the Kelly-based reds, since this region is reachable *only* via the
    aquatic route -- the whole point being that a molecule here would be
    classed "neither B nor M" under the Kelly-only picture. No hatch:
    cross-hatching previously used to distinguish this from the Kelly-based
    fill rendered as visibly overlapping/seamed geometry when the PDF was
    opened in a vector editor (Affinity Designer), even though it was a
    single opaque fill in the rasterised PNG."""
    kow_arr = np.linspace(kow_lim[0], kow_lim[1], 400)
    kow_b_aq = kow_arr[(kow_arr >= B_KOW_AQUATIC) & (kow_arr < VB_KOW)]
    if len(kow_b_aq):
        ax.fill_between(kow_b_aq, kow_b_aq - B_KOA, kaw_lim[1], color=C_B_AQUATIC, alpha=1.0,
                        linewidth=0, zorder=5)
    kow_vb_aq = kow_arr[kow_arr >= VB_KOW]
    if len(kow_vb_aq):
        ax.fill_between(kow_vb_aq, kow_vb_aq - B_KOA, kaw_lim[1], color=C_VB_AQUATIC, alpha=1.0,
                        linewidth=0, zorder=5)


def _draw_iso_koa_lines(ax, kow_lim, kaw_lim, highlight_koa=None):
    # Standard x-y Cartesian gridlines at every major tick, in addition to
    # the diagonal iso-logKoa lines below. zorder=6 matches those diagonal
    # lines -- i.e. above the opaque region fills (zorder<=5) -- so the grid
    # remains visible as a light reference overlay rather than being hidden
    # under solid colour. Colour is a mid-grey, not white: most region fills
    # here are pale tints (including the near-white C_NON_BM background), so
    # a white grid at low alpha washed out completely over them -- grey at
    # moderate alpha reads on both the pale background and the more
    # saturated red/blue/gold region fills.
    # set_axisbelow(False) is required here: matplotlib's default
    # axes.axisbelow='line' silently forces gridlines to a fixed low zorder
    # (below any patch/fill), overriding whatever zorder is passed to
    # ax.grid() below -- without this call the grid was being drawn but was
    # completely hidden under the opaque region fills.
    ax.set_axisbelow(False)
    ax.grid(True, which="major", axis="both", color="#6b6b6b", lw=0.5, alpha=0.55, zorder=6)
    kow_arr = np.linspace(kow_lim[0], kow_lim[1], 400)
    for koa_val in range(int(kow_lim[0]-kaw_lim[1]), int(kow_lim[1]-kaw_lim[0]), 1):
        kaw_line = kow_arr - koa_val
        in_view = (kaw_line >= kaw_lim[0]) & (kaw_line <= kaw_lim[1])
        if not in_view.any():
            continue
        if koa_val == highlight_koa:
            ax.plot(kow_arr, kaw_line, color=C_EDGE, lw=1.2, ls="-", zorder=6,
                    label=fr"$\log K_{{oa}}={koa_val}$")
        else:
            # Same grey-not-white reasoning as the grid above: this line
            # crosses both pale and saturated region fills, so it needs a
            # tone with reasonable contrast against both, not just dark
            # colours (where white used to disappear on the pale background).
            ax.plot(kow_arr, kaw_line, color="#6b6b6b", lw=0.5, ls=":", alpha=0.6, zorder=6)
        x_lbl = kow_lim[1] - 0.1
        y_lbl = x_lbl - koa_val
        if kaw_lim[0] <= y_lbl <= kaw_lim[1]:
            ax.text(x_lbl, y_lbl, fr"$K_{{oa}}={koa_val}$", zorder=6,
                    fontsize=FS_FLOOR, color="dimgray", va="center", ha="right", alpha=0.8, clip_on=True)


def _draw_boundary_lines_kaw(ax, kow_lim):
    lw = 1.0
    ax.axvline(B_KOW,  color=C_EDGE, lw=lw, ls="--", zorder=6)
    ax.axvline(VB_KOW, color=C_EDGE, lw=lw, ls="-",  zorder=6)
    ax.axvline(M_KOW,  color=C_EDGE, lw=lw, ls="-.", zorder=6)
    ax.axvline(VM_KOW, color=C_EDGE, lw=lw, ls=":",  zorder=6)
    # Aquatic-B threshold: only ~0.1 log-units from M_KOW (the gap itself is
    # this narrow), so it sits right next to the M_KOW line above. Coloured
    # to match the aquatic-route gold fill (C_VB_AQUATIC, the darker of the
    # two gold shades, for contrast against the page) rather than the
    # neutral C_EDGE used for every M/vM-family line, at full opacity and
    # slightly thicker, so it reads as "an aquatic-route threshold" rather
    # than blending into the cluster of M-related lines immediately to its
    # left.
    ax.axvline(B_KOW_AQUATIC, color=C_VB_AQUATIC, lw=lw * 1.1, ls=(0, (3, 1, 1, 1)), zorder=6.5, alpha=1.0)


# ── Figure 1 - Regulatory space ────────────────────────────────────────────────
def fig1_regulatory_space():
    kow_lim = (-2.0, 8.0)
    kaw_lim = (-6.0, 4.0)

    # With set_aspect("equal") below, the axes box's true shape is fixed by
    # the ratio of these two spans (10:10 here -- a square) regardless of
    # figsize -- so figsize is chosen to already match that ratio (plus
    # headroom for title/legend) rather than fighting it, which is what
    # previously squeezed the box into a narrow, cramped column with
    # overlapping labels.
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 9.2))

    _draw_regulatory_regions_kaw(ax, kow_lim, kaw_lim)
    _draw_iso_koa_lines(ax, kow_lim, kaw_lim, highlight_koa=int(B_KOA))
    _draw_boundary_lines_kaw(ax, kow_lim)

    def rlabel(x, y, txt, **kw):
        kw.setdefault("fontsize", FS_LABEL)
        kw.setdefault("fontweight", "bold")
        ax.text(x, y, txt, ha="center", va="center", zorder=7, **kw)

    # Label positions follow the thresholds (the UBA values reproduce the
    # original hand-placed positions). Under CLP, vM ends below B_KOW, so
    # there is no B+vM region to label.
    rlabel(0.5 * (M_KOW + VB_KOW), -4.0, "B", color=C_B_LINE)
    rlabel(6.5, -4.0, "vB",   color=C_VB_LINE)
    rlabel(M_KOW - 1.0,  2.0, "M",  color=C_M_LINE)
    rlabel(VM_KOW - 1.5, 2.0, "vM", color=C_M_LINE)
    rlabel(0.5 * (B_KOW + M_KOW), -4.0, "B+M", color=C_BM_LABEL)
    if VM_KOW > B_KOW:
        rlabel(1.5, -4.0, "B+vM", color=C_BM_LABEL)
    rlabel(6.5,  2.0, "vB\n(aquatic)", color=C_EDGE, fontweight="normal", fontstyle="italic")
    rlabel(4.75,  2.0, "B\n(aquatic)", color=C_EDGE, fontweight="normal", fontstyle="italic")

    gap_mid = 0.5 * (GAP3_KOW_MIN + GAP3_KOW_MAX)
    ax.annotate(
        "gap", xy=(gap_mid, 1.5), xytext=(gap_mid - 2.2, 1.8),
        ha="right", va="center", fontsize=FS_LABEL, fontstyle="italic", color=C_GAP_LABEL,
        arrowprops=dict(arrowstyle="->", color=C_GAP_LABEL, lw=1.0), zorder=8,
    )

    ann_y = kaw_lim[0] + 0.5
    for x, txt in [
        (B_KOW,  r"$\log K_{ow}=2$"),
        (VB_KOW, r"$\log K_{ow}=5$"),
        (M_KOW,  rf"$\log K_{{ow}}{{\approx}}{M_KOW:g}$"),
        (VM_KOW, rf"$\log K_{{ow}}{{\approx}}{VM_KOW:g}$"),
    ]:
        ax.annotate(txt, (x + 0.1, ann_y + 0.3), fontsize=FS_LABEL, color=C_EDGE, va="bottom", rotation=90)

    ax.set_xlim(kow_lim)
    ax.set_ylim(kaw_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel(r"$\log K_{ow}$", fontsize=FS_TITLE)
    ax.set_ylabel(r"$\log K_{aw}$", fontsize=FS_TITLE)
    ax.set_title(r"Regulatory hazard regions in the $(\log K_{ow},\;\log K_{aw})$ plane"
                 "\n" r"(iso-$\log K_{oa}$ lines are diagonals)" + _title_tag(), fontsize=FS_TITLE)

    legend_patches = [
        mpatches.Patch(fc=C_VM,    label="vM (very mobile)"),
        mpatches.Patch(fc=C_M_ONLY, label="M (mobile)"),
        mpatches.Patch(fc=C_B_ONLY, label="B, terrestrial (Kelly) only"),
        mpatches.Patch(fc=C_B_BOTH, label="B, both routes"),
        mpatches.Patch(fc=C_B_AQUATIC, label="B, aquatic (BCF) only"),
        mpatches.Patch(fc=C_VB,    label="vB, both routes"),
        mpatches.Patch(fc=C_VB_AQUATIC, label="vB, aquatic (BCF) only"),
        mpatches.Patch(fc=C_BM,    label="B and M"),
        mpatches.Patch(fc=C_GAP, hatch="///", edgecolor=C_GAP_EDGE, label="Gap (neither B nor M)"),
    ]
    ax.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, fontsize=FS_LABEL, framealpha=0.9, borderpad=0.6)

    fig.tight_layout()
    _save(fig, "fig1_regulatory_space")


# ── Figure 3 - Model comparison scatter (5 models) ─────────────────────────────
def fig3_model_comparison(merged):
    """merged: output of load_benchmark_predictions_all() -- ALL molecules with
    at least one of logKow/logKoa measured (see fig8_gap_zoom_models for the
    'category' column and how single-observation molecules get their missing
    endpoint filled in by cross-inference).

    Only the predicted point is drawn as a circle (no separate hollow
    "experimental" marker); every circle and its residual line are coloured
    by data provenance (CATEGORY_COLORS) rather than by model, since the
    model is already unambiguous from the panel. A residual line is drawn to
    every molecule's predicted point, using whichever endpoint(s) are
    genuinely measured for it: the full two-segment (logKow-then-logKoa)
    decomposition for 'both' molecules, and a single one-segment residual
    (the only one that is actually knowable) for the two single-observation
    categories -- e.g. for 'logKow_only', the segment runs from the true
    logKow at the modelled logKoa level to the predicted point, i.e. it is
    the logKow error alone, evaluated at the only logKoa value available.
    Lines are drawn on top of the circles at alpha=0.7, so they read clearly
    against the dense point cloud without fully hiding overlapping points.
    RMSE is computed against the full logKow-/logKoa-measured populations
    respectively, matching Table~S1 rather than the smaller paired-only
    subset."""
    # Same square (kow_lim, kaw_lim) window as fig1: among molecules
    # with a genuine paired (logKow, logKaw) value, 96-99% fall inside this
    # window (checked directly against the data), so the handful of extreme
    # outliers clipped here are a small price for a much more legible,
    # true-45-degree-looking square panel.
    kow_lim = (-4.0, 10.0)
    kaw_lim = (-11.0, 3.0)

    n_cols = 2
    n_rows = (len(MODEL_ORDER) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_IN, 5 * n_rows),
                             sharey=True, sharex=True, squeeze=False)

    for idx, mkey in enumerate(MODEL_ORDER):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        _draw_iso_koa_lines(ax, kow_lim, kaw_lim, highlight_koa=int(B_KOA))
        _draw_boundary_lines_kaw(ax, kow_lim)
        ax.set_facecolor(C_BG)

        kow_col = f"{mkey}_logKow"
        koa_col = f"{mkey}_logKoa"
        kaw_col = f"{mkey}_logKaw"
        valid = merged[kow_col].notna() & merged[koa_col].notna()
        sub = merged[valid].copy()

        kow_e = sub["logKow_exp"].values
        koa_e = sub["logKoa_exp"].values
        kaw_e = sub["logKaw_exp"].values  # NaN unless both endpoints measured
        kow_p = sub[kow_col].values
        koa_p = sub[koa_col].values
        kaw_p = sub[kaw_col].values
        category = sub["category"].values

        d_kow = kow_p - kow_e   # defined wherever logKow was measured
        d_koa = koa_p - koa_e   # defined wherever logKoa was measured
        n_kow = int(np.isfinite(d_kow).sum())
        n_koa = int(np.isfinite(d_koa).sum())
        rmse_kow = np.sqrt(np.nanmean(d_kow ** 2))
        rmse_koa = np.sqrt(np.nanmean(d_koa ** 2))

        # Predicted point: shown for every molecule, coloured by data provenance.
        for cat in ("both", "logKow_only", "logKoa_only"):
            m = category == cat
            ax.scatter(kow_p[m], kaw_p[m], c=CATEGORY_COLORS[cat], s=5, alpha=0.7,
                       linewidths=0, zorder=3, rasterized=True)

        # Residual line to the predicted point for EVERY molecule (not a
        # subsample), on top, opaque, coloured by provenance category. Built
        # as a single LineCollection per category rather than one ax.plot()
        # call per molecule -- thousands of individual plot() calls would be
        # far slower to render.
        segments, seg_colors = [], []

        m_both = category == "both"
        if m_both.any():
            xe, ye = kow_e[m_both], kaw_e[m_both]
            xp, yp = kow_p[m_both], kaw_p[m_both]
            xm, ym = xp, ye + d_kow[m_both]
            seg1 = np.stack([np.column_stack([xe, ye]), np.column_stack([xm, ym])], axis=1)
            seg2 = np.stack([np.column_stack([xm, ym]), np.column_stack([xp, yp])], axis=1)
            segments.extend(seg1.tolist())
            segments.extend(seg2.tolist())
            seg_colors.extend([CATEGORY_COLORS["both"]] * (2 * int(m_both.sum())))

        m_kow_only = category == "logKow_only"
        if m_kow_only.any():
            xe = kow_e[m_kow_only]
            xp, yp = kow_p[m_kow_only], kaw_p[m_kow_only]
            ye = yp - d_kow[m_kow_only]
            seg = np.stack([np.column_stack([xe, ye]), np.column_stack([xp, yp])], axis=1)
            segments.extend(seg.tolist())
            seg_colors.extend([CATEGORY_COLORS["logKow_only"]] * int(m_kow_only.sum()))

        m_koa_only = category == "logKoa_only"
        if m_koa_only.any():
            xp, yp = kow_p[m_koa_only], kaw_p[m_koa_only]
            ye = yp + d_koa[m_koa_only]
            seg = np.stack([np.column_stack([xp, ye]), np.column_stack([xp, yp])], axis=1)
            segments.extend(seg.tolist())
            seg_colors.extend([CATEGORY_COLORS["logKoa_only"]] * int(m_koa_only.sum()))

        ax.add_collection(LineCollection(segments, colors=seg_colors, linewidths=0.5,
                                         alpha=0.7, zorder=8, rasterized=True))

        ax.text(0.97, 0.12,
                f"RMSE($\\log K_{{ow}}$)={rmse_kow:.3f} (n={n_kow:,})\n"
                f"RMSE($\\log K_{{oa}}$)={rmse_koa:.3f} (n={n_koa:,})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_LABEL, color="#333",
                zorder=10, bbox=dict(fc="white", alpha=0.95, boxstyle="round,pad=0.3"))

        ax.set_xlim(kow_lim)
        ax.set_ylim(kaw_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_title(MODEL_LABELS[mkey], fontsize=FS_LABEL_HI, fontweight="bold")
        ax.set_xlabel(r"$\log K_{ow}$", fontsize=FS_TITLE)
        if col == 0:
            ax.set_ylabel(r"$\log K_{aw}$", fontsize=FS_TITLE)

    for idx in range(len(MODEL_ORDER), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS[cat],
               markersize=7, label=CATEGORY_LABELS[cat])
        for cat in ("both", "logKow_only", "logKoa_only")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=FS_LABEL_HI,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.10))

    n_both = int((merged["category"] == "both").sum())
    n_all = len(merged)
    fig.suptitle(
        "Predicted vs. experimental: residual decomposition\n"
        r"in $(\log K_{ow},\,\log K_{aw})$ space"
        "\n(circles: predicted point; lines: residual to it,\n"
        "coloured by which endpoint(s) are genuinely measured.\n"
        "RMSE uses all molecules with that endpoint measured\n"
        f"(n={n_all:,} total, n={n_both:,} with both measured).",
        y=1.05, fontsize=FS_TITLE,
    )
    fig.tight_layout()
    _save(fig, "fig3_model_comparison")


# ── Figure 7 - Y-randomization validation ──────────────────────────────────────
def fig7_yrandomization(results, null_dists):
    model_order = ["crippen", "naefacree_mixed"]
    endpoints = ["logKow", "logKoa"]
    fig, axes = plt.subplots(len(model_order), 2, figsize=(FIG_WIDTH_IN, 3.75 * len(model_order)), sharex=True)

    for row, mkey in enumerate(model_order):
        for col, endpoint in enumerate(endpoints):
            ax = axes[row][col]
            null_r2 = null_dists[(mkey, endpoint)]
            rrow = results[(results["model"] == mkey) & (results["endpoint"] == endpoint)].iloc[0]

            ax.hist(null_r2, bins=30, color="#6b6b6b", alpha=0.9, linewidth=0,
                    label=f"permuted (n={int(rrow['n_permutations'])})")
            ax.axvline(rrow["observed_r2"], color=C_HIGHLIGHT, lw=2, ls="-",
                       label=f"observed $R^2$={rrow['observed_r2']:.2f}")
            endpoint_label = r"$\log K_{ow}$" if endpoint == "logKow" else r"$\log K_{oa}$"
            if rrow["p_value"] == 0:
                p_str = f"$p$<{1.0/rrow['n_permutations']:.3f} (0/{int(rrow['n_permutations'])} permutations " r"$\geq$" " observed)"
            else:
                p_str = f"$p$={rrow['p_value']:.3f}"
            ax.set_title(f"{MODEL_LABELS.get(mkey, mkey)} -- {endpoint_label}\n{p_str}", fontsize=FS_LABEL_HI)
            ax.set_xlim(-0.3, 1.0)
            if row == len(model_order) - 1:
                ax.set_xlabel(r"$R^2$")
            if col == 0:
                ax.set_ylabel("Permutation count")
            ax.legend(loc="upper left", fontsize=FS_LABEL_HI, framealpha=0.9)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)

    fig.suptitle(r"Y-randomisation: permuted-label null distribution vs. observed $R^2$", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig7_yrandomization")


# ── Figure 8 - Gap zoom, per-model variation ────────────────────────────────────
def _in_gap(kow, koa):
    return (kow > GAP3_KOW_MIN) & (kow < GAP3_KOW_MAX) & (koa < B_KOA)


def _scatter_area_to_markersize(area):
    """Convert a matplotlib scatter 's' value (marker area, points^2) to the
    equivalent Line2D 'markersize' (~diameter, points) for a circular marker,
    so full circles drawn via ax.scatter and half-filled circles drawn via
    ax.plot (which only accepts 'markersize', not 's') end up the same
    apparent size."""
    return 1.2 * np.sqrt(area / np.pi)


def fig8_gap_zoom_models(merged):
    """merged: output of load_benchmark_predictions_all() -- ALL molecules
    with at least one of logKow/logKoa measured, tagged by 'category'
    ('both' / 'logKow_only' / 'logKoa_only'). Every molecule gets a point
    (full circle for 'both', half-filled for the single-observation
    categories -- fillstyle 'bottom' for logKow-only, i.e. logKoa
    missing/modelled; 'right' for logKoa-only, i.e. logKow missing/modelled),
    but an error line is drawn ONLY for molecules predicted into the gap, to
    keep the plot legible -- for 'both' molecules this is the usual
    experimental-to-predicted line; for single-observation molecules there is
    no experimental logKaw (it needs both endpoints), so it is instead a
    PARTIAL residual line using whichever endpoint is genuinely measured,
    held at the model's own value for the missing one (same construction as
    fig3_model_comparison). Full and half-filled circles are sized via
    _scatter_area_to_markersize() so they read as the same size despite being
    drawn through two different matplotlib APIs (scatter vs. plot)."""
    # Both limits are 4 log-units wide, on clean integer bounds -- with
    # set_aspect("equal") below, this gives true square panels with
    # integer-valued ticks landing exactly on the edges. Widened from the
    # original (4.0, 5.8) kow window (only 1.8 units) specifically to avoid
    # clipping any of the "predicted into gap" markers, which span
    # logKaw~[-1.33, 1.03] here (checked directly against the data); see
    # fig10 for the shared, slightly more generous choice needed there.
    # (UBA window; see MOBILITY_CRITERIA["clp"]["zoom"] for the CLP one.)
    kow_lim, kaw_lim = MOBILITY["zoom"]
    kow_arr = np.linspace(kow_lim[0], kow_lim[1], 400)

    panels = ["observed"] + MODEL_ORDER
    n_cols = 3
    n_rows = -(-len(panels) // n_cols)  # ceil
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_IN - 0.05, 3.7 * n_rows), sharey=True)
    axes = axes.flatten()
    for extra_ax in axes[len(panels):]:
        extra_ax.axis("off")

    both = merged[merged["category"] == "both"]
    kow_e = both["logKow_exp"].values
    koa_e = both["logKoa_exp"].values
    kaw_e = both["logKaw_exp"].values
    n_obs_gap = int(_in_gap(kow_e, koa_e).sum())
    n_total_all = len(merged)
    n_both = len(both)
    n_kow_only = int((merged["category"] == "logKow_only").sum())
    n_koa_only = int((merged["category"] == "logKoa_only").sum())
    n_shown = int(((kow_e >= kow_lim[0]) & (kow_e <= kow_lim[1])).sum())

    marker_kow_only = MarkerStyle("o", fillstyle="bottom")  # logKow measured, logKoa modelled
    marker_koa_only = MarkerStyle("o", fillstyle="right")    # logKoa measured, logKow modelled

    # Full ("both") circles are drawn via scatter (s = area, points^2); half-filled
    # ("only") circles via plot (markersize ~ diameter, points). Full circles are
    # made bigger and half circles smaller than the previous mismatched defaults,
    # calibrated to the same apparent size via _scatter_area_to_markersize().
    FULL_S_GAP, FULL_S_NONGAP = 36, 16
    HALF_MS_GAP = _scatter_area_to_markersize(FULL_S_GAP)
    HALF_MS_NONGAP = _scatter_area_to_markersize(FULL_S_NONGAP)

    for i, panel in enumerate(panels):
        ax = axes[i]
        ax.set_facecolor(C_BG)
        _draw_iso_koa_lines(ax, kow_lim, kaw_lim, highlight_koa=int(B_KOA))
        ax.axvline(VB_KOW, color=C_B_LINE, lw=1.4, ls="-")
        ax.axvline(M_KOW, color=C_M_LINE, lw=1.4, ls="-.")

        kow_gap = kow_arr[(kow_arr > GAP3_KOW_MIN) & (kow_arr < GAP3_KOW_MAX)]
        ax.fill_between(kow_gap, kow_gap - B_KOA, kaw_lim[1], color=C_GAP, alpha=1.0,
                        hatch="///", edgecolor=C_GAP_EDGE, linewidth=0.7, zorder=2)
        _draw_aquatic_bcf_overlay(ax, kow_lim, kaw_lim)

        ax.scatter(kow_e, kaw_e, s=10, c=C_MUTED, alpha=0.6, linewidths=0, zorder=3,
                  rasterized=True, label="Experimental")

        if panel == "observed":
            ax.set_title(f"Observed\n(n in gap = {n_obs_gap})", fontsize=FS_LABEL_HI, fontweight="bold")
        else:
            mkey = panel
            kow_p = both[f"{mkey}_logKow"].values
            koa_p = both[f"{mkey}_logKoa"].values
            kaw_p = both[f"{mkey}_logKaw"].values
            valid = np.isfinite(kow_p) & np.isfinite(kaw_p)
            pred_in_gap = _in_gap(kow_p[valid], koa_p[valid])
            n_pred_gap = int(pred_in_gap.sum())

            idx_valid = np.where(valid)[0]
            for j, is_gap in zip(idx_valid, pred_in_gap):
                # only draw/annotate molecules whose prediction lands near the zoom window
                if not (kow_lim[0] <= kow_p[j] <= kow_lim[1]):
                    continue
                color = C_HIGHLIGHT if is_gap else C_CALM
                alpha = 0.9 if is_gap else 0.12
                # Error line only for molecules actually predicted into the
                # gap -- drawing one for every point made the plot too busy.
                if is_gap:
                    ax.plot([kow_e[j], kow_p[j]], [kaw_e[j], kaw_p[j]], color=color, lw=1.3,
                            alpha=alpha, zorder=4)
                ax.scatter([kow_p[j]], [kaw_p[j]], s=FULL_S_GAP if is_gap else FULL_S_NONGAP,
                          c=color, alpha=alpha, linewidths=0, zorder=5)

            # Single-observation molecules: no genuine experimental anchor
            # exists in (logKow, logKaw) space (one coordinate is unmeasured),
            # but a PARTIAL residual line can still be drawn using whichever
            # endpoint is genuinely known, held at the model's own value for
            # the missing one -- the same construction used in fig3
            # (fig3_model_comparison): for logKow_only, the line runs from
            # the true logKow (at the modelled logKoa level) to the predicted
            # point; for logKoa_only, from the true logKoa (at the modelled
            # logKow level) to the predicted point.
            n_pred_gap_partial = 0
            for cat, marker in [("logKow_only", marker_kow_only), ("logKoa_only", marker_koa_only)]:
                sub = merged[merged["category"] == cat]
                kow_ps = sub[f"{mkey}_logKow"].values
                koa_ps = sub[f"{mkey}_logKoa"].values
                kaw_ps = sub[f"{mkey}_logKaw"].values
                kow_es = sub["logKow_exp"].values
                koa_es = sub["logKoa_exp"].values
                v = np.isfinite(kow_ps) & np.isfinite(kaw_ps)
                for k in np.where(v)[0]:
                    if not (kow_lim[0] <= kow_ps[k] <= kow_lim[1]):
                        continue
                    is_gap = (GAP3_KOW_MIN < kow_ps[k] < GAP3_KOW_MAX) and (koa_ps[k] < B_KOA)
                    n_pred_gap_partial += int(is_gap)
                    color = C_HIGHLIGHT if is_gap else C_CALM
                    alpha = 0.9 if is_gap else 0.35
                    xp, yp = kow_ps[k], kaw_ps[k]
                    # Partial residual line only for molecules predicted into
                    # the gap -- same declutter as the paired-molecule case.
                    if is_gap:
                        if cat == "logKow_only":
                            xe, ye = kow_es[k], yp - (kow_ps[k] - kow_es[k])
                        else:  # logKoa_only
                            xe, ye = xp, yp + (koa_ps[k] - koa_es[k])
                        ax.plot([xe, xp], [ye, yp], color=color, lw=1.3, alpha=alpha, zorder=4)
                    ax.plot([xp], [yp], marker=marker,
                            markersize=HALF_MS_GAP if is_gap else HALF_MS_NONGAP,
                            markerfacecolor=color,
                            markeredgecolor=color, markeredgewidth=0.7,
                            linestyle="none", alpha=alpha, zorder=5)

            ax.set_title(f"{MODEL_LABELS[mkey]}\n(n predicted in gap = {n_pred_gap + n_pred_gap_partial})",
                        fontsize=FS_LABEL_HI, fontweight="bold")

        ax.set_xlim(kow_lim)
        ax.set_ylim(kaw_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        if i >= len(panels) - n_cols:
            ax.set_xlabel(r"$\log K_{ow}$")

    for r in range(n_rows):
        axes[r * n_cols].set_ylabel(r"$\log K_{aw}$")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MUTED, markersize=6, label="Experimental"),
        Line2D([0], [0], color=C_HIGHLIGHT, lw=1.5, marker="o", markersize=5, label="Predicted into gap"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CALM, alpha=0.4,
              markersize=5, label="Predicted, not in gap (no line drawn)"),
        Line2D([0], [0], marker=marker_kow_only, color="w", markerfacecolor=C_EDGE,
              markeredgecolor=C_EDGE, markersize=8,
              label=r"Only $\log K_{ow}$ measured ($\log K_{oa}$ modelled)"),
        Line2D([0], [0], marker=marker_koa_only, color="w", markerfacecolor=C_EDGE,
              markeredgecolor=C_EDGE, markersize=8,
              label=r"Only $\log K_{oa}$ measured ($\log K_{ow}$ modelled)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=1, fontsize=FS_LABEL_HI,
              bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.suptitle(f"Gap membership: observed vs. predicted, by model\n"
                 f"(n={n_total_all:,} molecules total:\n"
                 f"{n_both:,} both endpoints measured,\n"
                 f"{n_kow_only:,} $\\log K_{{ow}}$-only, {n_koa_only:,} $\\log K_{{oa}}$-only;\n"
                 f"n={n_shown} experimental points shown in this window)" + _title_tag(),
                 fontsize=FS_TITLE, y=1.05 if not MOBILITY["title_tag"] else 1.08)
    fig.tight_layout()
    _save(fig, "fig8_gap_zoom_models")


# ── Figure 9 - Arp & Hale gap persistence (observed vs predicted) ──────────────
def _bucket_uba(cat):
    if not isinstance(cat, str):
        return "no data"
    if cat == "Potential PMT/vPvM":
        return "Potential P"
    if "vPvM" in cat and not cat.startswith("Not"):  # "Not PMT/vPvM" is not vP
        return "vP"
    if cat in ("PM", "PMT"):
        return "P"
    if cat in ("Not P", "Not P, Not M"):
        return "Not P"
    return "no data"


_BUCKET_STYLE = {
    "vP":          (BUCKET_COLORS["vP"], 30, 3, 0.9),
    "P":           (BUCKET_COLORS["P"], 24, 2, 0.85),
    "Potential P": (BUCKET_COLORS["Potential P"], 20, 1, 0.8),
    "Not P":       (BUCKET_COLORS["Not P"], 18, 1, 0.7),
    "no data":     (BUCKET_COLORS["no data"], 14, 0, 0.5),
}


def fig9_arphale_gap_persistence(df_arphale_lser, df_arphale_naef, df_arphale_pred):
    # kaw_lim was already 6 log-units wide (needed to hold this dataset's
    # long LSER-estimate outlier tail -- see fig10 for the same issue);
    # kow_lim is widened to match (rather than narrowing kaw_lim, which would
    # have clipped more of that tail) so both spans are equal and the panels
    # are true squares under set_aspect("equal") below.
    kow_lim = (2.0, 8.0)
    kaw_lim = (-3.0, 3.0)
    kow_arr = np.linspace(kow_lim[0], kow_lim[1], 400)

    panel_specs = [("lser", df_arphale_lser, "UFZ-LSER estimate\n(Arp & Hale)")]
    if df_arphale_naef is not None:
        panel_specs.append(("naef", df_arphale_naef, "Naef experimental\n(intersection)"))
    if df_arphale_pred is not None:
        panel_specs.append(("pred", df_arphale_pred, "Predicted\n(combined ridge)"))

    n_panels = len(panel_specs)
    fig, all_axes = plt.subplots(2, n_panels, figsize=(FIG_WIDTH_IN - 0.15, 5.6), sharey="row",
                                 gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.05})
    axes = all_axes[0]
    bar_axes = all_axes[1]

    def _draw_panel(ax, kow, koa, bucket, title):
        ax.set_facecolor(C_BG)
        _draw_iso_koa_lines(ax, kow_lim, kaw_lim, highlight_koa=int(B_KOA))
        ax.axvline(VB_KOW, color=C_B_LINE, lw=1.4, ls="-")
        ax.axvline(M_KOW, color=C_M_LINE, lw=1.4, ls="-.")
        kow_gap = kow_arr[(kow_arr > GAP3_KOW_MIN) & (kow_arr < GAP3_KOW_MAX)]
        ax.fill_between(kow_gap, kow_gap - B_KOA, kaw_lim[1], color=C_GAP, alpha=1.0,
                        hatch="///", edgecolor=C_GAP_EDGE, linewidth=0.7, zorder=2)
        _draw_aquatic_bcf_overlay(ax, kow_lim, kaw_lim)

        kaw = kow - koa
        in_gap = _in_gap(kow, koa)
        # context (light) then gap molecules (bold), per bucket
        for b, (color, size, zorder, alpha) in _BUCKET_STYLE.items():
            m_ctx = (bucket == b) & ~in_gap
            ax.scatter(kow[m_ctx], kaw[m_ctx], s=6, c=color, alpha=0.25, linewidths=0, zorder=3,
                      rasterized=True)
        for b, (color, size, zorder, alpha) in _BUCKET_STYLE.items():
            m_gap = (bucket == b) & in_gap
            n_b = int(m_gap.sum())
            # Labelled unconditionally (even at n=0): this bucket's colour is
            # also used for the unlabelled context scatter above, which
            # covers every molecule in the panel, not just those in the gap
            # -- without a legend entry here, a reader has no way to decode
            # what that colour means for a bucket that happens to have zero
            # molecules in the gap itself.
            ax.scatter(kow[m_gap], kaw[m_gap], s=size * 2, c=color, alpha=alpha,
                      edgecolors="black", linewidths=0.4, zorder=6 + zorder,
                      label=f"{b} (n={n_b})")
        ax.set_xlim(kow_lim)
        ax.set_ylim(kaw_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlabel(r"$\log K_{ow}$")
        n_total = len(kow)
        ax.set_title(f"{title}\n(n={n_total:,}; in gap: {int(in_gap.sum())})",
                    fontsize=FS_LABEL_HI, fontweight="bold")
        # No room for a legend when the (CLP) gap fills most of the panel;
        # the gap counts then go on the bar panel instead (_draw_bar_panel).
        if MOBILITY["legend_loc"]:
            ax.legend(loc=MOBILITY["legend_loc"], fontsize=FS_LABEL, framealpha=0.9, title="Persistence (gap)",
                      title_fontsize=FS_LABEL, markerscale=0.8, handletextpad=0.4, borderpad=0.4,
                      labelspacing=0.3)

    def _draw_bar_panel(ax, kow, koa, bucket):
        in_gap = _in_gap(kow, koa)
        n_total = len(kow)
        n_gap = int(in_gap.sum())
        labels = list(_BUCKET_STYLE.keys())
        y = np.arange(len(labels))[::-1]  # top-to-bottom: vP, P, Potential P, Not P, no data
        h = 0.38
        overall_pct = [100 * (bucket == b).sum() / n_total for b in labels]
        gap_pct = [100 * ((bucket == b) & in_gap).sum() / n_gap if n_gap else 0 for b in labels]
        colors = [_BUCKET_STYLE[b][0] for b in labels]
        ax.barh(y + h / 2, overall_pct, height=h, color=colors, alpha=0.4,
               edgecolor="black", linewidth=0.4, label="Overall")
        gap_bars = ax.barh(y - h / 2, gap_pct, height=h, color=colors, alpha=1.0,
                           edgecolor="black", linewidth=0.4, label="In gap")
        if not MOBILITY["legend_loc"]:
            ax.bar_label(gap_bars, labels=[f"n={int(((bucket == b) & in_gap).sum())}" for b in labels],
                         fontsize=FS_FLOOR, padding=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=FS_FLOOR)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", labelsize=FS_FLOOR)

    _KOW_COL = {"lser": "logKow", "naef": "naef_logKow", "pred": "pred_naefacree_mixed_logKow"}
    _KOA_COL = {"lser": "logKoa", "naef": "naef_logKoa", "pred": "pred_naefacree_mixed_logKoa"}
    for i, (kind, dfp, title) in enumerate(panel_specs):
        _draw_panel(axes[i], dfp[_KOW_COL[kind]].values, dfp[_KOA_COL[kind]].values,
                   dfp["p_bucket"].values, title)
        _draw_bar_panel(bar_axes[i], dfp[_KOW_COL[kind]].values, dfp[_KOA_COL[kind]].values,
                        dfp["p_bucket"].values)
        bar_axes[i].set_xlabel("% of molecules", fontsize=FS_LABEL)
    axes[0].set_ylabel(r"$\log K_{aw}$")
    # Single shared legend for the overall-vs-gap bar hatching/alpha convention
    # (colour already carries the persistence category, explained by the
    # scatter panels' own legends above).
    bar_legend = [
        mpatches.Patch(fc="grey", alpha=0.4, ec="black", linewidth=0.4, label="Overall"),
        mpatches.Patch(fc="grey", alpha=1.0, ec="black", linewidth=0.4, label="In gap"),
    ]
    fig.legend(handles=bar_legend, loc="lower center", ncol=2, fontsize=FS_LABEL,
              bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Persistence category of Arp & Hale molecules in the B/M gap" + _title_tag(),
                 fontsize=12, y=1.02 if not MOBILITY["title_tag"] else 1.05)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "fig9_arphale_gap_persistence")


# ── Figure 10 - Gap zoom, C&L Inventory, no ground truth ───────────────────────
def load_clinventory_predictions():
    """C&L Inventory molecules with FINAL-model (non-CV, external-inference)
    predicted logKow/logKoa for all 5 models, fit on the full Naef & Acree
    training data (built by build_deployment_predictions.py)."""
    return pd.read_csv(paths.require_cache(CLINVENTORY_PRED_CSV.name, "build_deployment_predictions.py"))


def fig10_clinventory_gap_zoom(df):
    """df: output of load_clinventory_predictions(). Unlike fig8, the C&L Inventory
    has no experimental logKow/logKoa at all, so every molecule's position
    is a model prediction for BOTH coordinates (no 'both'/'only' categories,
    no experimental anchor, and therefore no error lines -- just a coloured
    point per molecule, per model, showing gap membership)."""
    # Same 4x4 window as fig8_gap_zoom_models (see comment there), for true
    # square panels on clean integer bounds and direct visual comparability
    # between the two figures. Naef & Acree's own documented prediction
    # degeneracy (SI, "Model feature vocabularies") gives it a long tail of
    # extreme predictions on this unselected, no-ground-truth inventory (up
    # to logKaw~6.7); a handful of its outlier points fall outside this
    # window and are not drawn, while every other model's gap predictions
    # are fully visible. (UBA window; the CLP one is larger, see
    # MOBILITY_CRITERIA.)
    kow_lim, kaw_lim = MOBILITY["zoom"]
    kow_arr = np.linspace(kow_lim[0], kow_lim[1], 400)

    n_cols = 3
    n_rows = (len(MODEL_ORDER) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH_IN - 0.05, 3.7 * n_rows),
                             sharey=True, squeeze=False)
    axes = axes.flatten()

    FULL_S_GAP, FULL_S_NONGAP = 36, 16
    n_total = len(df)

    for idx, mkey in enumerate(MODEL_ORDER):
        row, col = divmod(idx, n_cols)
        ax = axes[idx]
        ax.set_facecolor(C_BG)
        _draw_iso_koa_lines(ax, kow_lim, kaw_lim, highlight_koa=int(B_KOA))
        ax.axvline(VB_KOW, color=C_B_LINE, lw=1.4, ls="-")
        ax.axvline(M_KOW, color=C_M_LINE, lw=1.4, ls="-.")

        kow_gap = kow_arr[(kow_arr > GAP3_KOW_MIN) & (kow_arr < GAP3_KOW_MAX)]
        ax.fill_between(kow_gap, kow_gap - B_KOA, kaw_lim[1], color=C_GAP, alpha=1.0,
                        hatch="///", edgecolor=C_GAP_EDGE, linewidth=0.7, zorder=2)
        _draw_aquatic_bcf_overlay(ax, kow_lim, kaw_lim)

        kow_p = df[f"{mkey}_logKow"].values
        koa_p = df[f"{mkey}_logKoa"].values
        valid = np.isfinite(kow_p) & np.isfinite(koa_p)
        in_window = valid & (kow_p >= kow_lim[0]) & (kow_p <= kow_lim[1])
        kp = kow_p[in_window]
        kw = kow_p[in_window] - koa_p[in_window]
        gap_mask = _in_gap(kp, koa_p[in_window])
        n_gap = int(gap_mask.sum())

        ax.scatter(kp[~gap_mask], kw[~gap_mask], s=FULL_S_NONGAP, c=C_CALM,
                  alpha=0.25, linewidths=0, zorder=3, rasterized=True)
        ax.scatter(kp[gap_mask], kw[gap_mask], s=FULL_S_GAP, c=C_HIGHLIGHT_ALT,
                  alpha=0.7, linewidths=0, zorder=5, rasterized=True)

        ax.set_title(f"{MODEL_LABELS[mkey]}\n(predicted in gap: {n_gap:,})",
                    fontsize=FS_LABEL_HI, fontweight="bold")
        ax.set_xlim(kow_lim)
        ax.set_ylim(kaw_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        if row == n_rows - 1:
            ax.set_xlabel(r"$\log K_{ow}$")
        if col == 0:
            ax.set_ylabel(r"$\log K_{aw}$")

    for idx in range(len(MODEL_ORDER), n_rows * n_cols):
        axes[idx].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HIGHLIGHT_ALT,
              markersize=7, label="Predicted into gap"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CALM, alpha=0.5,
              markersize=7, label="Predicted, not in gap"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=FS_LABEL_HI,
              bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"Gap membership: C&L Inventory, by model\n"
                 f"(no experimental ground truth; n={n_total:,} molecules total;\n"
                 f"predictions from final models fit on the full\n"
                 f"Naef & Acree data and applied here)" + _title_tag(),
                 fontsize=FS_TITLE, y=1.06 if not MOBILITY["title_tag"] else 1.09)
    fig.tight_layout()
    _save(fig, "fig10_clinventory_gap_zoom")


def gap_counts(merged, df_arphale, df_arphale_naef, df_arphale_pred, df_cl, rmse):
    """Gap membership counts under the active M/vM criteria, one row per
    (dataset, model), with the same counting rules as fig8-fig10. For the
    Arp & Hale sets, also the persistence buckets (and the raw 'Not M'
    category, bucketed as 'no data') of the gap molecules. 'n_gap_rmse' counts
    predictions within +/-1 RMSE (that model's CV RMSE) of the gap bounds,
    'n_gap_core' those still in the gap after shrinking it by 1 RMSE on each
    side, and 'n_vm_vb_gap' those neither vM nor vB (VM_KOW < logKow < VB_KOW)."""
    rows = []

    def _row(dataset, model, kow, koa, bucket=None, raw_cat=None, rmse_kow=None, rmse_koa=None):
        valid = np.isfinite(kow) & np.isfinite(koa)
        kow, koa = kow[valid], koa[valid]
        in_gap = _in_gap(kow, koa)
        row = {"criteria": MOBILITY["name"], "dataset": dataset, "model": model,
               "gap_kow_min": GAP3_KOW_MIN, "gap_kow_max": round(GAP3_KOW_MAX, 3),
               "n_total": int(valid.sum()), "n_gap": int(in_gap.sum()),
               "n_vm_vb_gap": int(((kow > VM_KOW) & (kow < VB_KOW)).sum())}
        if rmse_kow is not None:
            near = ((kow > GAP3_KOW_MIN - rmse_kow) & (kow < GAP3_KOW_MAX + rmse_kow)
                    & (koa < B_KOA + rmse_koa))
            row["n_gap_rmse"] = int(near.sum())
            core = ((kow > GAP3_KOW_MIN + rmse_kow) & (kow < GAP3_KOW_MAX - rmse_kow)
                    & (koa < B_KOA - rmse_koa))
            row["n_gap_core"] = int(core.sum())
        if bucket is not None:
            b = bucket[valid][in_gap]
            for name in _BUCKET_STYLE:
                row[f"gap_{name}"] = int((b == name).sum())
            row["gap_raw_Not_M"] = int((raw_cat[valid][in_gap] == "Not M").sum())
        rows.append(row)

    both = merged[merged["category"] == "both"]
    _row("naef_observed", "experimental", both["logKow_exp"].values, both["logKoa_exp"].values)
    for mkey in MODEL_ORDER:
        _row("naef_predicted", mkey, merged[f"{mkey}_logKow"].values, merged[f"{mkey}_logKoa"].values,
             rmse_kow=rmse[("logKow", mkey)], rmse_koa=rmse[("logKoa", mkey)])
    for name, dfp, kcol, acol in [
        ("arphale_lser", df_arphale, "logKow", "logKoa"),
        ("arphale_naef_experimental", df_arphale_naef, "naef_logKow", "naef_logKoa"),
        ("arphale_predicted", df_arphale_pred, "pred_naefacree_mixed_logKow", "pred_naefacree_mixed_logKoa"),
    ]:
        if dfp is not None:
            _row(name, "naefacree_mixed" if name == "arphale_predicted" else "-",
                 dfp[kcol].values, dfp[acol].values, dfp["p_bucket"].values, dfp["vPvM_UBA"].values)
    if df_cl is not None:
        for mkey in MODEL_ORDER:
            _row("clinventory", mkey, df_cl[f"{mkey}_logKow"].values, df_cl[f"{mkey}_logKoa"].values,
                 rmse_kow=rmse[("logKow", mkey)], rmse_koa=rmse[("logKoa", mkey)])
    return rows


def main():
    print("Loading benchmark metrics + predictions...")
    merged_pred_all = load_benchmark_predictions_all()
    df_arphale = load_arphale_persistence()
    df_arphale_pred = load_arphale_model_predictions()
    df_arphale_naef = load_arphale_naef_experimental()
    try:
        df_cl = load_clinventory_predictions()
    except paths.MissingDataError as exc:
        print(f"  [info] skipping the C&L Inventory figures: {exc}")
        df_cl = None

    # Threshold-independent SI figures (fig3 draws the UBA boundary lines
    # only as a backdrop).
    use_mobility_criteria("uba")
    print("SI Figure S2 (fig3): model comparison...")
    fig3_model_comparison(merged_pred_all)
    print("SI Figure S1 (fig7): y-randomisation...")
    paths.require_cache(YRAND_RESULTS_CSV.name, "yrandomization.py")
    fig7_yrandomization(*load_yrandomization())

    for key in MOBILITY_CRITERIA:
        use_mobility_criteria(key)
        print(f"== {MOBILITY['name']} mobility criteria (M: logKow<{M_KOW:g}, vM: logKow<{VM_KOW:g}) ==")
        print("Figure 1 (fig1): regulatory space...")
        fig1_regulatory_space()
        print("Figure 2 (fig8): gap zoom, per-model variation...")
        fig8_gap_zoom_models(merged_pred_all)
        print("Figure 4 (fig9): Arp & Hale gap persistence...")
        fig9_arphale_gap_persistence(df_arphale, df_arphale_naef, df_arphale_pred)
        if df_cl is not None:
            print("Figure 3 (fig10): C&L Inventory gap zoom...")
            fig10_clinventory_gap_zoom(df_cl)
    use_mobility_criteria("uba")
    print("Done.")


if __name__ == "__main__":
    main()
