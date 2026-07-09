"""
Volcano plot of differential SLiM enrichment: one point per SLiM class (not
per motif -- direction of enrichment is a property of the class, via its
log2 fold-change, so this is the natural chart for "which classes favor
excitatory vs. inhibitory", unlike the motif-level scatter which answers a
different question (which genes/motifs carry the signal).

  x = log2 fold-change (Excitatory rate / Inhibitory rate), length-normalized
      (see run_slim_differential_enrichment_v3.py)
  y = -log10(BH-FDR adjusted p), Fisher's exact
  colour = Excitatory-enriched (significant, log2FC>0) / Inhibitory-enriched
      (significant, log2FC<0) / Not significant -- diverging encoding, using
      the same magenta/aqua already assigned to Excitatory/Inhibitory
      throughout this analysis
  every point is the same size and shape (plain circle), no dark outlines --
      colour + position carry the distinction, not a second style channel;
      FIMO-tested vs. PDZ-tested (regex-based) classes are not visually
      distinguished on the chart itself

Callouts: one box per significant class, listing exemplar genes that carry
that SLiM (not the log2FC/padj numbers -- those are visible from position).
All boxes sit in the upper-right quadrant, which is otherwise empty
whitespace (real hits cluster at low-to-moderate log2FC and low-to-moderate
significance), arranged as a single column so the proven non-crossing
property holds: every target point is strictly left of the column, and box
vertical order matches target significance order exactly (ties broken on
log2_FC so no arbitrary ordering), so no two leader lines can cross.

All 24 significant classes favor Excitatory (0 favor Inhibitory) even after
the length correction -- a real result, not a chart artifact.

Input : slim_differential_enrichment_v3.xlsx / Differential_All
        slim_differential_enrichment_v2.xlsx / Hits_FIMO, Hits_PDZ (exemplar genes)
        synapse_genes_*.xlsx (Entry Name -> primary gene symbol)
        motif_predictions.xlsx (sequence_name -> Entry Name)
Output: slim_enrichment_volcano.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = "/home/au729231/SynapseGigamapper/notebook/ESMC_outputs"
SUMMARY_XL = f"{OUT_DIR}/slim_differential_enrichment_v3.xlsx"
HITS_XL = f"{OUT_DIR}/slim_differential_enrichment_v2.xlsx"
PRED_XL = f"{OUT_DIR}/motif_predictions.xlsx"
FIG_PNG = f"{OUT_DIR}/slim_enrichment_volcano.png"

GENE_MAP_FILES = [
    f"{OUT_DIR}/synapse_genes_training.xlsx",
    f"{OUT_DIR}/synapse_genes_development.xlsx",
    f"{OUT_DIR}/synapse_genes_test.xlsx",
    f"{OUT_DIR}/synapse_genes_holdout.xlsx",
]

SIG_THRESHOLD = 0.05
N_CALLOUTS = 13  # includes CIN85/CD2AP SH3 domain binding motif (rank 13, padj=0.0065)
N_EXEMPLAR_GENES = 3
POINT_SIZE = 65

COLOR_EXC = "#FFB3FF"
COLOR_INH = "#1baf7a"
COLOR_NS = "#c9c9c9"

# ── Entry Name -> primary gene symbol ───────────────────────────────────────
entry_to_gene = {}
for path in GENE_MAP_FILES:
    g = pd.read_excel(path, usecols=["Entry Name", "Gene Names"])
    for _, row in g.iterrows():
        if pd.notna(row["Gene Names"]):
            entry_to_gene[str(row["Entry Name"])] = str(row["Gene Names"]).split()[0].upper()

# ── exemplar genes per class: pull hits, map sequence_name -> gene ─────────
xl = pd.ExcelFile(PRED_XL)
pred = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True).drop_duplicates(subset=["Motif Name"])
name_to_gene = dict(zip(pred["Motif Name"], pred["Entry Name"].map(entry_to_gene)))

fimo_hits = pd.read_excel(HITS_XL, sheet_name="Hits_FIMO")[["motif_id", "sequence_name", "q-value"]]
pdz_hits = pd.read_excel(HITS_XL, sheet_name="Hits_PDZ")[["motif_id", "sequence_name"]]
pdz_hits["q-value"] = 0.0
all_hits = pd.concat([fimo_hits, pdz_hits], ignore_index=True)
all_hits["Gene"] = all_hits["sequence_name"].map(name_to_gene)
all_hits = all_hits.dropna(subset=["Gene"]).sort_values("q-value")

def exemplar_genes(elm_id, n=N_EXEMPLAR_GENES):
    all_genes = all_hits.loc[all_hits["motif_id"] == elm_id, "Gene"].drop_duplicates().tolist()
    if not all_genes:
        return "(no gene-level hits recorded)"
    shown = ", ".join(all_genes[:n])
    return shown + ", ..." if len(all_genes) > n else shown


# ── differential result ─────────────────────────────────────────────────────
diff = pd.read_excel(SUMMARY_XL, sheet_name="Differential_All")


def direction(row):
    if not row["Significant"]:
        return "Not significant"
    return "Excitatory-enriched" if row["log2_FC"] > 0 else "Inhibitory-enriched"


diff["Direction"] = diff.apply(direction, axis=1)
diff["_neglog10_padj"] = -np.log10(diff["Fisher_padj_BH"].clip(lower=1e-12))
print(diff["Direction"].value_counts())

# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8.5))

color_map = {"Excitatory-enriched": COLOR_EXC, "Inhibitory-enriched": COLOR_INH, "Not significant": COLOR_NS}
for direction_label in ["Not significant", "Inhibitory-enriched", "Excitatory-enriched"]:
    sub = diff[diff["Direction"] == direction_label]
    if len(sub) == 0:
        continue
    ax.scatter(sub["log2_FC"], sub["_neglog10_padj"], s=POINT_SIZE, marker="o",
               color=color_map[direction_label], edgecolors="white", linewidths=0.4,
               alpha=0.85 if direction_label != "Not significant" else 0.55,
               zorder=4 if direction_label != "Not significant" else 2)

ax.axhline(-np.log10(SIG_THRESHOLD), color="#888888", linewidth=1.1, linestyle="--", zorder=1)
ax.axvline(0, color="#bbbbbb", linewidth=1.0, linestyle="-", zorder=1)
ax.text(0.995, -np.log10(SIG_THRESHOLD), f"padj = {SIG_THRESHOLD}  ", va="bottom", ha="right",
        transform=ax.get_yaxis_transform(), fontsize=8.5, color="#777777")

ax.set_xlabel("log₂ fold-change (Excitatory rate / Inhibitory rate, per residue scanned)", fontsize=11)
ax.set_ylabel("−log₁₀ (BH-FDR adjusted p, Fisher's exact)", fontsize=11)

n_exc_sig = (diff["Direction"] == "Excitatory-enriched").sum()
n_inh_sig = (diff["Direction"] == "Inhibitory-enriched").sum()
ax.set_title(
    "Differential SLiM Enrichment: Excitatory vs. Inhibitory Synapse Motifs (Volcano Plot)\n"
    f"{n_exc_sig} classes Excitatory-enriched, {n_inh_sig} Inhibitory-enriched, of {len(diff)} tested "
    "(padj<0.05, length-corrected, one point per SLiM class)",
    fontsize=12.5, fontweight="bold", pad=12,
)
ax.spines[["top", "right"]].set_visible(False)

# ── legend ───────────────────────────────────────────────────────────────────
color_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=COLOR_EXC,
           markeredgecolor="white", label=f"Excitatory-enriched (n={n_exc_sig})"),
    Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=COLOR_INH,
           markeredgecolor="white", label=f"Inhibitory-enriched (n={n_inh_sig})"),
    Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=COLOR_NS,
           markeredgecolor="white", alpha=0.7, label=f"Not significant (n={(diff['Direction']=='Not significant').sum()})"),
]
leg = ax.legend(handles=color_handles, title="Direction",
                 loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True,
                 fontsize=9.3, title_fontsize=9.8, framealpha=0.95, borderaxespad=0)
leg.get_frame().set_edgecolor("#ddd")

# ── callouts: fan LEFT and RIGHT into the whitespace flanking the cluster,
# each side its own short non-crossing column (plus PKA handled separately,
# top). This is a provably-safe generalisation of a single column: split the
# cluster by x into a left half and a right half; route the left half to a
# column just left of the cluster's leftmost point, the right half to a
# column just right of the cluster's rightmost point. Within each column,
# box order matches target significance order exactly (the same rule that
# guarantees no crossing in a single column) -- and because the two columns'
# lines live in disjoint x-ranges (left column -> left-half targets only,
# right column -> right-half targets only), the two fans can't cross each
# other either. Columns sit close to the cluster (not off in a far corner),
# so lines stay short.
reps = diff[diff["Significant"]].sort_values(["Fisher_padj_BH", "log2_FC"], ascending=[True, True]).head(N_CALLOUTS)

outlier = reps[reps["_neglog10_padj"] > 6]          # PKA -- far above the rest, handled separately
cluster = reps[reps["_neglog10_padj"] <= 6].copy().sort_values("log2_FC")

n = len(cluster)
n_left = n // 2
left_group = cluster.iloc[:n_left].sort_values("_neglog10_padj", ascending=False)
right_group = cluster.iloc[n_left:].sort_values("_neglog10_padj", ascending=False)

LEFT_X = cluster["log2_FC"].min() - 1.3
RIGHT_X = cluster["log2_FC"].max() + 1.3


def place_column(group, box_x, ha):
    if len(group) == 0:
        return
    y_lo = max(group["_neglog10_padj"].min() - 0.5, 0.1)
    y_hi = group["_neglog10_padj"].max() + 0.9
    box_ys = np.linspace(y_hi, y_lo, len(group))
    for (_, row), by in zip(group.iterrows(), box_ys):
        genes = exemplar_genes(row["ELM_Motif"])
        label = f"{row['Function']}\n{genes}"
        ax.annotate(
            label, xy=(row["log2_FC"], row["_neglog10_padj"]), xycoords="data",
            xytext=(box_x, by), textcoords="data",
            fontsize=7.6, ha=ha, va="center", family="monospace",
            bbox=dict(boxstyle="round,pad=0.32", fc="#fffefa", ec="#999", lw=0.7),
            arrowprops=dict(arrowstyle="-", color="#888", lw=0.8, connectionstyle="arc3,rad=0"),
            zorder=6,
        )


place_column(left_group, LEFT_X, "right")
place_column(right_group, RIGHT_X, "left")

for _, row in outlier.iterrows():
    genes = exemplar_genes(row["ELM_Motif"])
    label = f"{row['Function']}\n{genes}"
    ax.annotate(
        label, xy=(row["log2_FC"], row["_neglog10_padj"]), xycoords="data",
        xytext=(row["log2_FC"] + 1.1, row["_neglog10_padj"] - 0.15), textcoords="data",
        fontsize=7.6, ha="left", va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.32", fc="#fffefa", ec="#999", lw=0.7),
        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8, connectionstyle="arc3,rad=0"),
        zorder=6,
    )

# symmetric xlim so 0 (no fold-change) sits in the visual centre of the plot
left_needed = min(diff["log2_FC"].min(), LEFT_X) - 3.2
right_needed = RIGHT_X + 3.4
M = max(abs(left_needed), abs(right_needed))
ax.set_xlim(-M, M)
ax.set_ylim(bottom=-0.3)

caption = (
    "Source: slim_differential_enrichment_v3.xlsx -- Fisher's exact on hit rate per residue scanned (not per sequence;\n"
    "corrected for Excitatory motifs averaging 10.2 aa vs. Inhibitory's 8.4 aa, which biased the earlier per-sequence\n"
    "version), one pooled BH-FDR correction across 162 FIMO-testable ELM classes + 4 PDZ classes (tested via direct\n"
    "regex + true-C-terminus check since elm2meme drops PDZ classes from FIMO entirely). One point per class;\n"
    "callout genes are the most confident (lowest FIMO/regex match q-value) exemplars carrying each significant class."
)
fig.text(0.01, 0.01, caption, fontsize=7.5, color="#999999", linespacing=1.5)

fig.subplots_adjust(left=0.07, right=0.78, top=0.88, bottom=0.19)
plt.savefig(FIG_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {FIG_PNG}")
