"""
MEME suite + ELM-based short linear motif (SLiM) discovery and enrichment
analysis on excitatory vs inhibitory synapse motif sequences.

Pipeline:
  1. Prepare FASTA files (excitatory / inhibitory / both)
  2. Download ELM class regex patterns → elm2meme → MEME format DB
  3. STREME  : de novo SLiM discovery (excitatory primary, inhibitory control)
  4. AME     : enrichment of known ELM SLiMs in excitatory vs inhibitory
  5. FIMO    : scan all sequences for ELM hits, compare hit rates
  6. TOMTOM  : compare de novo STREME motifs against ELM DB
  7. Summary plots + Excel
"""

import os, subprocess, sys, io, re, json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR     = "/home/au729231/SynapseGigamapper/notebook/ESMC_outputs"
MEME_DIR    = f"{OUT_DIR}/meme_analysis"
INPUT_XL    = f"{OUT_DIR}/motif_biophysical_properties.xlsx"
ELM_TSV     = f"{MEME_DIR}/elm_classes.tsv"
ELM_MEME    = f"{MEME_DIR}/elm_classes.meme"
FASTA_EXC   = f"{MEME_DIR}/excitatory.fasta"
FASTA_INH   = f"{MEME_DIR}/inhibitory.fasta"
FASTA_ALL   = f"{MEME_DIR}/all_motifs.fasta"
STREME_DIR  = f"{MEME_DIR}/streme_exc_vs_inh"
AME_EXC_DIR = f"{MEME_DIR}/ame_excitatory"
AME_INH_DIR = f"{MEME_DIR}/ame_inhibitory"
FIMO_EXC    = f"{MEME_DIR}/fimo_excitatory"
FIMO_INH    = f"{MEME_DIR}/fimo_inhibitory"
TOMTOM_DIR  = f"{MEME_DIR}/tomtom_streme_vs_elm"
SUMMARY_XL  = f"{OUT_DIR}/slim_enrichment_summary.xlsx"
PLOT_PNG    = f"{OUT_DIR}/slim_enrichment_plot.png"

os.makedirs(MEME_DIR, exist_ok=True)

COLOR_MAP = {"Excitatory": "#FFB3FF", "Inhibitory": "#90EE90", "Both": "#6699FF"}

# ── 1. load sequences ─────────────────────────────────────────────────────────
print("Loading sequences...")
df = pd.read_excel(INPUT_XL)
exc = df[df["Synapse_Type"] == "Excitatory"].reset_index(drop=True)
inh = df[df["Synapse_Type"] == "Inhibitory"].reset_index(drop=True)
both = df[df["Synapse_Type"] == "Both"].reset_index(drop=True)
print(f"  Excitatory: {len(exc)}  Inhibitory: {len(inh)}  Both: {len(both)}")

def write_fasta(df_sub, path, label_col="Motif Name", seq_col="Motif Sequence"):
    with open(path, "w") as fh:
        for _, row in df_sub.iterrows():
            name = str(row[label_col]).replace(" ", "_").replace("/", "_")
            seq  = str(row[seq_col]).upper().strip()
            fh.write(f">{name}\n{seq}\n")
    print(f"  Written: {path}  ({len(df_sub)} sequences)")

write_fasta(exc,  FASTA_EXC)
write_fasta(inh,  FASTA_INH)
write_fasta(df,   FASTA_ALL)

# ── 2. download ELM and convert to MEME format ────────────────────────────────
print("\nDownloading ELM class definitions...")
ELM_URL = "http://elm.eu.org/elms/elms_index.tsv"
urllib.request.urlretrieve(ELM_URL, ELM_TSV)
elm_raw = pd.read_csv(ELM_TSV, sep="\t", comment="#",
                       names=["Accession","ELMIdentifier","FunctionalSiteName",
                               "Description","Regex","Probability",
                               "NumInstances","NumInstancesWithPDB"])
print(f"  {len(elm_raw)} ELM classes downloaded")

# Convert ELM regex TSV → MEME format using elm2meme
print("Converting ELM → MEME format with elm2meme...")
elm2meme_cmd = ["elm2meme", ELM_TSV]
result = subprocess.run(elm2meme_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  elm2meme warning: {result.stderr[:300]}")
with open(ELM_MEME, "w") as fh:
    fh.write(result.stdout)
motif_count = result.stdout.count("MOTIF ")
print(f"  {motif_count} motifs written to {ELM_MEME}")

# ── 3. STREME – de novo protein motif discovery ───────────────────────────────
print("\nRunning STREME (de novo discovery: excitatory vs inhibitory)...")
streme_cmd = [
    "streme", "--protein",
    "--p", FASTA_EXC,
    "--n", FASTA_INH,
    "--minw", "3", "--maxw", "12",
    "--thresh", "0.05",
    "--oc", STREME_DIR,
    "--verbosity", "1",
]
r = subprocess.run(streme_cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(f"  STREME stderr: {r.stderr[-500:]}")
else:
    print("  STREME complete")

streme_meme = f"{STREME_DIR}/streme.txt"

# ── 4. AME – enrichment of ELM SLiMs ─────────────────────────────────────────
def run_ame(primary_fasta, control_fasta, out_dir, label):
    print(f"\nRunning AME ({label})...")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ame",
        "--text",
        "--control", control_fasta,
        "--verbose", "1",
        "--method", "fisher",
        "--evalue-report-threshold", "100",
        primary_fasta, ELM_MEME,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tsv_path = f"{out_dir}/ame.tsv"
    with open(tsv_path, "w") as fh:
        fh.write(r.stdout)
    if r.returncode != 0:
        print(f"  AME warning ({label}): {r.stderr[-300:]}")
    else:
        lines = [l for l in r.stdout.splitlines() if not l.startswith("#") and l.strip()]
        print(f"  AME complete ({label}) — {max(0, len(lines)-1)} results")

run_ame(FASTA_EXC, FASTA_INH, AME_EXC_DIR, "Excitatory vs Inhibitory")
run_ame(FASTA_INH, FASTA_EXC, AME_INH_DIR, "Inhibitory vs Excitatory")

# ── 5. FIMO – scan all sequences for ELM motifs ───────────────────────────────
def run_fimo(fasta, out_dir, label):
    print(f"\nRunning FIMO ({label})...")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "fimo",
        "--thresh", "1e-3",
        "--oc", out_dir,
        "--verbosity", "2",
        "--bfile", "--motif--",
        ELM_MEME, fasta,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FIMO warning ({label}): {r.stderr[-300:]}")
    else:
        print(f"  FIMO complete ({label})")

run_fimo(FASTA_EXC, FIMO_EXC, "Excitatory")
run_fimo(FASTA_INH, FIMO_INH, "Inhibitory")

# ── 6. TOMTOM – de novo vs ELM ────────────────────────────────────────────────
if os.path.exists(streme_meme) and os.path.getsize(streme_meme) > 100:
    print("\nRunning TOMTOM (de novo vs ELM)...")
    cmd = [
        "tomtom",
        "--oc", TOMTOM_DIR,
        "--thresh", "0.5",
        "--evalue",
        "--verbosity", "2",
        streme_meme, ELM_MEME,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  TOMTOM warning: {r.stderr[-300:]}")
    else:
        print("  TOMTOM complete")
else:
    print("\nSkipping TOMTOM — no STREME motifs found")
    TOMTOM_DIR = None

# ── 7. parse results ───────────────────────────────────────────────────────────
def parse_ame(ame_dir):
    tsv = f"{ame_dir}/ame.tsv"
    if not os.path.exists(tsv):
        return pd.DataFrame()
    lines = [l for l in open(tsv).readlines() if not l.startswith("#") and l.strip()]
    if len(lines) < 2:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO("".join(lines)), sep="\t")
    except Exception:
        return pd.DataFrame()

def parse_fimo(fimo_dir):
    tsv = f"{fimo_dir}/fimo.tsv"
    if not os.path.exists(tsv):
        return pd.DataFrame()
    df_out = pd.read_csv(tsv, sep="\t", comment="#")
    df_out = df_out.dropna(subset=["motif_id"])
    return df_out

print("\nParsing results...")
ame_exc = parse_ame(AME_EXC_DIR)
ame_inh = parse_ame(AME_INH_DIR)
fimo_exc = parse_fimo(FIMO_EXC)
fimo_inh = parse_fimo(FIMO_INH)

print(f"  AME excitatory enrichments: {len(ame_exc)}")
print(f"  AME inhibitory enrichments: {len(ame_inh)}")
print(f"  FIMO excitatory hits: {len(fimo_exc)}")
print(f"  FIMO inhibitory hits: {len(fimo_inh)}")

# ── FIMO differential enrichment ─────────────────────────────────────────────
# For each ELM motif, count how many sequences in each group have ≥1 hit
def hit_matrix(fimo_df, n_seqs):
    if fimo_df.empty:
        return pd.DataFrame(columns=["n_seqs_with_hit", "frac"])
    counts = (fimo_df.groupby("motif_id")["sequence_name"]
                     .nunique()
                     .rename("n_seqs_with_hit")
                     .to_frame())
    counts["frac"] = counts["n_seqs_with_hit"] / n_seqs
    return counts

hits_exc = hit_matrix(fimo_exc, len(exc))
hits_inh = hit_matrix(fimo_inh, len(inh))

all_motifs = set(hits_exc.index) | set(hits_inh.index)
diff_rows = []
for motif in all_motifs:
    n_e = int(hits_exc.loc[motif, "n_seqs_with_hit"]) if motif in hits_exc.index else 0
    n_i = int(hits_inh.loc[motif, "n_seqs_with_hit"]) if motif in hits_inh.index else 0
    # Fisher's exact test: hit vs no-hit in exc vs inh
    table = [[n_e, len(exc) - n_e], [n_i, len(inh) - n_i]]
    _, p = stats.fisher_exact(table, alternative="two-sided")
    diff_rows.append({
        "ELM_Motif":       motif,
        "n_exc_seqs_hit":  int(n_e),
        "n_inh_seqs_hit":  int(n_i),
        "frac_exc":        round(n_e / len(exc), 4),
        "frac_inh":        round(n_i / len(inh), 4),
        "Fisher_p":        p,
        "log2_FC":         np.log2((n_e / len(exc) + 1e-9) / (n_i / len(inh) + 1e-9)),
    })

diff_df = pd.DataFrame(diff_rows)
if not diff_df.empty:
    _, padj, _, _ = multipletests(diff_df["Fisher_p"], method="fdr_bh")
    diff_df["Fisher_padj_BH"] = np.round(padj, 6)
    diff_df["Significant"]    = padj < 0.05
    diff_df = diff_df.sort_values("Fisher_padj_BH")
    print(f"\nDifferentially enriched ELM SLiMs (FDR<0.05): {diff_df['Significant'].sum()}")
    print(diff_df[diff_df["Significant"]].to_string(index=False))

# ── STREME motif summary ──────────────────────────────────────────────────────
streme_motifs = []
if os.path.exists(streme_meme):
    with open(streme_meme) as fh:
        content = fh.read()
    blocks = re.findall(r"^MOTIF\s+(\S+)\s+(\S+)", content, re.MULTILINE)
    for b in blocks:
        streme_motifs.append({"Motif_ID": b[0], "Alt_ID": b[1]})
    print(f"\nSTREME de novo motifs discovered: {len(streme_motifs)}")

# ── TOMTOM matches ────────────────────────────────────────────────────────────
tomtom_df = pd.DataFrame()
if TOMTOM_DIR and os.path.exists(f"{TOMTOM_DIR}/tomtom.tsv"):
    tomtom_df = pd.read_csv(f"{TOMTOM_DIR}/tomtom.tsv", sep="\t", comment="#")
    tomtom_df = tomtom_df.dropna(subset=["Query_ID"])
    print(f"TOMTOM matches: {len(tomtom_df)}")

# ── save Excel ────────────────────────────────────────────────────────────────
print(f"\nWriting summary → {SUMMARY_XL}")
with pd.ExcelWriter(SUMMARY_XL, engine="openpyxl") as writer:
    for sheet, data in [
        ("FIMO_Differential",       diff_df if not diff_df.empty else pd.DataFrame()),
        ("AME_Excitatory",          ame_exc),
        ("AME_Inhibitory",          ame_inh),
        ("STREME_DeNovo",           pd.DataFrame(streme_motifs)),
        ("TOMTOM_DeNovo_vs_ELM",    tomtom_df),
        ("FIMO_Hits_Excitatory",    fimo_exc),
        ("FIMO_Hits_Inhibitory",    fimo_inh),
    ]:
        if data is not None and len(data):
            data.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for col in ws.columns:
                ml = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(ml + 2, 45)
        else:
            pd.DataFrame({"No_data": []}).to_excel(writer, sheet_name=sheet, index=False)

# ── plots ─────────────────────────────────────────────────────────────────────
print("Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Short Linear Motif (SLiM) Enrichment Analysis\n"
             "Excitatory vs Inhibitory Synapse Motifs",
             fontsize=13, fontweight="bold")

# Panel 1: Volcano plot of FIMO differential enrichment
ax = axes[0]
if not diff_df.empty:
    sig   = diff_df["Significant"]
    exc_e = diff_df["log2_FC"] > 0
    colors_vol = np.where(sig & exc_e,  "#FFB3FF",
                 np.where(sig & ~exc_e, "#90EE90", "lightgrey"))
    ax.scatter(diff_df["log2_FC"], -np.log10(diff_df["Fisher_padj_BH"] + 1e-10),
               c=colors_vol, s=40, alpha=0.8, linewidths=0.3, edgecolors="grey")
    ax.axhline(-np.log10(0.05), color="black", lw=1, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    # label top hits
    top = pd.concat([diff_df[sig].nlargest(8, "log2_FC"),
                     diff_df[sig].nsmallest(8, "log2_FC")])
    for _, row in top.iterrows():
        ax.text(row["log2_FC"], -np.log10(row["Fisher_padj_BH"] + 1e-10) + 0.1,
                row["ELM_Motif"], fontsize=5.5, ha="center", va="bottom")
    patches_vol = [
        mpatches.Patch(color="#FFB3FF", label="Excitatory-enriched (FDR<0.05)"),
        mpatches.Patch(color="#90EE90", label="Inhibitory-enriched (FDR<0.05)"),
        mpatches.Patch(color="lightgrey", label="Not significant"),
    ]
    ax.legend(handles=patches_vol, fontsize=7, loc="upper left")
ax.set_xlabel("log₂ Fold-Change (Excitatory / Inhibitory)", fontsize=10)
ax.set_ylabel("−log₁₀ (FDR-adjusted p)", fontsize=10)
ax.set_title("FIMO: ELM Motif Enrichment\n(Fisher's exact, BH-FDR)", fontsize=10)
sns.despine(ax=ax)

# Panel 2: Top AME enrichments bar chart
ax2 = axes[1]
top_ame = pd.DataFrame()
if not ame_exc.empty and "motif_ID" in ame_exc.columns:
    p_col = "p-value" if "p-value" in ame_exc.columns else ame_exc.columns[-2]
    top_ame_e = ame_exc.rename(columns={"motif_ID": "ELM", p_col: "p"}).nsmallest(10, "p")
    top_ame_e["group"] = "Excitatory"
    top_ame_i = ame_inh.rename(columns={"motif_ID": "ELM", p_col: "p"}).nsmallest(10, "p")
    top_ame_i["group"] = "Inhibitory"
    top_ame = pd.concat([top_ame_e, top_ame_i], ignore_index=True)
    top_ame = top_ame.dropna(subset=["p"])

if not top_ame.empty:
    top_ame["-log10p"] = -np.log10(top_ame["p"] + 1e-10)
    top_ame_sorted = top_ame.sort_values("-log10p", ascending=True)
    colors_bar = [COLOR_MAP[g] for g in top_ame_sorted["group"]]
    ax2.barh(range(len(top_ame_sorted)), top_ame_sorted["-log10p"],
             color=colors_bar, edgecolor="grey", linewidth=0.4)
    ax2.set_yticks(range(len(top_ame_sorted)))
    ax2.set_yticklabels(top_ame_sorted["ELM"].str[:25], fontsize=7)
    ax2.axvline(-np.log10(0.05), color="black", lw=1, ls="--", alpha=0.6)
    patches_bar = [mpatches.Patch(color=COLOR_MAP[g], label=g) for g in ["Excitatory","Inhibitory"]]
    ax2.legend(handles=patches_bar, fontsize=8)
else:
    ax2.text(0.5, 0.5, "No AME results\n(insufficient sequences or\nno significant hits)",
             ha="center", va="center", transform=ax2.transAxes, fontsize=10, color="grey")
ax2.set_xlabel("−log₁₀ (p-value)", fontsize=10)
ax2.set_title("AME: Top ELM SLiM Enrichments\n(Fisher's exact)", fontsize=10)
sns.despine(ax=ax2)

# Panel 3: FIMO hit fraction comparison (top differentially enriched)
ax3 = axes[2]
if not diff_df.empty and diff_df["Significant"].any():
    top_diff = diff_df[diff_df["Significant"]].head(15).copy()
    x = np.arange(len(top_diff))
    w = 0.35
    ax3.barh(x + w/2, top_diff["frac_exc"], w,
             color="#FFB3FF", edgecolor="grey", linewidth=0.4, label="Excitatory")
    ax3.barh(x - w/2, top_diff["frac_inh"], w,
             color="#90EE90", edgecolor="grey", linewidth=0.4, label="Inhibitory")
    ax3.set_yticks(x)
    ax3.set_yticklabels(top_diff["ELM_Motif"].str[:25], fontsize=7)
    ax3.set_xlabel("Fraction of sequences with ≥1 hit", fontsize=10)
    ax3.set_title("Top Differential ELM SLiMs\n(FIMO hits, FDR<0.05)", fontsize=10)
    ax3.legend(fontsize=8)
elif not diff_df.empty:
    top15 = diff_df.head(15).copy()
    x = np.arange(len(top15))
    w = 0.35
    ax3.barh(x + w/2, top15["frac_exc"], w,
             color="#FFB3FF", edgecolor="grey", linewidth=0.4, label="Excitatory")
    ax3.barh(x - w/2, top15["frac_inh"], w,
             color="#90EE90", edgecolor="grey", linewidth=0.4, label="Inhibitory")
    ax3.set_yticks(x)
    ax3.set_yticklabels(top15["ELM_Motif"].str[:25], fontsize=7)
    ax3.set_xlabel("Fraction of sequences with ≥1 hit", fontsize=10)
    ax3.set_title("Top ELM SLiMs by Fisher p\n(FIMO hits, not FDR significant)", fontsize=10)
    ax3.legend(fontsize=8)
else:
    ax3.text(0.5, 0.5, "No FIMO results", ha="center", va="center",
             transform=ax3.transAxes, fontsize=10, color="grey")
sns.despine(ax=ax3)

plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=200, bbox_inches="tight")
print(f"Plot saved → {PLOT_PNG}")
print(f"Summary Excel → {SUMMARY_XL}")
print("\nDone.")
