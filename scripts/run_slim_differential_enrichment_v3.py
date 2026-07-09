"""
Differential SLiM enrichment, v3: fixes a length confound found in v2.

v2's Fisher's test compared "fraction of SEQUENCES with >=1 hit" between
Excitatory and Inhibitory groups. But Excitatory motifs average 10.2 aa vs.
Inhibitory's 8.4 aa (checked directly) -- longer sequences mechanically have
more chances to contain any given short motif at least once, independent of
real biology. That confound is the most likely explanation for v2's result
being suspiciously one-sided: all 48 significant classes favored Excitatory,
zero favored Inhibitory.

Fix: compare hit RATE PER RESIDUE SCANNED instead of per-sequence presence/
absence -- [total hit count, total residues - hits] vs the same for the
other group, Fisher's exact. This treats "how many hits landed, out of how
many amino acid positions were available for a hit to land on" as the
comparison, which is invariant to the two groups having different average
sequence lengths (a real biological rate difference isn't).

PDZ classes are unaffected by this fix: their test already compares hit
count over a position-defined candidate set (motifs within 10aa of the true
C-terminus), not sequence count, so it isn't subject to the same length bias
-- kept as computed in v2.

Output: slim_differential_enrichment_v3.xlsx
          - Differential_All (rate-based Fisher's test, FIMO classes;
            PDZ classes carried over unchanged from v2)
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT_DIR = "/home/au729231/SynapseGigamapper/notebook/ESMC_outputs"
MEME_DIR = f"{OUT_DIR}/meme_analysis"
PRED_XL = f"{OUT_DIR}/motif_predictions.xlsx"
FIMO_TSV = f"{MEME_DIR}/fimo_all_unfiltered/fimo.tsv"
ELM_TSV = f"{MEME_DIR}/elm_classes.tsv"
V2_XL = f"{OUT_DIR}/slim_differential_enrichment_v2.xlsx"
OUT_XL = f"{OUT_DIR}/slim_differential_enrichment_v3.xlsx"

xl = pd.ExcelFile(PRED_XL)
dfs = [xl.parse(s) for s in xl.sheet_names]
df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Motif Name"]).reset_index(drop=True)


def classify(row):
    e, i = row["Excitatory Synapse_Score"], row["Inhibitory Synapse_Score"]
    if e >= 0.5 and i >= 0.5:
        return "Both"
    if e >= 0.5:
        return "Excitatory"
    if i >= 0.5:
        return "Inhibitory"
    c = str(row["Compartment"])
    return "Excitatory" if "Excitatory" in c else "Inhibitory"


df["Synapse_Type"] = df.apply(classify, axis=1)
exc_names = set(df.loc[df["Synapse_Type"] == "Excitatory", "Motif Name"])
inh_names = set(df.loc[df["Synapse_Type"] == "Inhibitory", "Motif Name"])
len_lookup = df.set_index("Motif Name")["Motif Length"]

total_len_exc = int(len_lookup.loc[list(exc_names)].sum())
total_len_inh = int(len_lookup.loc[list(inh_names)].sum())
n_exc, n_inh = len(exc_names), len(inh_names)
print(f"Excitatory: {n_exc} motifs, {total_len_exc} total residues scanned (mean {total_len_exc/n_exc:.2f} aa)")
print(f"Inhibitory: {n_inh} motifs, {total_len_inh} total residues scanned (mean {total_len_inh/n_inh:.2f} aa)")

# ── FIMO hits: use raw hit COUNT (not unique-sequence count) per class ─────
fimo_all = pd.read_csv(FIMO_TSV, sep="\t", comment="#").dropna(subset=["motif_id"])
fimo_exc = fimo_all[fimo_all["sequence_name"].isin(exc_names)]
fimo_inh = fimo_all[fimo_all["sequence_name"].isin(inh_names)]

counts_exc = fimo_exc.groupby("motif_id").size().rename("n_hits_exc")
counts_inh = fimo_inh.groupby("motif_id").size().rename("n_hits_inh")
all_ids = set(counts_exc.index) | set(counts_inh.index)

rows = []
for motif in all_ids:
    n_e = int(counts_exc.get(motif, 0))
    n_i = int(counts_inh.get(motif, 0))
    # rate-based 2x2: hits vs non-hit residue-positions, per group
    table = [[n_e, total_len_exc - n_e], [n_i, total_len_inh - n_i]]
    _, p = stats.fisher_exact(table, alternative="two-sided")
    rate_e = n_e / total_len_exc
    rate_i = n_i / total_len_inh
    rows.append({
        "ELM_Motif": motif, "Method": "FIMO",
        "n_hits_exc": n_e, "n_hits_inh": n_i,
        "rate_exc_per_1000aa": round(rate_e * 1000, 4), "rate_inh_per_1000aa": round(rate_i * 1000, 4),
        "Fisher_p": p, "log2_FC": np.log2((rate_e + 1e-9) / (rate_i + 1e-9)),
    })
diff_df = pd.DataFrame(rows)

# ── carry over the PDZ rows from v2 unchanged (already position-normalized) ─
v2 = pd.read_excel(V2_XL, sheet_name="Differential_All")
pdz_v2 = v2[v2["Method"] == "PDZ_regex"][["ELM_Motif", "Method", "n_exc_seqs_hit", "n_inh_seqs_hit", "Fisher_p", "log2_FC"]]
pdz_v2 = pdz_v2.rename(columns={"n_exc_seqs_hit": "n_hits_exc", "n_inh_seqs_hit": "n_hits_inh"})
pdz_v2["rate_exc_per_1000aa"] = None
pdz_v2["rate_inh_per_1000aa"] = None

diff_df = pd.concat([diff_df, pdz_v2], ignore_index=True)

# ── ONE pooled BH-FDR correction ────────────────────────────────────────────
_, padj, _, _ = multipletests(diff_df["Fisher_p"], method="fdr_bh")
diff_df["Fisher_padj_BH"] = padj  # full precision -- rounding to 6dp previously
                                   # zeroed out a true ~9.5e-10 value, badly
                                   # distorting any -log10(padj) plot of this
diff_df["Significant"] = padj < 0.05
diff_df = diff_df.sort_values("Fisher_padj_BH").reset_index(drop=True)

elm_meta = pd.read_csv(ELM_TSV, sep="\t", comment="#",
    names=["Accession", "ELMIdentifier", "FunctionalSiteName", "Description",
           "Regex", "Probability", "NumInstances", "NumInstancesWithPDB"])
fn_map = elm_meta.set_index("Accession")["FunctionalSiteName"].to_dict()
pdz_fn = {"LIG_PDZ_Class_1": "PDZ domain ligands", "LIG_PDZ_Class_2": "PDZ domain ligands",
          "LIG_PDZ_Class_3": "PDZ domain ligands", "LIG_PDZ_Wminus1_1": "PDZ domain ligands"}
fn_map.update(pdz_fn)
diff_df["Function"] = diff_df["ELM_Motif"].map(fn_map).fillna(diff_df["ELM_Motif"])

n_sig = diff_df["Significant"].sum()
n_exc_enriched = (diff_df["Significant"] & (diff_df["log2_FC"] > 0)).sum()
n_inh_enriched = (diff_df["Significant"] & (diff_df["log2_FC"] < 0)).sum()
print(f"\nTotal classes tested: {len(diff_df)}  |  significant: {n_sig}")
print(f"  Excitatory-enriched: {n_exc_enriched}  |  Inhibitory-enriched: {n_inh_enriched}")
print(f"\n{diff_df[diff_df['Significant']][['ELM_Motif','Method','Function','log2_FC','Fisher_padj_BH']].to_string(index=False)}")

with pd.ExcelWriter(OUT_XL, engine="openpyxl") as writer:
    diff_df.to_excel(writer, sheet_name="Differential_All", index=False)
    for ws in writer.sheets.values():
        for col in ws.columns:
            ml = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(ml + 2, 45)
print(f"\nSaved -> {OUT_XL}")
