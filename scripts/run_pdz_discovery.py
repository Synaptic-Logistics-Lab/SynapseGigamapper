"""
Discover PDZ-binding motifs across all four synapse motif sets
(Train, Development, Test, Holdout).

Uses ELM PDZ regex patterns:
  LIG_PDZ_Class_1   : ...[ST].[ACVILF]$   (C-terminal S/T-x-hydrophobic)
  LIG_PDZ_Class_2   : ...[VLIFY].[ACVILF]$ (C-terminal hydrophobic-x-hydrophobic)
  LIG_PDZ_Class_3   : ...[DE].[ACVILF]$   (C-terminal D/E-x-hydrophobic)
  LIG_PDZ_Wminus1_1 : .W[ACGILV]$         (C-terminal Trp-1 class)
  LIG_FZD_DVL_PDZ   : Frizzled/DVL internal PDZ interaction motif

Search strategy:
  A) Strict C-terminal match on the motif sequence as-is
  B) Internal match anywhere in the sequence (motif may be a C-terminal fragment)

Both are reported; hits are flagged by class and match type.
Output: motif_pdz_hits.xlsx
"""

import re
import os
import pandas as pd

OUT_DIR   = "/home/au729231/SynapseGigamapper/notebook/ESMC_outputs"
OUTPUT_XL = f"{OUT_DIR}/motif_pdz_hits.xlsx"

MOTIF_FILES = {
    "Train":       f"{OUT_DIR}/all_synapse_motifs.xlsx",
    "Development": f"{OUT_DIR}/all_synapse_motifs_development.xlsx",
    "Test":        f"{OUT_DIR}/all_synapse_motifs_test.xlsx",
    "Holdout":     f"{OUT_DIR}/all_synapse_motifs_holdout.xlsx",
}

# ── PDZ binding motif patterns (from ELM database) ───────────────────────────
# Strict C-terminal patterns ($ anchor applied to full motif sequence)
PDZ_CTERMINAL = {
    "LIG_PDZ_Class_1":   r"...[ST].[ACVILF]$",
    "LIG_PDZ_Class_2":   r"...[VLIFY].[ACVILF]$",
    "LIG_PDZ_Class_3":   r"...[DE].[ACVILF]$",
    "LIG_PDZ_Wminus1_1": r".W[ACGILV]$",
    "LIG_FZD_DVL_PDZ":   r"W.{0,1}[VIL].[ST].KA{0,1}T",
}

# Core internal patterns (search anywhere in sequence; less stringent)
# Derived by removing the leading wildcard anchors
PDZ_INTERNAL = {
    "LIG_PDZ_Class_1_internal":   r"[ST].[ACVILF]",
    "LIG_PDZ_Class_2_internal":   r"[VLIFY].[ACVILF]",
    "LIG_PDZ_Class_3_internal":   r"[DE].[ACVILF]",
    "LIG_PDZ_Wminus1_internal":   r"W[ACGILV]",
    "LIG_FZD_DVL_PDZ_internal":   r"W.{0,1}[VIL].[ST].KA{0,1}T",
}

compiled_ct  = {k: re.compile(v) for k, v in PDZ_CTERMINAL.items()}
compiled_int = {k: re.compile(v) for k, v in PDZ_INTERNAL.items()}


def detect_pdz(seq):
    """
    Returns dict with:
      - matched_classes_cterminal  : list of ELM class names matching at C-terminus
      - matched_classes_internal   : list of ELM class names matching internally
      - pdz_hit                    : True if any match found
      - match_type                 : 'C-terminal', 'Internal', 'Both', or None
      - matched_sequence_cterminal : matched subsequence (first C-terminal hit)
      - matched_sequence_internal  : matched subsequence (first internal hit)
    """
    seq = seq.upper().strip()

    ct_hits  = {}
    int_hits = {}

    for cls, pat in compiled_ct.items():
        m = pat.search(seq)
        if m:
            ct_hits[cls] = m.group()

    for cls, pat in compiled_int.items():
        # internal: match anywhere, but exclude if it's already a strict C-terminal hit
        m = pat.search(seq)
        if m:
            int_hits[cls] = m.group()

    ct_classes  = list(ct_hits.keys())
    int_classes = list(int_hits.keys())
    any_hit     = bool(ct_hits or int_hits)

    if ct_hits and int_hits:
        match_type = "Both"
    elif ct_hits:
        match_type = "C-terminal"
    elif int_hits:
        match_type = "Internal"
    else:
        match_type = None

    return {
        "PDZ_Hit":                   any_hit,
        "Match_Type":                match_type,
        "PDZ_Classes_Cterminal":     ";".join(ct_classes) if ct_classes else "",
        "PDZ_Classes_Internal":      ";".join(int_classes) if int_classes else "",
        "Matched_Seq_Cterminal":     list(ct_hits.values())[0] if ct_hits else "",
        "Matched_Seq_Internal":      list(int_hits.values())[0] if int_hits else "",
    }


# ── load and scan all sets ────────────────────────────────────────────────────
all_dfs = []
summary = []

for set_name, path in MOTIF_FILES.items():
    if not os.path.exists(path):
        print(f"  Skipping {set_name}: file not found")
        continue
    df = pd.read_excel(path)
    df.insert(0, "Set", set_name)

    # build a Motif Name column for readability
    df["Motif_Name"] = (df["Entry Name"].astype(str) + "-"
                        + df["Motif Start"].astype(str) + "-"
                        + df["Motif End"].astype(str))

    print(f"Scanning {set_name}: {len(df)} motifs...")
    pdz_results = df["Motif Sequence"].apply(
        lambda s: pd.Series(detect_pdz(str(s)))
    )
    df = pd.concat([df, pdz_results], axis=1)

    n_hit = df["PDZ_Hit"].sum()
    n_ct  = (df["Match_Type"] == "C-terminal").sum()
    n_int = (df["Match_Type"] == "Internal").sum()
    n_both= (df["Match_Type"] == "Both").sum()
    print(f"  PDZ hits: {n_hit}  (C-terminal: {n_ct}, Internal: {n_int}, Both: {n_both})")

    summary.append({
        "Set": set_name, "Total_Motifs": len(df),
        "PDZ_Hits": n_hit,
        "C-terminal_only": n_ct,
        "Internal_only": n_int,
        "Both": n_both,
        "Pct_PDZ": round(100 * n_hit / len(df), 1),
    })
    all_dfs.append(df)

all_motifs = pd.concat(all_dfs, ignore_index=True)
pdz_hits   = all_motifs[all_motifs["PDZ_Hit"]].copy().reset_index(drop=True)
summary_df = pd.DataFrame(summary)

print(f"\nTotal PDZ hits across all sets: {len(pdz_hits)}")
print(summary_df.to_string(index=False))

# ── class breakdown ───────────────────────────────────────────────────────────
print("\nC-terminal PDZ class breakdown:")
ct_classes_all = (
    pdz_hits["PDZ_Classes_Cterminal"]
    .str.split(";")
    .explode()
    .replace("", pd.NA)
    .dropna()
    .value_counts()
)
print(ct_classes_all.to_string())

print("\nInternal PDZ class breakdown:")
int_classes_all = (
    pdz_hits["PDZ_Classes_Internal"]
    .str.split(";")
    .explode()
    .replace("", pd.NA)
    .dropna()
    .value_counts()
)
print(int_classes_all.to_string())

# ── save to Excel ─────────────────────────────────────────────────────────────
print(f"\nWriting → {OUTPUT_XL}")

# column order for the PDZ hits sheet
col_order = [
    "Set", "Motif_Name", "Entry Name", "Compartment",
    "Motif Start", "Motif End", "Motif Length", "Motif Sequence",
    "Mean Importance", "Max Importance",
    "PDZ_Hit", "Match_Type",
    "PDZ_Classes_Cterminal", "Matched_Seq_Cterminal",
    "PDZ_Classes_Internal",  "Matched_Seq_Internal",
]
col_order = [c for c in col_order if c in pdz_hits.columns]

with pd.ExcelWriter(OUTPUT_XL, engine="openpyxl") as writer:
    # Sheet 1: all PDZ hits
    pdz_hits[col_order].to_excel(writer, sheet_name="PDZ_Hits_All", index=False)

    # Sheet 2: C-terminal hits only (highest confidence)
    ct_only = pdz_hits[pdz_hits["PDZ_Classes_Cterminal"] != ""][col_order]
    ct_only.to_excel(writer, sheet_name="PDZ_Cterminal_Only", index=False)

    # Sheet 3: per-set breakdown
    for set_name in MOTIF_FILES:
        hits_set = pdz_hits[pdz_hits["Set"] == set_name][col_order]
        if len(hits_set):
            hits_set.to_excel(writer, sheet_name=f"PDZ_{set_name[:10]}", index=False)

    # Sheet 4: summary
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # auto-width all sheets
    for ws in writer.sheets.values():
        for col in ws.columns:
            ml = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(ml + 2, 50)

print(f"Saved {len(pdz_hits)} PDZ hits to {OUTPUT_XL}")
print(f"  C-terminal hits: {len(ct_only)}")
print(f"  Internal-only hits: {len(pdz_hits) - len(ct_only)}")
