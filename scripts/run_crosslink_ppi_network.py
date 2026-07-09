"""
Build and visualize the inter-protein crosslink-validated PPI network:
genes containing exact-crosslink-matched motifs, linked to their crosslink
interactor genes, coloured by their SynGO GO term (Cellular Component /
Biological Process, from the SynGO_2024 gene set database) -- genes not
annotated in SynGO at all are coloured black ("Unannotated in SynGO").

Inputs:
  - crosslink_motif_interactors.xlsx / Inter_Protein_Only, Inter_Protein_Pair_Summary
    (built by run_crosslink_interactor_lookup.py)
  - SynGO_2024.gmt (predictions dir) -- SynGO GO term -> gene set

Outputs:
  - crosslink_ppi_network.xlsx        (Nodes, Edges, Pathway_Assignment sheets)
  - interactive_html/crosslink_ppi_network.html  (vis-network, offline HTML)
"""

import json
import os
import re
import pandas as pd

OUT_DIR    = "/home/au729231/SynapseGigamapper/notebook/ESMC_outputs"
INTERACTOR_XL = f"{OUT_DIR}/crosslink_motif_interactors.xlsx"
SYNGO_GMT  = f"{OUT_DIR}/predictions/SynGO_2024.gmt"
STATS_XL   = f"{OUT_DIR}/crosslink_ppi_network.xlsx"
HTML_DIR   = f"{OUT_DIR}/interactive_html"
HTML_OUT   = f"{HTML_DIR}/crosslink_ppi_network.html"
VIS_JS     = f"{HTML_DIR}/vis-network.min.js"
VIS_CDN    = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

TOP_N_PATHWAYS = 12   # distinct SynGO-term colour categories in the legend; rest -> "Other"

os.makedirs(HTML_DIR, exist_ok=True)

# ── load edges ──────────────────────────────────────────────────────────────────
edges_detail = pd.read_excel(INTERACTOR_XL, sheet_name="Inter_Protein_Only")
pair_summary = pd.read_excel(INTERACTOR_XL, sheet_name="Inter_Protein_Pair_Summary")

edges_detail["GeneA"] = edges_detail["Gene Name"].astype(str).str.upper()
edges_detail["GeneB"] = edges_detail["Interactor_Gene"].astype(str).str.upper()

# aggregate edge list: one row per unordered gene pair, with supporting evidence
def pair_key(a, b):
    return tuple(sorted([a, b]))

edges_detail["PairKey"] = edges_detail.apply(lambda r: pair_key(r["GeneA"], r["GeneB"]), axis=1)
edge_rows = []
for pk, grp in edges_detail.groupby("PairKey"):
    g1, g2 = pk
    motifs_evidence = "; ".join(sorted(set(
        f"{row['Motif Sequence']} ({row['Entry Name']} {row['Motif Start']}-{row['Motif End']})"
        for _, row in grp.iterrows()
    )))
    edge_rows.append({
        "Gene1": g1, "Gene2": g2,
        "N_Motif_Links": len(grp),
        "Supporting_Motifs": motifs_evidence,
    })
edges = pd.DataFrame(edge_rows).sort_values("N_Motif_Links", ascending=False).reset_index(drop=True)
print(f"Edges (unique gene pairs): {len(edges)}")

nodes_set = sorted(set(edges["Gene1"]) | set(edges["Gene2"]))
print(f"Nodes (unique genes): {len(nodes_set)}")

# genes that carry an exact-crosslink-matched motif themselves ("Gene Name" column)
# vs. genes that only appear as someone else's interactor and carry no motif of
# their own -- the latter get drawn as hollow circles in the network, and the
# former get a circle size scaled by how many distinct motifs they carry.
motif_carriers = set(edges_detail["GeneA"])
has_motif = {n: (n in motif_carriers) for n in nodes_set}
print(f"Genes with their own motif: {sum(has_motif.values())} / {len(nodes_set)} "
      f"(interactor-only, no motif: {len(nodes_set) - sum(has_motif.values())})")

n_motifs = (
    edges_detail.groupby("GeneA")[["Entry Name", "Motif Start", "Motif End"]]
    .apply(lambda g: g.drop_duplicates().shape[0])
    .to_dict()
)
motif_count = {n: n_motifs.get(n, 0) for n in nodes_set}

degree = {}
for _, r in edges.iterrows():
    degree[r["Gene1"]] = degree.get(r["Gene1"], 0) + 1
    degree[r["Gene2"]] = degree.get(r["Gene2"], 0) + 1

# ── assign each gene its SynGO GO term (direct database membership, not a
# statistical enrichment test) ─────────────────────────────────────────────────
# SynGO_2024.gmt term names carry a trailing "(GO:0000000) CC"/"BP" suffix;
# split off the category, then split off the GO id so the id can go in the
# tooltip while the legend/label shows the function descriptor only.
syngo_terms = {}  # clean_name -> {"category", "go_id", "genes", "size"}
with open(SYNGO_GMT) as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        term_field = parts[0].strip()
        genes = {g.strip().upper() for g in parts[2:] if g.strip()}
        if not genes:
            continue
        m = re.search(r"\b(BPp|BP|CC)$", term_field)
        if not m:
            continue
        category = "BP" if m.group(1).startswith("BP") else "CC"
        display_name = term_field[: m.start()].strip()
        m2 = re.match(r"^(.*)\s\((GO:\d+)\)$", display_name)
        clean_name, go_id = (m2.group(1), m2.group(2)) if m2 else (display_name, "")
        syngo_terms[clean_name] = {"category": category, "go_id": go_id, "genes": genes, "size": len(genes)}
print(f"\nSynGO_2024.gmt: {len(syngo_terms)} GO terms")

# composite/ambiguous symbols (e.g. "CALM1;CALM2;CALM3") -> any member counts
def gene_parts(node):
    return [p.upper() for p in str(node).split(";")]

# each gene may belong to many SynGO terms -- prefer Cellular Component
# (physical location, e.g. "integral component of postsynaptic membrane")
# over Biological Process (activity/function, e.g. "ligand-gated ion channel
# activity involved in regulation of X membrane potential") when a gene has
# both, since CC is what a reader intuitively expects from "annotation". The
# remaining tie-break is smallest GLOBAL (SynGO database-wide) gene-set size
# -- the most specific term over the most generic ("synapse", "postsynapse").
#
# This default rule mislabels GRIA1-4 as presynaptic: their smallest global
# CC term is "presynaptic active zone membrane" (52 genes) even though their
# well-known primary role is postsynaptic. Tried several universal
# alternatives (largest-global, smallest-local, largest-local) -- none work
# across the board; e.g. smallest-local fixes GRIA but then mislabels
# genuinely presynaptic proteins like VAMP2/SYT1 as postsynaptic, because
# "fewest co-occurrences" just means "rare annotation," not "primary/
# best-known role." So this is a targeted override for the GRIA family only:
# use smallest LOCAL (within this network) gene-set size for those genes
# specifically, where it happens to correctly resolve to postsynaptic
# membrane (10 other network genes) over the presynaptic alternatives
# (14-15) -- every other gene keeps the smallest-global default.
CATEGORY_RANK = {"CC": 0, "BP": 1}
# same "postsynaptic receptor subunit with a smaller/rarer presynaptic
# co-annotation" situation as GRIA1-4: GRIN1/GRIN2B's smallest-local CC term
# ties at 14 between "Postsynaptic Density Membrane" and "Presynaptic
# Membrane", which the alphabetical final tie-break resolves to postsynaptic.
LOCAL_SPECIFICITY_OVERRIDE = {"GRIA1", "GRIA2", "GRIA3", "GRIA4", "GRIN1", "GRIN2B"}

# Neither smallest-global nor smallest-local resolves every gene correctly --
# VAMP2 (synaptobrevin-2, the canonical presynaptic vesicle SNARE) has
# "Postsynaptic Cytosol" as its smallest term BOTH globally (51 genes) and
# locally (8 genes in this network), narrowly beating its actual textbook
# annotation "Integral Component Of Synaptic Vesicle Membrane" (56 global /
# 11 local) either way. Rather than chase another universal metric, force
# specific genes to a specific (verified-present-in-their-match-list) term.
# SYT1 (synaptotagmin-1, the presynaptic Ca2+ sensor for vesicle fusion) has
# the same problem, resolved the same way -- it too is a canonical integral
# synaptic-vesicle-membrane protein.
TERM_OVERRIDE = {
    "VAMP2": "Integral Component Of Synaptic Vesicle Membrane",
    "SYT1": "Integral Component Of Synaptic Vesicle Membrane",
}

node_to_pathway = {}
node_to_goid = {}
node_matches = {}
for node in nodes_set:
    parts = gene_parts(node)
    node_matches[node] = [(tname, info["category"], info["size"], info["go_id"]) for tname, info in syngo_terms.items()
                           if any(p in info["genes"] for p in parts)]

local_term_count = pd.Series(
    [tname for matches in node_matches.values() for tname, _, _, _ in matches]
).value_counts().to_dict()

for node in nodes_set:
    matches = node_matches[node]
    if matches:
        forced_term = next((p for p in gene_parts(node) if p in TERM_OVERRIDE), None)
        forced = [m for m in matches if forced_term and m[0] == TERM_OVERRIDE[forced_term]]
        if forced:
            best = forced[0]
        else:
            use_local = any(p in LOCAL_SPECIFICITY_OVERRIDE for p in gene_parts(node))
            specificity = (lambda x: local_term_count[x[0]]) if use_local else (lambda x: x[2])
            best = sorted(matches, key=lambda x: (CATEGORY_RANK.get(x[1], 9), specificity(x), x[0]))[0]
        node_to_pathway[node] = best[0]
        node_to_goid[node] = best[3]
    else:
        node_to_pathway[node] = "Unannotated in SynGO"
        node_to_goid[node] = ""

n_assigned = sum(1 for v in node_to_pathway.values() if v != "Unannotated in SynGO")
print(f"  {n_assigned}/{len(nodes_set)} genes annotated in SynGO")

# keep only the top-N most-populous SynGO terms as distinct colour categories
term_counts = pd.Series(list(node_to_pathway.values())).value_counts()
top_terms = [t for t in term_counts.index if t != "Unannotated in SynGO"][:TOP_N_PATHWAYS]
print(f"\nTop SynGO term categories (by # genes assigned):")
print(term_counts.head(TOP_N_PATHWAYS + 1).to_string())

def bucket(term):
    # "group"/legend bucketing only -- caps the legend at the top-N most
    # populous terms so it stays readable. NOT used for node colour: every
    # SynGO-annotated gene gets its own (cyclically-repeating) colour below,
    # so "not in SynGO" (black) is never confused with "in SynGO but a less
    # common term".
    if term == "Unannotated in SynGO":
        return "Unannotated in SynGO"
    return term if term in top_terms else "Other SynGO term"

node_category = {n: bucket(node_to_pathway[n]) for n in nodes_set}

# ── colour palette (Cell-style: muted, colourblind-safe qualitative set) ──────
PALETTE = [
    "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#F0E442", "#8B008B", "#046A38", "#B0413E", "#4C72B0", "#8C6D31",
]
categories = top_terms + ["Other SynGO term", "Unannotated in SynGO"]

# every distinct SynGO term (not just the top-N shown in the legend) gets its
# own colour, cycling through the palette -- black is reserved exclusively for
# genes with NO SynGO annotation at all ("Unannotated in SynGO")
all_terms = [t for t in term_counts.index if t != "Unannotated in SynGO"]
term_colors = {term: PALETTE[i % len(PALETTE)] for i, term in enumerate(all_terms)}
term_colors["Unannotated in SynGO"] = "#000000"

cat_colors = {cat: term_colors[cat] for cat in top_terms}
cat_colors["Other SynGO term"] = "#999999"  # legend swatch only; actual node colours vary, see above
cat_colors["Unannotated in SynGO"] = "#000000"

# ── build node table ────────────────────────────────────────────────────────────
nodes_df = pd.DataFrame([{
    "Gene": n,
    "Degree": degree.get(n, 0),
    "Has_Motif": has_motif[n],
    "N_Motifs": motif_count[n],
    "Top_Pathway_Term": node_to_pathway[n],
    "Pathway_Category": node_category[n],
    "SynGO_GO_ID": node_to_goid[n],
    "Color": term_colors[node_to_pathway[n]],
} for n in nodes_set]).sort_values("Degree", ascending=False).reset_index(drop=True)

print(f"\nWriting -> {STATS_XL}")
with pd.ExcelWriter(STATS_XL, engine="openpyxl") as writer:
    nodes_df.to_excel(writer, sheet_name="Network_Nodes", index=False)
    edges.to_excel(writer, sheet_name="Network_Edges", index=False)
    term_counts.rename("N_Genes").reset_index().rename(columns={"index": "Pathway_Term"}).to_excel(
        writer, sheet_name="Pathway_Assignment_Summary", index=False)
    for ws in writer.sheets.values():
        for col in ws.columns:
            ml = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(ml + 2, 60)
print(f"Saved {len(nodes_df)} nodes, {len(edges)} edges")

# ── build interactive HTML (vis-network) ───────────────────────────────────────
vis_nodes = []
for _, r in nodes_df.iterrows():
    go_ref = f" ({r['SynGO_GO_ID']})" if r["SynGO_GO_ID"] else ""
    motif_note = "Yes" if r["Has_Motif"] else "No (interactor only)"
    tooltip = (f"{r['Gene']}\nDegree: {r['Degree']}\nCarries own motif: {motif_note}"
               f" ({int(r['N_Motifs'])} motif{'s' if r['N_Motifs'] != 1 else ''})\n"
               f"SynGO term: {r['Top_Pathway_Term']}{go_ref}")
    if r["Has_Motif"]:
        color = {"background": r["Color"], "border": r["Color"]}
        border_width = 0
    else:
        # interactor-only genes with no motif of their own -> hollow circle
        color = {"background": "#ffffff", "border": r["Color"]}
        border_width = 2.5
    vis_nodes.append({
        # circle size (vis-network "value" + scaling) encodes # motifs carried,
        # not network degree -- interactor-only (hollow) nodes carry 0 motifs
        # and render at the minimum size.
        # NB: deliberately NOT using vis-network's reserved "group" field here --
        # setting it silently made vis-network auto-generate its OWN colour for
        # each group name and override our explicit per-node "color" (that's
        # why e.g. NCAM1 rendered as an unrelated green instead of its assigned
        # olive tan). "pathwayGroup" is a plain custom property vis-network
        # doesn't touch, kept only for reference/debugging.
        "id": r["Gene"], "label": r["Gene"],
        "value": int(r["N_Motifs"]), "color": color, "borderWidth": border_width,
        "title": tooltip, "pathwayGroup": r["Pathway_Category"],
    })

vis_edges = []
for _, r in edges.iterrows():
    evidence = r["Supporting_Motifs"]
    if len(evidence) > 500:
        evidence = evidence[:500] + " ..."
    tooltip = f"{r['Gene1']} <-> {r['Gene2']}\nMotif links: {r['N_Motif_Links']}\n{evidence}"
    vis_edges.append({
        "from": r["Gene1"], "to": r["Gene2"],
        "value": int(r["N_Motif_Links"]), "title": tooltip,
    })

legend_rows = "".join(
    f'<div class="lg-row"><span class="lg-sw" style="background:{cat_colors[c]}"></span>'
    f'<span class="lg-lbl">{c}</span><span class="lg-n">{term_counts.get(c, 0)}</span></div>'
    for c in categories
)

with open(VIS_JS) as fh:
    vis_js = fh.read()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Crosslink-Validated PPI Network</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;background:#fff;font-family:Arial,sans-serif;color:#222;overflow:hidden}
#layout{display:flex;width:100vw;height:100vh}
#net{flex:1;min-width:0;position:relative;background:#fff}
#sidebar{width:320px;min-width:320px;display:flex;flex-direction:column;
         border-left:1px solid #e0e0e0;background:#fff;overflow-y:auto}
#hdr{padding:12px 16px;border-bottom:1px solid #eee;background:#fafafa}
#hdr h1{font-size:14px;font-weight:700;color:#111;margin-bottom:4px}
#hdr p{font-size:10.5px;color:#888;line-height:1.5}
#ctrl{padding:10px 16px;border-bottom:1px solid #eee}
.sh{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#777;margin:0 0 7px}
.rw{display:flex;align-items:center;gap:6px;font-size:11px;margin-bottom:8px}
input[type=text]{flex:1;font-size:12px;padding:5px 7px;border:1px solid #ccc;border-radius:4px}
.btn{padding:5px 10px;border:1.5px solid #ccc;border-radius:4px;font-size:11px;
     cursor:pointer;background:#fff;color:#333}
.btn:hover{background:#f0f0f0}
#legend{padding:10px 16px;border-bottom:1px solid #eee}
.lg-row{display:flex;align-items:center;gap:7px;font-size:10.5px;margin-bottom:5px}
.lg-sw{display:inline-block;width:11px;height:11px;border-radius:3px;flex-shrink:0}
.lg-lbl{flex:1;color:#333}
.lg-n{color:#999;font-size:9.5px}
#info{padding:12px 16px;font-size:11.5px;line-height:1.7;white-space:pre-wrap;color:#333}
.vis-tooltip{white-space:pre-line !important;max-width:320px;font-size:11px !important;line-height:1.5 !important}
#info b{color:#111}
.note{font-size:9.5px;color:#999;padding:10px 16px;line-height:1.5;border-top:1px solid #eee;margin-top:auto}
</style>
</head>
<body>
<div id="layout">
  <div id="net"></div>
  <div id="sidebar">
    <div id="hdr">
      <h1>Crosslink-Validated PPI Network</h1>
      <p>N_NODES_HERE genes &middot; N_EDGES_HERE interactions<br>
         Nodes = genes with exact-crosslink-matched motifs &amp; their interactors.
         Edges = inter-protein crosslink evidence. Colour = SynGO GO term
         (Cellular Component / Biological Process); black "Unannotated in SynGO" = not
         annotated in SynGO. Hollow circles = genes that only exist in the
         crosslinked proteome as an interactor and carry no motif of their own.<br>
         Drag=pan &middot; Scroll=zoom &middot; Click node/edge=details &middot; Drag node=reposition</p>
    </div>
    <div id="ctrl">
      <div class="sh">Search</div>
      <div class="rw">
        <input type="text" id="search" placeholder="Gene symbol..." onkeyup="doSearch()">
      </div>
      <div class="rw">
        <button class="btn" onclick="network.fit()">Reset view</button>
        <button class="btn" onclick="togglePhysics()" id="physBtn">Physics: On</button>
      </div>
      <div class="sh">Label font size</div>
      <div class="rw">
        <input type="range" id="fontSlider" min="6" max="24" value="11" step="1"
               oninput="setFontSize(this.value)" style="flex:1">
        <span id="fontVal" style="width:22px;text-align:right">11</span>
      </div>
      <div class="sh">Export</div>
      <div class="rw">
        <select id="exportScale" style="flex:1;font-size:11px;padding:4px 6px;border:1px solid #ccc;border-radius:4px">
          <option value="2">2x</option>
          <option value="4" selected>4x (print, ~300 DPI)</option>
          <option value="8">8x (poster)</option>
        </select>
        <button class="btn" onclick="exportHiRes()">Save PNG</button>
      </div>
    </div>
    <div id="legend">
      <div class="sh">SynGO term (node colour)</div>
      LEGEND_ROWS_HERE
    </div>
    <div id="info">Click a node or edge for details.</div>
    <div class="note">
      Each gene is coloured by its most specific (smallest gene-set) SynGO GO
      term; only genes with NO SynGO annotation at all are black ("Unannotated in SynGO").
      The legend lists the top TOP_N_PATHWAYS_HERE terms by gene count --
      "Other SynGO term" covers every remaining term, each still drawn in its
      own (non-black) colour on the network, just not individually listed here.
      Node size = number of distinct motifs the gene carries (0 for hollow,
      interactor-only nodes).
    </div>
  </div>
</div>

<script>
VIS_JS_HERE
</script>
<script>
const NODES = new vis.DataSet(NODES_JSON_HERE);
const EDGES = new vis.DataSet(EDGES_JSON_HERE);

const container = document.getElementById('net');
const data = {nodes: NODES, edges: EDGES};
const options = {
    nodes: {
        shape: 'dot', scaling: {min: 6, max: 28},
        font: {size: 11, color: '#222'},
        borderWidth: 0,
    },
    edges: {
        color: {color: '#7a7a7a', highlight: '#333', opacity: 0.8},
        smooth: {type: 'continuous'},
        scaling: {min: 1, max: 6},
    },
    physics: {
        solver: 'barnesHut',
        barnesHut: {gravitationalConstant: -12000, springLength: 110, springConstant: 0.03},
        stabilization: {iterations: 200},
    },
    interaction: {hover: true, tooltipDelay: 100},
};
const network = new vis.Network(container, data, options);

let physicsOn = true;
function togglePhysics() {
    physicsOn = !physicsOn;
    network.setOptions({physics: {enabled: physicsOn}});
    document.getElementById('physBtn').textContent = 'Physics: ' + (physicsOn ? 'On' : 'Off');
}

let baseFontSize = 11;
function setFontSize(v) {
    baseFontSize = parseInt(v);
    document.getElementById('fontVal').textContent = baseFontSize;
    network.setOptions({nodes: {font: {size: baseFontSize}}});
    doSearch();
}

// Re-renders vis-network's own canvas into a much bigger pixel buffer (same
// pan/zoom, just higher pixel density) and saves that as a PNG -- publication
// / print-resolution export, not just a screen-resolution screenshot.
// NB: vis-network's redraw() keeps its canvas's drawing-buffer size in sync
// with its container element, so directly resizing canvas.width/height gets
// silently reset on the next redraw() call -- the container itself has to be
// enlarged first (network.setSize matches the buffer to it), captured, then
// both put back.
function exportHiRes() {
    const scale = parseInt(document.getElementById('exportScale').value);
    const container = document.getElementById('net');
    const canvas = network.canvas.frame.canvas;
    const w = container.clientWidth, h = container.clientHeight;

    container.style.width = (w * scale) + 'px';
    container.style.height = (h * scale) + 'px';
    network.setSize((w * scale) + 'px', (h * scale) + 'px');
    network.redraw();

    const dataURL = canvas.toDataURL('image/png');

    container.style.width = '';
    container.style.height = '';
    network.setSize('100%', '100%');
    network.redraw();

    const a = document.createElement('a');
    a.href = dataURL;
    a.download = 'crosslink_ppi_network_' + scale + 'x.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

network.on('click', function (params) {
    const info = document.getElementById('info');
    if (params.nodes.length > 0) {
        const n = NODES.get(params.nodes[0]);
        const connected = network.getConnectedNodes(n.id);
        info.innerHTML = '<b>' + n.label + '</b>\n' + n.title.split('\n').slice(1).join('\n') +
            '\nConnected genes (' + connected.length + '): ' + connected.join(', ');
    } else if (params.edges.length > 0) {
        const e = EDGES.get(params.edges[0]);
        info.innerHTML = e.title;
    }
});

function doSearch() {
    const q = document.getElementById('search').value.trim().toUpperCase();
    if (!q) { NODES.forEach(n => NODES.update({id: n.id, font: {size: baseFontSize, color: '#222'}})); return; }
    NODES.forEach(function (n) {
        const hit = n.id.toUpperCase().includes(q);
        NODES.update({id: n.id, font: {size: hit ? baseFontSize + 5 : baseFontSize, color: hit ? '#D55E00' : '#ccc'}});
    });
    const hitIds = NODES.get({filter: n => n.id.toUpperCase().includes(q)}).map(n => n.id);
    if (hitIds.length) network.selectNodes(hitIds);
}
</script>
</body>
</html>
"""

html = HTML
html = html.replace("N_NODES_HERE", str(len(nodes_df)))
html = html.replace("N_EDGES_HERE", str(len(edges)))
html = html.replace("LEGEND_ROWS_HERE", legend_rows)
html = html.replace("TOP_N_PATHWAYS_HERE", str(TOP_N_PATHWAYS))
html = html.replace("VIS_JS_HERE", vis_js)
html = html.replace("NODES_JSON_HERE", json.dumps(vis_nodes))
html = html.replace("EDGES_JSON_HERE", json.dumps(vis_edges))

with open(HTML_OUT, "w") as fh:
    fh.write(html)
print(f"Interactive network saved -> {HTML_OUT} ({os.path.getsize(HTML_OUT)//1024} KB)")
