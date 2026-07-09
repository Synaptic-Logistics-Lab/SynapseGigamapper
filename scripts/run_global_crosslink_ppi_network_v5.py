"""
Recolors the four-region global crosslink PPI network HTML
(global_crosslink_ppi_network_v4.html / run_global_crosslink_ppi_network_v4.py):

  - motif-matched only:   blue   (#2a78d6) -> magenta (#cc00cc)
  - SynGO only:           orange (#eb6834) -> green   (#2ca02c)
  - both:                 purple (#8B008B) -> green   (#2ca02c)
  - neither:              grey   (#8c8a7e) -> white   (#ffffff, effectively hidden on the white canvas)

Reuses the already-computed global_crosslink_ppi_network_v4.xlsx
(Global_Nodes +Region, Global_Edges +Region) -- no need to reprocess data,
region membership is unchanged, only the color mapping changes.

Output is a NEW file (global_crosslink_ppi_network_v5.html) -- v4 is left untouched.
"""

import json
import os
import pandas as pd

REPO_DIR  = "/home/au729231/SynapseGigamapper"
OUT_DIR   = f"{REPO_DIR}/notebook/ESMC_outputs"
STATS_XL  = f"{OUT_DIR}/global_crosslink_ppi_network_v4.xlsx"
HTML_DIR  = f"{OUT_DIR}/interactive_html"
HTML_OUT  = f"{HTML_DIR}/global_crosslink_ppi_network_v5.html"
VIS_JS    = f"{HTML_DIR}/vis-network.min.js"

REGION_COLORS = {
    "motif_only": "#cc00cc",   # magenta (was blue #2a78d6)
    "syngo_only": "#2ca02c",   # green (was orange #eb6834)
    "both":       "#2ca02c",   # green (was purple #8B008B)
    "neither":    "#ffffff",   # white / hidden (was grey #8c8a7e)
}
REGION_EDGE_COLORS = dict(REGION_COLORS)

REGION_LABELS = {
    "motif_only": "Motif-matched only",
    "syngo_only": "SynGO only",
    "both":       "Motif-matched + SynGO",
    "neither":    "Neither",
}

print("Loading precomputed global network tables (v4, region assignment unchanged)...")
nodes_df = pd.read_excel(STATS_XL, sheet_name="Global_Nodes")
edges_df = pd.read_excel(STATS_XL, sheet_name="Global_Edges")
print(f"  {len(nodes_df)} nodes, {len(edges_df)} edges")

node_counts = nodes_df["Region"].value_counts().to_dict()
edge_counts = edges_df["Region"].value_counts().to_dict()

# ── build interactive HTML (vis-network) ───────────────────────────────────
os.makedirs(HTML_DIR, exist_ok=True)

vis_nodes = []
for _, r in nodes_df.iterrows():
    region = r["Region"]
    tooltip = (f"{r['Gene']}\nRegion: {REGION_LABELS[region]}\nGlobal degree: {r['Global_Degree']}\n"
               f"In motif-matched network: {'Yes' if r['In_Motif_Network'] else 'No'}\n"
               f"In SynGO: {'Yes' if r['In_SynGO'] else 'No'}")
    if r["In_Motif_Network"]:
        tooltip += f"\nMotifs carried: {int(r['N_Motifs'])}\nSynGO term: {r['Pathway_Category']}"
    show_label = region in ("motif_only", "both")
    vis_nodes.append({
        "id": r["Gene"], "label": r["Gene"],
        "value": 2 + int(r["Global_Degree"]),
        "color": {"background": REGION_COLORS[region], "border": REGION_COLORS[region]},
        "font": {"size": 11 if show_label else 0, "color": "#222"},
        "title": tooltip,
        "region": region,
    })

vis_edges = []
for _, r in edges_df.iterrows():
    region = r["Region"]
    tooltip = (f"{r['Gene1']} <-> {r['Gene2']}\nRegion: {REGION_LABELS[region]}\n"
               f"Peptide-pair evidence: {r['N_Peptide_Links']}")
    vis_edges.append({
        "from": r["Gene1"], "to": r["Gene2"],
        "value": 3 if region in ("motif_only", "both") else (2 if region == "syngo_only" else 1),
        "color": {"color": REGION_EDGE_COLORS[region]},
        "title": tooltip,
        "region": region,
    })

with open(VIS_JS) as fh:
    vis_js = fh.read()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Global Crosslink PPI Network -- Four Toggleable Regions</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;background:#fff;font-family:Arial,sans-serif;color:#222;overflow:hidden}
#layout{display:flex;width:100vw;height:100vh}
#net{flex:1;min-width:0;position:relative;background:#fff}
#sidebar{width:330px;min-width:330px;display:flex;flex-direction:column;
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
#regions{padding:10px 16px;border-bottom:1px solid #eee}
.reg-row{display:flex;align-items:center;gap:8px;font-size:11.5px;margin-bottom:9px;cursor:pointer}
.reg-sw{display:inline-block;width:13px;height:13px;border-radius:3px;flex-shrink:0;border:1px solid #ccc}
.reg-lbl{flex:1;color:#222}
.reg-n{color:#999;font-size:9.5px}
#info{padding:12px 16px;font-size:11.5px;line-height:1.7;white-space:pre-wrap;color:#333}
.vis-tooltip{white-space:pre-line !important;max-width:320px;font-size:11px !important;line-height:1.5 !important}
.note{font-size:9.5px;color:#999;padding:10px 16px;line-height:1.5;border-top:1px solid #eee;margin-top:auto}
</style>
</head>
<body>
<div id="layout">
  <div id="net"></div>
  <div id="sidebar">
    <div id="hdr">
      <h1>Global Crosslink PPI Network</h1>
      <p>N_NODES_HERE genes &middot; N_EDGES_HERE inter-protein interactions, from the FULL
         crosslink-MS proteomics dataset. Every node/edge belongs to exactly one of four
         regions (checkboxes below) -- toggle any combination to isolate the part of the
         network you want to examine.<br>
         Drag=pan &middot; Scroll=zoom &middot; Click node/edge=details &middot; Drag node=reposition</p>
    </div>
    <div id="regions">
      <div class="sh">Regions (click to show/hide)</div>
      <div class="reg-row" onclick="toggleRegion('motif_only')">
        <input type="checkbox" id="chk-motif_only" checked onclick="event.stopPropagation();toggleRegion('motif_only')">
        <span class="reg-sw" style="background:MOTIF_ONLY_COLOR_HERE"></span>
        <span class="reg-lbl">Motif-matched only</span>
        <span class="reg-n">N_MOTIF_ONLY_NODES_HERE nodes / N_MOTIF_ONLY_EDGES_HERE edges</span>
      </div>
      <div class="reg-row" onclick="toggleRegion('syngo_only')">
        <input type="checkbox" id="chk-syngo_only" checked onclick="event.stopPropagation();toggleRegion('syngo_only')">
        <span class="reg-sw" style="background:SYNGO_ONLY_COLOR_HERE"></span>
        <span class="reg-lbl">SynGO only</span>
        <span class="reg-n">N_SYNGO_ONLY_NODES_HERE nodes / N_SYNGO_ONLY_EDGES_HERE edges</span>
      </div>
      <div class="reg-row" onclick="toggleRegion('both')">
        <input type="checkbox" id="chk-both" checked onclick="event.stopPropagation();toggleRegion('both')">
        <span class="reg-sw" style="background:BOTH_COLOR_HERE"></span>
        <span class="reg-lbl">Motif-matched + SynGO</span>
        <span class="reg-n">N_BOTH_NODES_HERE nodes / N_BOTH_EDGES_HERE edges</span>
      </div>
      <div class="reg-row" onclick="toggleRegion('neither')">
        <input type="checkbox" id="chk-neither" checked onclick="event.stopPropagation();toggleRegion('neither')">
        <span class="reg-sw" style="background:NEITHER_COLOR_HERE"></span>
        <span class="reg-lbl">Neither</span>
        <span class="reg-n">N_NEITHER_NODES_HERE nodes / N_NEITHER_EDGES_HERE edges</span>
      </div>
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
      <div class="sh">Label font size (motif-matched regions)</div>
      <div class="rw">
        <input type="range" id="fontSlider" min="0" max="24" value="11" step="1"
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
    <div id="info">Click a node or edge for details.</div>
    <div class="note">
      Node size = global degree. Region assignment: SynGO membership from predictions/syngo.xlsx
      (HGNC symbol + synonyms); an edge is "SynGO" if BOTH its genes are SynGO. Nodes and edges
      are each an internally consistent 4-way partition (every node/edge in exactly one region).
      "Neither" is rendered white (hidden against the canvas) in this color scheme.
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
        shape: 'dot', scaling: {min: 3, max: 26},
        font: {size: 11, color: '#222'},
        borderWidth: 0,
    },
    edges: {
        smooth: {type: 'continuous'},
        scaling: {min: 1, max: 4},
    },
    physics: {
        solver: 'barnesHut',
        barnesHut: {gravitationalConstant: -9000, springLength: 90, springConstant: 0.02},
        stabilization: {iterations: 150},
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

const regionVisible = {motif_only: true, syngo_only: true, both: true, neither: true};
function toggleRegion(region) {
    regionVisible[region] = !regionVisible[region];
    document.getElementById('chk-' + region).checked = regionVisible[region];
    const nodeUpdates = [];
    NODES.forEach(function (n) {
        if (n.region === region) nodeUpdates.push({id: n.id, hidden: !regionVisible[region]});
    });
    NODES.update(nodeUpdates);
    const edgeUpdates = [];
    EDGES.forEach(function (e) {
        if (e.region === region) edgeUpdates.push({id: e.id, hidden: !regionVisible[region]});
    });
    EDGES.update(edgeUpdates);
}

let baseFontSize = 11;
function setFontSize(v) {
    baseFontSize = parseInt(v);
    document.getElementById('fontVal').textContent = baseFontSize;
    const updates = [];
    NODES.forEach(function (n) {
        if (n.region === 'motif_only' || n.region === 'both') updates.push({id: n.id, font: {size: baseFontSize, color: '#222'}});
    });
    NODES.update(updates);
    doSearch();
}

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
    a.download = 'global_crosslink_ppi_network_v5_' + scale + 'x.png';
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
            '\nConnected genes (' + connected.length + '): ' + connected.slice(0, 40).join(', ') +
            (connected.length > 40 ? ' ... (' + (connected.length - 40) + ' more)' : '');
    } else if (params.edges.length > 0) {
        const e = EDGES.get(params.edges[0]);
        info.innerHTML = e.title;
    }
});

function doSearch() {
    const q = document.getElementById('search').value.trim().toUpperCase();
    if (!q) {
        NODES.forEach(n => NODES.update({id: n.id, font: {size: (n.region === 'motif_only' || n.region === 'both') ? baseFontSize : 0, color: '#222'}}));
        return;
    }
    NODES.forEach(function (n) {
        const hit = n.id.toUpperCase().includes(q);
        const defSize = (n.region === 'motif_only' || n.region === 'both') ? baseFontSize : 0;
        NODES.update({id: n.id, font: {size: hit ? baseFontSize + 6 : defSize, color: hit ? '#D55E00' : '#222'}});
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
html = html.replace("N_EDGES_HERE", str(len(edges_df)))
html = html.replace("MOTIF_ONLY_COLOR_HERE", REGION_COLORS["motif_only"])
html = html.replace("SYNGO_ONLY_COLOR_HERE", REGION_COLORS["syngo_only"])
html = html.replace("BOTH_COLOR_HERE", REGION_COLORS["both"])
html = html.replace("NEITHER_COLOR_HERE", REGION_COLORS["neither"])
html = html.replace("N_MOTIF_ONLY_NODES_HERE", str(node_counts.get("motif_only", 0)))
html = html.replace("N_MOTIF_ONLY_EDGES_HERE", str(edge_counts.get("motif_only", 0)))
html = html.replace("N_SYNGO_ONLY_NODES_HERE", str(node_counts.get("syngo_only", 0)))
html = html.replace("N_SYNGO_ONLY_EDGES_HERE", str(edge_counts.get("syngo_only", 0)))
html = html.replace("N_BOTH_NODES_HERE", str(node_counts.get("both", 0)))
html = html.replace("N_BOTH_EDGES_HERE", str(edge_counts.get("both", 0)))
html = html.replace("N_NEITHER_NODES_HERE", str(node_counts.get("neither", 0)))
html = html.replace("N_NEITHER_EDGES_HERE", str(edge_counts.get("neither", 0)))
html = html.replace("VIS_JS_HERE", vis_js)
html = html.replace("NODES_JSON_HERE", json.dumps(vis_nodes))
html = html.replace("EDGES_JSON_HERE", json.dumps(vis_edges))

with open(HTML_OUT, "w") as fh:
    fh.write(html)
print(f"Interactive network saved -> {HTML_OUT} ({os.path.getsize(HTML_OUT)//1024} KB)")
