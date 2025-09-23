# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from networkx.drawing import circular_layout

# -----------------------------
# Inputs from previous steps
# -----------------------------
rel_path = Path("ADR_Relations_From_Itolizumab_Case.csv")  # 5-col relations
classes_path = Path("ADR_Ontology_Class_Table_PrimeKG.csv")  # ontology classes table (labels/names)

# Load data
rels_df = pd.read_csv(rel_path)
classes_df = pd.read_csv(classes_path)

# -----------------------------
# Step 1: Output (FromEntity, Relation, ToEntity) triples
# -----------------------------
triples_df = rels_df[["FromEntity", "Relation", "ToEntity"]].copy()
triples_out = Path("ADR_Triples_For_KG.csv")
triples_df.to_csv(triples_out, index=False)

# -----------------------------
# Step 2: Build mapping (Instance → Ontology Class)
# -----------------------------
# Use FromEntityClass and ToEntityClass columns; map to human-readable class names if available.
label_to_name = dict(zip(classes_df["ClassLabel"], classes_df["ClassName"]))

mapping_rows = []
for _, row in rels_df.iterrows():
    fe, fcl = row["FromEntity"], row["FromEntityClass"]
    te, tcl = row["ToEntity"], row["ToEntityClass"]
    mapping_rows.append([fe, fcl, label_to_name.get(fcl, fcl)])
    mapping_rows.append([te, tcl, label_to_name.get(tcl, tcl)])

map_df = pd.DataFrame(mapping_rows, columns=["Entity", "ClassLabel", "ClassName"]).drop_duplicates()
mapping_out = Path("ADR_Instance_Class_Mapping.csv")
map_df.to_csv(mapping_out, index=False)

# -----------------------------
# Step 3: Build a two-layer knowledge graph and draw with NetworkX
#  - Top: Ontology view (unique classes present)
#  - Bottom: Instance view (unique entities present)
#  - Edges:
#     a) Instance edges (FromEntity --Relation--> ToEntity)
#     b) Ontology edges between classes with same relation labels (ClassA --relation--> ClassB)
#     c) Mapping edges (Entity --> Class), dashed
# -----------------------------
G = nx.MultiDiGraph()

# Add ontology nodes (classes used)
classes_used = set(map_df["ClassLabel"])
for cl in classes_used:
    G.add_node(("C", cl), layer="ontology", label=label_to_name.get(cl, cl))

# Add instance nodes
for ent, cl in zip(map_df["Entity"], map_df["ClassLabel"]):
    G.add_node(("E", ent), layer="instance", label=ent, class_label=cl)

# Add instance relation edges
for _, r in rels_df.iterrows():
    G.add_edge(("E", r["FromEntity"]), ("E", r["ToEntity"]), key=f"inst::{r['Relation']}", relation=r["Relation"], kind="instance")

# Add ontology relation edges (class-to-class) derived from instance relations
for _, r in rels_df.drop_duplicates(subset=["FromEntityClass", "Relation", "ToEntityClass"]).iterrows():
    G.add_edge(("C", r["FromEntityClass"]), ("C", r["ToEntityClass"]), key=f"ont::{r['Relation']}", relation=r["Relation"], kind="ontology")

# Add mapping edges (Entity -> Class)
for _, r in map_df.iterrows():
    G.add_edge(("E", r["Entity"]), ("C", r["ClassLabel"]), key="map::instance_of", relation="instance_of", kind="mapping")

# -----------------------------
# Layout: two horizontal bands (ontology on top, instance below)
# -----------------------------
# Compute positions separately and then merge with fixed y-levels
ont_nodes = [n for n in G.nodes if n[0] == "C"]
inst_nodes = [n for n in G.nodes if n[0] == "E"]

# Use spring layouts within each band for readability
# (seed fixed for reproducibility)
pos_ont = nx.circular_layout(G.subgraph(ont_nodes))
pos_inst = nx.spring_layout(G.subgraph(inst_nodes), seed=17, k=1.0)

# Normalize x to [0,1] for each band
def normalize_x(pos_dict):
    xs = [p[0] for p in pos_dict.values()]
    if not xs:
        return pos_dict
    mn, mx = min(xs), max(xs)
    span = (mx - mn) if mx > mn else 1.0
    return {n: ((p[0] - mn) / span, p[1]) for n, p in pos_dict.items()}

pos_ont = normalize_x(pos_ont)
pos_inst = normalize_x(pos_inst)

# Assign y-levels
y_ont, y_inst = 0.55, 0.3
pos = {}
for n, (x, y) in pos_ont.items():
    pos[n] = (x, y_ont + 0.1 * y)
for n, (x, y) in pos_inst.items():
    pos[n] = (x, y_inst + 0.1 * y)

# -----------------------------
# Drawing
# -----------------------------
fig = plt.figure(figsize=(36, 48), dpi=600)  # high DPI paper-ready
ax = plt.gca()
ax.set_axis_off()

# Node styling
ont_node_opts = dict(node_shape="o", node_size=5000, linewidths=0.8, edgecolors="#1f2937")
inst_node_opts = dict(node_shape="o", node_size=5000, linewidths=0.8, edgecolors="#1f2937")

# Colors
ont_face = "#E9F2FF"   # light blue
inst_face = "#FFF5D7"  # light yellow

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=ont_nodes, node_color=ont_face, **ont_node_opts)
nx.draw_networkx_nodes(G, pos, nodelist=inst_nodes, node_color=inst_face, **inst_node_opts)

# Labels
labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=24)

# Edge subsets
edges_instance = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("kind") == "instance"]
edges_ontology = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("kind") == "ontology"]
edges_mapping = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("kind") == "mapping"]

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges_ontology],
                       arrows=True, arrowstyle="->", width=1.4, alpha=0.9, edge_color="#2563eb")  # blue

nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges_instance],
                       arrows=True, arrowstyle="->", width=1.2, alpha=0.85, edge_color="#16a34a")  # green

nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges_mapping],
                       arrows=True, arrowstyle="->", width=1.0, alpha=0.5, style="dashed", edge_color="#6b7280")  # grey dashed

# Edge labels (relation)
edge_label_pos = 0.5
edge_labels = {(u, v): G.edges[u, v, k]["relation"] for u, v, k in edges_ontology + edges_instance}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=20, label_pos=edge_label_pos, rotate=False)

# Horizontal separator line
ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, color="#9ca3af", linewidth=1.0, linestyle="--")
ax.text(0.01, 0.93, "Ontology View", transform=ax.transAxes, fontsize=36, weight="bold", color="#1f2937")
ax.text(0.01, 0.48, "Instance View", transform=ax.transAxes, fontsize=36, weight="bold", color="#1f2937")


png_path = Path("KG.png")
svg_path = Path("KG.svg")
plt.tight_layout()
plt.savefig(png_path, dpi=1200)
plt.savefig(svg_path)
plt.close(fig)



