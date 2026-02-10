"""
Build a directed graph from rules.json using NetworkX and visualise it
with pyvis, exporting to an interactive HTML file.

Root nodes (no incoming edges) are coloured GREEN.
Final nodes (is_final=true) are coloured RED.
The top-3 longest paths are printed and exported as a single combined
sub-graph HTML file that also includes every node directly connected to
the path nodes.

Usage:
    pip install networkx pyvis
    python 02_build_graph.py

Output:
    rules_graph.html                   – full interactive graph
    rules_subgraph_top3_combined.html  – combined sub-graph of top-3 paths + neighbours
"""

import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network

# ── Colour constants ──────────────────────────────────────────────────────────
ROOT_COLOUR = "#4CAF50"     # green  – root nodes (in-degree == 0)
FINAL_COLOUR = "#E53935"    # red    – is_final nodes
ENTITY_COLOURS = {
    "school": "#4FC3F7",    # light blue
    "student": "#81C784",   # light green
    "district": "#FFB74D",  # orange
    "teacher": "#CE93D8",   # purple
    "state": "#E57373",     # salmon
}
DEFAULT_COLOUR = "#B0BEC5"  # grey fallback
PATH_HIGHLIGHT = "#FFD600"  # yellow – edges on the highlighted path


def load_rules(path: str = "rules.json") -> list[dict]:
    """Load the rules JSON file and return a list of rule dicts."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_networkx_graph(rules: list[dict]) -> nx.DiGraph:
    """Create a NetworkX directed graph from the rules list."""
    G = nx.DiGraph()

    for rule in rules:
        rid = rule["rule_id"]
        entity = rule.get("entity_applied", "unknown")
        is_final = rule.get("is_final", False)

        G.add_node(
            rid,
            rule_name=rule["rule_name"],
            entity=entity,
            is_final=is_final,
            action=rule.get("action", "N/A"),
            outcome=rule.get("outcome", "N/A"),
        )

        for route in rule.get("outgoing_routes", []):
            target = route.get("next_rule")
            if target is not None:
                G.add_edge(
                    rid,
                    target,
                    condition=route.get("condition", ""),
                )

    return G


# ── Node colour logic ─────────────────────────────────────────────────────────

def node_colour(G: nx.DiGraph, node_id: str) -> str:
    """Return the display colour for a node.

    Priority: root (green) > final (red) > entity colour.
    """
    attrs = G.nodes[node_id]
    if G.in_degree(node_id) == 0:
        return ROOT_COLOUR
    if attrs.get("is_final", False):
        return FINAL_COLOUR
    return ENTITY_COLOURS.get(attrs.get("entity", ""), DEFAULT_COLOUR)


def node_shape(G: nx.DiGraph, node_id: str) -> str:
    attrs = G.nodes[node_id]
    if G.in_degree(node_id) == 0:
        return "diamond"
    if attrs.get("is_final", False):
        return "box"
    return "ellipse"


def node_label(G: nx.DiGraph, node_id: str) -> str:
    attrs = G.nodes[node_id]
    return f"{node_id}\n{attrs.get('rule_name', '')}"


def node_title(G: nx.DiGraph, node_id: str) -> str:
    attrs = G.nodes[node_id]
    role = "ROOT" if G.in_degree(node_id) == 0 else ("FINAL" if attrs.get("is_final") else "")
    return (
        f"<b>{node_id} – {attrs.get('rule_name', '')}</b><br>"
        f"<b>Entity:</b> {attrs.get('entity', 'N/A')}<br>"
        f"<b>Action:</b> {attrs.get('action', 'N/A')}<br>"
        f"<b>Outcome:</b> {attrs.get('outcome', 'N/A')}<br>"
        f"<b>Type:</b> {role}<br>"
    )


# ── Longest-path discovery ────────────────────────────────────────────────────

def find_all_paths_from_roots(G: nx.DiGraph) -> list[list[str]]:
    """Return every simple path from every root to every leaf in the DAG."""
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]

    all_paths: list[list[str]] = []
    for root in roots:
        for leaf in leaves:
            for path in nx.all_simple_paths(G, root, leaf):
                all_paths.append(path)
    return all_paths


def top_k_longest_paths(G: nx.DiGraph, k: int = 3) -> list[list[str]]:
    """Return the *k* longest simple root-to-leaf paths (by node count).

    Ties are broken by lexicographic order of the path so the result is
    deterministic.
    """
    paths = find_all_paths_from_roots(G)
    paths.sort(key=lambda p: (-len(p), p))
    # deduplicate identical paths that may arise from multi-edges
    seen: set[tuple[str, ...]] = set()
    unique: list[list[str]] = []
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:k]


# ── pyvis export helpers ──────────────────────────────────────────────────────

PYVIS_OPTIONS = """
{
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed",
      "levelSeparation": 200,
      "nodeSpacing": 250
    }
  },
  "physics": {
    "hierarchicalRepulsion": { "nodeDistance": 250 }
  },
  "edges": {
    "arrows": { "to": { "enabled": true } },
    "font": { "size": 10, "align": "middle" },
    "smooth": { "type": "cubicBezier" }
  },
  "nodes": {
    "font": { "size": 12, "face": "arial" },
    "borderWidth": 2
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100
  }
}
"""


def _add_graph_to_net(
    net: Network,
    G: nx.DiGraph,
    highlight_edges: set[tuple[str, str]] | None = None,
) -> None:
    """Transfer nodes and edges from *G* into a pyvis *net*."""
    for nid in G.nodes:
        net.add_node(
            nid,
            label=node_label(G, nid),
            title=node_title(G, nid),
            color=node_colour(G, nid),
            shape=node_shape(G, nid),
            size=30 if G.in_degree(nid) == 0 or G.nodes[nid].get("is_final") else 20,
        )

    highlight_edges = highlight_edges or set()
    for src, dst, attrs in G.edges(data=True):
        is_hl = (src, dst) in highlight_edges
        net.add_edge(
            src,
            dst,
            label=attrs.get("condition", ""),
            title=attrs.get("condition", ""),
            color=PATH_HIGHLIGHT if is_hl else "#888888",
            width=3.5 if is_hl else 1.5,
        )


def export_full_graph(G: nx.DiGraph, output_path: str) -> None:
    """Export the entire graph to an interactive HTML file."""
    net = Network(
        height="100%", width="100%",
        directed=True, notebook=False, cdn_resources="remote",
    )
    net.set_options(PYVIS_OPTIONS)
    _add_graph_to_net(net, G)
    net.save_graph(output_path)
    print(f"Full graph  → {output_path}")


def export_combined_subgraph(
    G: nx.DiGraph,
    paths: list[list[str]],
    output_path: str,
) -> None:
    """Export a single sub-graph that combines all *paths* plus every node
    directly connected (predecessor or successor) to any path node.

    Edges that belong to one of the top-k paths are highlighted in yellow.
    """
    # 1. Collect all nodes on the paths
    path_nodes: set[str] = set()
    for p in paths:
        path_nodes.update(p)

    # 2. Expand with immediate neighbours (predecessors + successors)
    expanded_nodes: set[str] = set(path_nodes)
    for n in path_nodes:
        expanded_nodes.update(G.predecessors(n))
        expanded_nodes.update(G.successors(n))

    # 3. Build the induced sub-graph
    sub = G.subgraph(expanded_nodes).copy()

    # 4. Collect all consecutive edges that lie on *any* of the paths
    path_edges: set[tuple[str, str]] = set()
    for p in paths:
        for i in range(len(p) - 1):
            path_edges.add((p[i], p[i + 1]))

    net = Network(
        height="100%", width="100%",
        directed=True, notebook=False, cdn_resources="remote",
    )
    net.set_options(PYVIS_OPTIONS)
    _add_graph_to_net(net, sub, highlight_edges=path_edges)
    net.save_graph(output_path)
    print(
        f"Combined sub-graph ({len(paths)} paths, "
        f"{len(path_nodes)} path nodes, "
        f"{len(expanded_nodes)} total nodes incl. neighbours) → {output_path}"
    )


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(G: nx.DiGraph) -> None:
    roots = sorted(n for n in G.nodes if G.in_degree(n) == 0)
    leaves = sorted(n for n in G.nodes if G.out_degree(n) == 0)

    print("=" * 70)
    print(f"GRAPH SUMMARY")
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")
    print()
    print(f"  Root nodes (GREEN, in-degree=0): {len(roots)}")
    for r in roots:
        print(f"    {r} – {G.nodes[r].get('rule_name', '')}")
    print()
    print(f"  Final nodes (RED, is_final=true): "
          f"{sum(1 for n in G.nodes if G.nodes[n].get('is_final'))}")
    for n in sorted(G.nodes):
        if G.nodes[n].get("is_final"):
            print(f"    {n} – {G.nodes[n].get('rule_name', '')}")
    print()
    print(f"  Leaf nodes (out-degree=0): {len(leaves)}")
    for lf in leaves:
        print(f"    {lf} – {G.nodes[lf].get('rule_name', '')}")
    print("=" * 70)


def print_top_paths(paths: list[list[str]], G: nx.DiGraph) -> None:
    print()
    print("TOP-3 LONGEST ROOT→LEAF PATHS")
    print("-" * 70)
    for i, path in enumerate(paths, 1):
        names = [f"{n} ({G.nodes[n].get('rule_name', '')})" for n in path]
        print(f"\n  Path #{i}  –  length {len(path)} nodes")
        print(f"  {'  →  '.join(names)}")
    print("-" * 70)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = Path(__file__).parent
    rules_path = base / "rules.json"

    rules = load_rules(rules_path)
    G = build_networkx_graph(rules)

    print_summary(G)

    # ── Top-3 longest paths ───────────────────────────────────────────────
    top3 = top_k_longest_paths(G, k=3)
    print_top_paths(top3, G)

    # ── Export full graph ─────────────────────────────────────────────────
    export_full_graph(G, str(base / "rules_graph.html"))

    # ── Export combined sub-graph for the top-3 paths + neighbours ───────
    if top3:
        export_combined_subgraph(
            G, top3,
            str(base / "rules_subgraph_top3_combined.html"),
        )
