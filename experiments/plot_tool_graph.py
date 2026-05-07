"""
Visualize the MCP tool graph from tools.json.
Renders nodes (API endpoints) and edges (data chains) as a clean diagram.
"""
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_graph(tools_path: str, output_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError:
        print("pip install matplotlib")
        sys.exit(1)

    with open(tools_path) as f:
        tools = json.load(f)

    tool = tools[0]
    nodes = tool.get("graph_nodes", [])
    edges = tool.get("graph_edges", [])

    if not nodes:
        print("No graph nodes found.")
        return

    # Build node info
    node_info = {}
    for n in nodes:
        nid = n["node_id"]
        method = n["method"]
        params = []
        for r in n.get("extraction_recipe", []):
            p = r["param_path"].replace("$.", "")
            cls = r["classification"]
            params.append((p, cls))
        node_info[nid] = {
            "endpoint": n["endpoint_identity"],
            "method": method,
            "url": n["url_template"].split("/")[-1],
            "params": params,
        }

    # ── Layout ──
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-1.5, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    colors = {
        "node_bg": "#161b22",
        "node_border": "#30363d",
        "n0_header": "#1f6feb",
        "n1_header": "#8957e5",
        "n2_header": "#da3633",
        "text": "#e6edf3",
        "text_dim": "#8b949e",
        "chained": "#f78166",
        "user_intent": "#7ee787",
        "static": "#d2a8ff",
        "arrow": "#58a6ff",
        "success": "#3fb950",
    }

    header_colors = {"n0": colors["n0_header"], "n1": colors["n1_header"], "n2": colors["n2_header"]}

    # Positions — generous spacing, n2 shifted right and centered vertically
    positions = {
        "n0": (0.5, 9.0),
        "n1": (0.5, 3.5),
        "n2": (8.5, 6.25),
    }

    node_width = 4.2
    node_heights = {}

    def draw_node(nid, x, y):
        info = node_info[nid]
        params = info["params"]

        header_h = 0.7
        param_h = 0.35
        padding = 0.3
        total_h = header_h + padding + len(params) * param_h + padding
        node_heights[nid] = total_h

        # Shadow
        shadow = FancyBboxPatch(
            (x + 0.06, y - total_h - 0.06), node_width, total_h,
            boxstyle="round,pad=0.12",
            facecolor="#010409",
            edgecolor="none",
            alpha=0.5,
        )
        ax.add_patch(shadow)

        # Node background
        rect = FancyBboxPatch(
            (x, y - total_h), node_width, total_h,
            boxstyle="round,pad=0.12",
            facecolor=colors["node_bg"],
            edgecolor=colors["node_border"],
            linewidth=2,
        )
        ax.add_patch(rect)

        # Header bar
        hdr_color = header_colors.get(nid, colors["n0_header"])
        header_rect = FancyBboxPatch(
            (x + 0.08, y - header_h - 0.08), node_width - 0.16, header_h,
            boxstyle="round,pad=0.06",
            facecolor=hdr_color,
            edgecolor="none",
        )
        ax.add_patch(header_rect)

        # Node ID + endpoint
        ax.text(
            x + 0.3, y - header_h / 2 - 0.08,
            f"{nid}",
            fontsize=14, fontweight="bold", color="white",
            va="center", fontfamily="monospace",
        )
        ax.text(
            x + 0.8, y - header_h / 2 - 0.08,
            f"{info['method']} .../{info['url']}",
            fontsize=9.5, color="#e0e0e0",
            va="center", fontfamily="monospace",
        )

        # Parameters
        py = y - header_h - padding - 0.05
        for pname, cls in params:
            color = colors.get(cls, colors["text_dim"])
            marker = {"chained": "\u25b6", "user_intent": "\u25cb", "static": "\u25a0"}.get(cls, "\u00b7")
            ax.text(
                x + 0.3, py,
                f"{marker} {pname}",
                fontsize=8.5, color=color,
                va="center", fontfamily="monospace",
            )
            ax.text(
                x + node_width - 0.2, py,
                cls,
                fontsize=7, color=colors["text_dim"],
                va="center", ha="right", fontfamily="monospace",
                style="italic",
            )
            py -= param_h

    # Draw nodes
    for nid, (x, y) in positions.items():
        draw_node(nid, x, y)

    # Draw edges with labels offset to avoid overlap
    edge_configs = {
        "n0": {"rad": 0.2, "label_offset_x": 0.6, "label_offset_y": 0.7},
        "n1": {"rad": -0.2, "label_offset_x": 0.6, "label_offset_y": -0.7},
    }

    for edge in edges:
        fn = edge["from_node"]
        tn = edge["to_node"]
        fx, fy = positions[fn]
        tx, ty = positions[tn]
        cfg = edge_configs.get(fn, {"rad": 0, "label_offset_x": 0, "label_offset_y": 0})

        start_x = fx + node_width + 0.15
        start_y = fy - node_heights[fn] / 2
        end_x = tx - 0.15
        end_y = ty - node_heights[tn] / 2

        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            connectionstyle=f"arc3,rad={cfg['rad']}",
            arrowstyle="-|>",
            color=colors["arrow"],
            linewidth=2.5,
            mutation_scale=18,
        )
        ax.add_patch(arrow)

        # Edge label — offset so they don't collide
        mid_x = (start_x + end_x) / 2 + cfg["label_offset_x"]
        mid_y = (start_y + end_y) / 2 + cfg["label_offset_y"]

        from_path = edge["from_json_path"].replace("$.data.", "")
        to_path = edge["to_param_path"].replace("$.", "")
        label = f"{from_path}\n\u2192 {to_path}"
        ax.text(
            mid_x, mid_y,
            label,
            fontsize=7.5, color=colors["chained"],
            ha="center", va="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1a1e24", edgecolor=colors["chained"], alpha=0.95, linewidth=1.2),
        )

    # Title
    ax.text(
        7.0, 10.5,
        f"Tool Graph: {tool['name']}",
        fontsize=18, fontweight="bold", color=colors["text"],
        ha="center", fontfamily="monospace",
    )
    ax.text(
        7.0, 10.0,
        tool["short_description"],
        fontsize=11, color=colors["text_dim"],
        ha="center", fontfamily="monospace",
    )

    # Status badge
    status = tool.get("status", "unknown")
    badge_color = colors["success"] if status == "verified" else colors["chained"]
    ax.text(
        13.0, 10.5,
        f"\u2713 {status}",
        fontsize=11, fontweight="bold", color=badge_color,
        ha="center", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1e24", edgecolor=badge_color, linewidth=1.5),
    )

    # Legend
    legend_y = -0.8
    legend_items = [
        ("\u25cb  user_intent — derived from task text", colors["user_intent"]),
        ("\u25b6  chained — piped from upstream node response", colors["chained"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.text(
            3.0 + i * 5.5, legend_y,
            label,
            fontsize=9, color=color,
            fontfamily="monospace",
        )

    # Flow annotation
    ax.text(
        7.0, -1.2,
        'n0 (source lookup) + n1 (dest lookup)  \u2500\u2500\u2500  chain station codes  \u2500\u2500\u2500\u25b6  n2 (train search)',
        fontsize=9.5, color=colors["text_dim"],
        ha="center", fontfamily="monospace",
    )

    if output_path is None:
        output_path = str(Path(__file__).parent / "tool_graph.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Graph saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "confirmtkt_com"
    tools_path = Path(__file__).parent.parent / "morphnet" / "sites" / site / "tools.json"
    plot_graph(str(tools_path))
