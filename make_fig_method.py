"""Figure 1: method pipeline (schematic, drawn in matplotlib).

A  per-query legal expert graph        B  conserved-flow dynamic          C  route & compose
D  train end to end (implicit diff.)   E  when it helps: coupling cond.   F  evaluation
Panel E's inset uses the measured density-sweep numbers (results/density_sweep_flow_vs_independent.txt).

Writes paper/figs/method.{pdf,png} (paper version, no title) and
paper/figs/method_slide.{pdf,png} (16:9 slide version with title).
Run: python3 make_fig_method.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from figstyle import PALETTE as P, save

W, H_BODY, H_TITLE = 170.0, 92.0, 13.0   # drawing units; 1 unit = 0.04 in

# ------------------------------------------------------------------ primitives
def card(ax, x, y, w, h, fc=P["wash"], ec=P["grid"], lw=0.8, r=1.6, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

def text(ax, x, y, s, size=6.5, color=P["forest_dark"], weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=10, **kw)

def arrow(ax, x0, y0, x1, y1, color=P["sage"], lw=1.4, ms=8, style="-|>", ls="-", z=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

def header(ax, x, y, w, letter, title):
    text(ax, x, y, letter, size=8.5, weight="bold", color=P["forest"])
    text(ax, x + 4.5, y, title, size=7.6, weight="bold")
    ax.plot([x, x + w], [y - 3.0, y - 3.0], color=P["sage_light"], lw=0.8, zorder=2)

def node(ax, x, y, r=1.9, fc=P["mint"], ec=P["forest"], lw=0.9, label=None, size=5.2, z=6):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z))
    if label:
        text(ax, x, y, label, size=size, ha="center", color=P["forest_dark"])

def pill(ax, x, y, w, s, fc, tc=P["forest_dark"], size=5.8, h=4.2):
    card(ax, x, y - h / 2, w, h, fc=fc, ec="none", r=h / 2, z=3)
    text(ax, x + w / 2, y, s, size=size, ha="center", color=tc)

# ------------------------------------------------------------------ the legal graph (shared by A and B)
LAYERS, EXPERTS = 3, 4
LEGAL = {  # legal transitions between consecutive layers: (layer, i) -> set of j at layer+1
    (0, 0): {1, 2}, (0, 1): {0, 3}, (0, 2): {1}, (0, 3): {2, 3},
    (1, 0): {0, 3}, (1, 1): {2}, (1, 2): {0, 1}, (1, 3): {1, 3},
}
PATH = [1, 0, 3]  # the path the flow settles on: E2 -> E1 -> E4 (layers 1..3)

def graph_coords(x0, y0, w, h):
    xs = np.linspace(x0, x0 + w, LAYERS + 2)
    ys = np.linspace(y0 + h, y0, EXPERTS)
    src, snk = (xs[0], y0 + h / 2), (xs[-1], y0 + h / 2)
    pos = {(l, e): (xs[l + 1], ys[e]) for l in range(LAYERS) for e in range(EXPERTS)}
    return src, snk, pos

def edges(src, snk, pos):
    out = [(src, pos[(0, e)], ("s", e)) for e in range(EXPERTS)]
    for (l, i), js in LEGAL.items():
        out += [(pos[(l, i)], pos[(l + 1, j)], (l, i, j)) for j in js]
    out += [(pos[(LAYERS - 1, e)], snk, ("t", e)) for e in range(EXPERTS)]
    return out

def on_path(key):
    if key[0] == "s": return key[1] == PATH[0]
    if key[0] == "t": return key[1] == PATH[-1]
    l, i, j = key
    return PATH[l] == i and PATH[l + 1] == j

def draw_graph(ax, x0, y0, w, h, mode, r=1.9, labels=True):
    """mode: 'legal' (A), 'flow0' (B, early), 'flowT' (B, converged)."""
    src, snk, pos = graph_coords(x0, y0, w, h)
    rng = np.random.default_rng(3)
    for a, b, key in edges(src, snk, pos):
        if mode == "legal":
            ax.plot(*zip(a, b), color=P["sage_light"], lw=0.8, zorder=3)
        elif mode == "flow0":
            ax.plot(*zip(a, b), color=P["green"], lw=0.6 + 1.6 * rng.random(), alpha=0.55, zorder=3,
                    solid_capstyle="round")
        else:
            if on_path(key):
                ax.plot(*zip(a, b), color=P["forest"], lw=3.0, zorder=4, solid_capstyle="round")
            else:
                ax.plot(*zip(a, b), color=P["sage_light"], lw=0.5, alpha=0.6, zorder=3)
    for (l, e), (x, y) in pos.items():
        hot = mode == "flowT" and PATH[l] == e
        node(ax, x, y, r=r, fc=P["forest"] if hot else P["mint"],
             ec=P["forest"] if mode != "legal" or True else P["sage"],
             label=(f"E{e+1}" if labels else None), size=5.0 if r > 1.6 else 4.2)
        if hot and labels:
            text(ax, x, y, f"E{e+1}", size=5.0, ha="center", color="white")
    for (x, y), lab in [(src, "s"), (snk, "t")]:
        node(ax, x, y, r=r * 0.9, fc=P["forest_dark"], ec=P["forest_dark"])
        text(ax, x, y, lab, size=5.2 if r > 1.6 else 4.4, ha="center", color="white", weight="bold")
    return src, snk, pos

# ------------------------------------------------------------------ figure
def build(with_title):
    Htot = H_BODY + (H_TITLE if with_title else 0)
    fig = plt.figure(figsize=(W * 0.04, Htot * 0.04))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Htot); ax.axis("off")
    ax.set_aspect("equal")

    if with_title:
        text(ax, W / 2, H_BODY + 8.2, "Confluence: routing Mixture-of-Experts with a conserved flow",
             size=11, weight="bold", ha="center")
        text(ax, W / 2, H_BODY + 3.2, "One legal path of experts per input, trained end to end, "
             "and a condition for when coupled routing beats an independent gate", size=6.8,
             ha="center", color=P["sage"])

    cols = [3, 60, 117]; cw = 50
    top, bot = 86, 41

    # ---------------- A: legal graph
    x0 = cols[0]; header(ax, x0, top, cw, "A", "Per-input legal expert graph")
    src, snk, pos = draw_graph(ax, x0 + 2, 56, 44, 20, "legal")
    # illegal transitions: shown as crossed dashed links (they are NOT edges of the graph)
    for (l, i, j) in [(0, 2, 3), (1, 1, 0)]:
        (xa, ya), (xb, yb) = pos[(l, i)], pos[(l + 1, j)]
        ax.plot([xa, xb], [ya, yb], color=P["red"], lw=0.8, ls=(0, (2, 1.5)), zorder=2, alpha=0.9)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        text(ax, mx, my, "×", size=8.5, ha="center", color=P["red"], weight="bold")
    for l in range(LAYERS):
        text(ax, pos[(l, 0)][0], 52.3, f"layer {l+1}", size=5.4, ha="center", color=P["sage"])
    text(ax, x0 + cw / 2, 47.0, "Edges are exactly the legal expert transitions.\n"
         "An illegal transition ($\\times$) has no edge, so it can never be used.",
         size=5.9, ha="center", color=P["sage"], linespacing=1.35)

    # ---------------- B: conserved-flow dynamic
    x0 = cols[1]; header(ax, x0, top, cw, "B", "Conserved-flow dynamic (Physarum)")
    card(ax, x0, 63.5, 21, 16.5, fc="white", ec=P["sage_light"])
    eqs = [("conserve", r"$\mathcal{L}(w)\,p=b$"), ("flux", r"$Q=w\odot B^{\top}p$"),
           ("reinforce", r"$D\leftarrow D+\Delta t(|Q|-D)$")]
    for k, (lab, eq) in enumerate(eqs):
        yy = 76.5 - k * 5.0
        text(ax, x0 + 1.5, yy, lab, size=5.0, color=P["sage"])
        text(ax, x0 + 1.5, yy - 2.1, eq, size=6.0)
    draw_graph(ax, x0 + 23.0, 68.5, 11.0, 11.5, "flow0", r=1.0, labels=False)
    arrow(ax, x0 + 35.6, 74.2, x0 + 38.4, 74.2, color=P["forest"], lw=1.1, ms=6)
    draw_graph(ax, x0 + 39.8, 68.5, 10.5, 11.5, "flowT", r=1.0, labels=False)
    text(ax, x0 + 29.2, 64.8, "early", size=5.2, ha="center", color=P["sage"])
    text(ax, x0 + 44.6, 64.8, "fixed point", size=5.2, ha="center", color=P["forest"], weight="bold")
    pill(ax, x0 + 0.5, 57.8, 22.5, "conserved at every node", P["mint"], size=5.3)
    pill(ax, x0 + 24.5, 57.8, 25.5, "one sparse path survives", P["mint"], size=5.3)
    text(ax, x0 + cw / 2, 47.0, "The encoder sets per-edge costs; the flow relaxes\n"
         "to a single legal path of experts (no learned gate).",
         size=5.9, ha="center", color=P["sage"], linespacing=1.35)

    # ---------------- C: route & compose vs top-k gate
    x0 = cols[2]; header(ax, x0, top, cw, "C", "Route and compose")
    def chain(y, picks, color, fc, bad=None, label=None):
        xs = [x0 + 3 + k * 11.5 for k in range(5)]
        names = ["x"] + [f"E{p+1}" for p in picks] + [r"$\hat{y}$"]
        for k, (xx, nm) in enumerate(zip(xs, names)):
            if k in (0, 4):
                node(ax, xx, y, r=2.0, fc=P["forest_dark"] if color == P["forest"] else P["gray"],
                     ec="none"); text(ax, xx, y, nm, size=5.6, ha="center", color="white", weight="bold")
            else:
                card(ax, xx - 3.4, y - 2.6, 6.8, 5.2, fc=fc, ec=color, lw=0.9, r=1.0, z=5)
                text(ax, xx, y, nm, size=5.8, ha="center", weight="bold",
                     color="white" if fc == P["forest"] else P["forest_dark"])
            if k < 4:
                c = P["red"] if bad == k else color
                arrow(ax, xx + (2.2 if k == 0 else 3.6), y, xs[k + 1] - (3.6 if k < 3 else 2.2), y,
                      color=c, lw=1.1, ms=6, ls="--" if bad == k else "-")
                if bad == k:
                    text(ax, (xx + xs[k + 1]) / 2, y + 2.6, "×", size=8.5, ha="center", color=P["red"],
                         weight="bold")
        if label:
            text(ax, x0 + 1, y + 6.0, label[0], size=6.0, weight="bold", color=label[1])
    chain(70.5, PATH, P["forest"], P["forest"], label=("Conserved flow (ours)", P["forest"]))
    text(ax, x0 + 1, 64.2, "one legal path, 1 expert/layer:  " r"$\hat{y}=f_{E4}\circ f_{E1}\circ f_{E2}(x)$",
         size=5.6, color=P["forest_dark"])
    chain(54.5, [1, 2, 3], P["sage"], "white", bad=2,
          label=("Top-$k$ gate (standard MoE)", P["sage"]))
    text(ax, x0 + cw / 2, 47.0, "The gate scores each layer independently and can\n"
         "chain experts in an order the task forbids.",
         size=5.9, ha="center", color=P["sage"], linespacing=1.35)

    # row connector
    arrow(ax, cols[0] + cw + 1.5, 68, cols[1] - 1.5, 68, color=P["sage"], lw=1.6, ms=9)
    arrow(ax, cols[1] + cw + 1.5, 68, cols[2] - 1.5, 68, color=P["sage"], lw=1.6, ms=9)
    ax.plot([cols[2] + cw, cols[2] + cw, cols[0] + 1.5, cols[0] + 1.5],
            [44.0, 42.6, 42.6, 41.4], color=P["sage_light"], lw=1.0, zorder=1)
    arrow(ax, cols[0] + 1.5, 41.6, cols[0] + 1.5, 40.2, color=P["sage_light"], lw=1.0, ms=6)

    # ---------------- D: implicit differentiation
    x0 = cols[0]; header(ax, x0, bot - 3, cw, "D", "Train end to end: implicit differentiation")
    y = bot - 3
    card(ax, x0, y - 21, 25, 15.5, fc="white", ec=P["sage_light"])
    rows = [("fixed point", r"$D^{\star}=F(D^{\star},\theta)$"),
            ("backward solve", r"$(I-\partial_D F)^{\top}u=g$"),
            ("gradient", r"$\nabla_\theta=(\partial_\theta F)^{\top}u$")]
    for k, (lab, eq) in enumerate(rows):
        yy = y - 8 - k * 4.6
        text(ax, x0 + 1.5, yy, lab, size=5.0, color=P["sage"])
        text(ax, x0 + 1.5, yy - 2.0, eq, size=6.0)
    for k, s in enumerate(["$O(E)$ memory, any depth", "matrix-free VJP + CG",
                           "checked vs. finite diff."]):
        pill(ax, x0 + 26.5, y - 9 - k * 5.3, 23.5, s, P["mint"], size=5.2)
    text(ax, x0 + cw / 2, 5.0, "Gradients flow through the solver without\n"
         "unrolling it, so training cost stays linear in edges.",
         size=5.9, ha="center", color=P["sage"], linespacing=1.35)

    # ---------------- E: coupling condition (measured inset)
    x0 = cols[1]; header(ax, x0, y, cw, "E", "When does it help? Coupling")
    card(ax, x0, y - 13.5, 24, 8.8, fc=P["mint"], ec="none")
    text(ax, x0 + 1.5, y - 7.3, "coupled steps", size=5.8, weight="bold", color=P["forest"])
    text(ax, x0 + 1.5, y - 11.2, "flow > independent gate", size=5.3)
    card(ax, x0 + 26, y - 13.5, 24, 8.8, fc="#f1f1ef", ec="none")
    text(ax, x0 + 27.5, y - 7.3, "marginalizable steps", size=5.8, weight="bold", color=P["gray"])
    text(ax, x0 + 27.5, y - 11.2, "tie (e.g. MetaQA 2-hop)", size=5.3)
    # inset: measured flow advantage vs density
    ia = fig.add_axes([ (x0 + 7) / W, 13.2 / ax.get_ylim()[1], 40 / W, 10.5 / ax.get_ylim()[1] ])
    dens = [0.35, 0.50, 0.65, 0.80, 1.00]
    adv = [0.159, 0.071, 0.060, 0.033, -0.019]
    ia.axhline(0, color=P["gray"], lw=0.7)
    ia.fill_between(dens, adv, 0, where=[a > 0 for a in adv], color=P["forest"], alpha=0.15, lw=0)
    ia.plot(dens, adv, "o-", color=P["forest"], lw=1.3, ms=2.8)
    ia.invert_xaxis(); ia.set_xticks([0.4, 0.6, 0.8, 1.0]); ia.set_yticks([0, 0.1])
    ia.tick_params(labelsize=4.8, length=1.5, pad=1)
    ia.set_xlabel("")
    text(ax, x0 + 49.5, 9.3, "density $\\rightarrow$ more coupled", size=4.8, ha="right", color=P["sage"])
    ia.set_ylabel("flow adv.", fontsize=5.0, labelpad=0.5)
    ia.grid(False); ia.patch.set_alpha(0)
    text(ax, x0 + cw / 2, 4.4, "Measured: the advantage grows with coupling\n"
         "and crosses zero when steps are independent.",
         size=5.9, ha="center", color=P["sage"], linespacing=1.35)

    # ---------------- F: evaluation
    x0 = cols[2]; header(ax, x0, y, cw, "F", "Evaluate")
    tests = [("constraint-density sweep", "synthetic, 3 seeds"),
             ("MoE FFN layer", "accuracy vs. compute"),
             ("MoE transformer", "$E$=6 experts, $K$=3 layers"),
             ("MetaQA 2-hop KGQA", "43k entities, real data")]
    for k, (a, b) in enumerate(tests):
        yy = y - 7.5 - k * 4.4
        ax.add_patch(Circle((x0 + 2.2, yy), 0.9, fc=P["forest"], ec="none", zorder=5))
        text(ax, x0 + 4.5, yy, a, size=5.8, weight="bold")
        text(ax, x0 + 30, yy, b, size=5.2, color=P["sage"])
    pill(ax, x0 + 0.5, 12.8, 13.5, "accuracy", P["mint"], size=5.3)
    pill(ax, x0 + 15.0, 12.8, 17.5, "experts / layer", "#eef3d6", size=5.3)
    pill(ax, x0 + 33.5, 12.8, 16.5, "illegal rate", "#fde4e1", size=5.3, tc=P["red"])
    text(ax, x0 + cw / 2, 5.0, "Same encoder, same legal graph, same compute;\n"
         "only the router changes.",
         size=5.9, ha="center", color=P["sage"], linespacing=1.35)
    arrow(ax, cols[0] + cw + 1.5, 26, cols[1] - 1.5, 26, color=P["sage"], lw=1.6, ms=9)
    arrow(ax, cols[1] + cw + 1.5, 26, cols[2] - 1.5, 26, color=P["sage"], lw=1.6, ms=9)
    return fig

if __name__ == "__main__":
    save(build(False), "method", tight=False)
    save(build(True), "method_slide", tight=False)
