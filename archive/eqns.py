import os
os.environ["MPLCONFIGDIR"] = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS, HS = 16, 15
INK, ACC, HEAD = "#1a1a1a", "#2d7d46", "#b23b3b"
fig = plt.figure(figsize=(9.6, 13.4)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
y = 0.975
def head(t):
    global y
    ax.text(0.05, y, t, fontsize=HS, fontweight="bold", color=HEAD, transform=ax.transAxes)
    y -= 0.040
def eq(s, note="", c=INK):
    global y
    ax.text(0.09, y, s, fontsize=FS, color=c, transform=ax.transAxes, math_fontfamily="cm")
    if note:
        ax.text(0.60, y + 0.004, note, fontsize=10.5, color="#666", transform=ax.transAxes, style="italic")
    y -= 0.049
def gap(h=0.012):
    global y; y -= h

ax.text(0.05, y, "Mycelial Routing — the equations", fontsize=20, fontweight="bold",
        color=INK, transform=ax.transAxes); y -= 0.055

head("Objects")
eq(r"$N \;=\; 2 + KM$", "source + sink + (position x primitive) nodes")
eq(r"$(k,m)\!\to\!(k{+}1,m')\ \ \mathrm{exists}\ \Leftrightarrow\ T_{m,m'}=1$", "legality is topology")
eq(r"$B_{i,e}=+1,\ \ B_{j,e}=-1\ \ \mathrm{for\ edge}\ e=(i\!\to\! j)$", "signed incidence")
gap()

head("Step 1   input  ->  edge costs")
eq(r"$L_e \;=\; \mathrm{softplus}(-z_{k,m}) \,+\, \varepsilon$", "high logit -> low cost")
gap()

head("Step 2   conserved-flow dynamic  (Physarum)")
eq(r"$w_e \;=\; D_e \,/\, L_e$", "conductance / cost")
eq(r"$Q_e \;=\; w_e\,(B^{\top}p)_e \;=\; w_e\,(p_i-p_j)$", "Ohm: flux from pressure drop")
eq(r"$\mathcal{L}(w)\,p \;=\; b,\qquad \mathcal{L}(w)=B\,\mathrm{diag}(w)\,B^{\top}$", "Kirchhoff = Laplacian solve")
eq(r"$D_e \;\leftarrow\; D_e + \Delta t\,(\,|Q_e| - D_e\,)$", "thicken what flows")
eq(r"$D_e^{*} \;=\; |\,Q_e^{*}(D^{*})\,|$", "fixed point = emergent path")
gap()

head("Step 3   flow  ->  selection")
eq(r"$t_v \;=\; \frac{1}{2}\,\sum_e |B_{v,e}|\,|Q_e|$", "node throughput")
eq(r"$a_{k,m} \;=\; \dfrac{t_{(k,m)}^{\,\gamma}}{\sum_{m'} t_{(k,m')}^{\,\gamma}}$", "sharpen to a simplex", ACC)
gap(0.028)

head("Step 4   sequential composition")
eq(r"$h_0 = x,\qquad h_{k+1} \;=\; \sum_m a_{k,m}\; g_m(h_k)$", "select-then-apply", ACC)
eq(r"$\hat{y} \;=\; h_K \;=\; g_{m_K}\!\circ\cdots\circ g_{m_1}(x)$", "composition, not average")
gap()

head("Step 5   learn")
eq(r"$\mathcal{J} \;=\; \|\, \hat{y} - y \,\|^{2}$", "backprop through the whole solve")
gap()

head("The two guarantees")
eq(r"$e \notin E \;\Rightarrow\; Q_e = 0$", "legality by construction", ACC)
eq(r"$\sum_{\mathrm{in}} Q \;=\; \sum_{\mathrm{out}} Q$", "conservation couples positions", ACC)

fig.savefig("mycelial_equations.png", dpi=145, facecolor="white", bbox_inches="tight")
print("saved mycelial_equations.png")
