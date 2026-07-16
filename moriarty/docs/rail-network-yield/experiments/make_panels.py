"""
Generate 5 validation panels for the rail-network-yield paper.

Each panel: white background, four charts in a row, minimal text, at least one
3D chart. Every chart plots real numbers computed from the model in model.py --
no conceptual diagrams, no text-only charts, no tables.

Output: ../figures/panel_1_equivalence.png ... panel_5_quantum.png

Run:  python make_panels.py
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model import (
    Section, Vehicle, Item, Network,
    blockseconds, blocksecond_cost, useful_displacement, configuration_yield,
    section_optimal_speed, yield_density, capability_ceiling,
    brute_force_optimum_delivering, separation_cost, enumerate_configurations,
    Configuration, Assignment,
)
from model import _delivers_all  # noqa
from run_experiments import _beta0_in_yield_units  # noqa

# ---- style ----
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
})

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# a restrained categorical palette
C_H = "#1f6feb"    # high-capability / primary
C_R = "#e8710a"    # regional / secondary
C_ACC = "#2a9d5c"  # accent
C_GREY = "#888888"
CMAP = "viridis"

PANEL_W, PANEL_H = 16, 4.0  # inches; 4 charts in a row


def _finish(fig, path):
    fig.tight_layout(pad=1.4)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


# ======================================================================
# Shared model instances
# ======================================================================

def corridor_ABC():
    coords = {"A": 0.0, "B": 100.0, "C": 200.0}
    sections = [
        Section("AB_fast", "A", "B", 100.0, 300.0),
        Section("AB_slow", "A", "B", 100.0, 160.0),
        Section("BC", "B", "C", 100.0, 300.0),
    ]
    net = Network(list(coords), sections, coords, beta0_s=120.0)
    vehicles = {"H": Vehicle("H", 300.0, 2, 0.0, 1e-6),
                "R": Vehicle("R", 160.0, 2, 0.0, 1e-6)}
    items = {"x1": Item("x1", "A", "C"), "x2": Item("x2", "A", "C")}
    return net, vehicles, items


def nbg_berlin():
    coords = {"Nuremberg": 0.0, "Erfurt": 190.0, "HalleLeipzig": 310.0,
              "Bitterfeld": 340.0, "Berlin": 460.0}
    sections = [
        Section("NBG-Erf", "Nuremberg", "Erfurt", 190.0, 300.0),
        Section("Erf-HL", "Erfurt", "HalleLeipzig", 120.0, 300.0),
        Section("HL-Bit", "HalleLeipzig", "Bitterfeld", 30.0, 160.0),
        Section("Bit-Ber", "Bitterfeld", "Berlin", 120.0, 160.0),
    ]
    net = Network(list(coords), sections, coords, beta0_s=120.0)
    return net, sections


# ======================================================================
# Panel 1: Equivalence
# ======================================================================

def panel_1():
    net, vehicles, items = corridor_ABC()
    lam = 1e-3
    y_star, c_star = brute_force_optimum_delivering(net, vehicles, items, lam, "C", 6)

    # (A) sorted yields of task-delivering single-section neighbours
    sec_names = [s.name for s in net.sections]
    neigh_yields = []
    for c in enumerate_configurations(net, vehicles, items, 6):
        diffs = [nm for nm in sec_names if c.assignments[nm] != c_star.assignments[nm]]
        if len(diffs) != 1:
            continue
        if not _delivers_all(net, c, items, "C"):
            continue
        neigh_yields.append(configuration_yield(net, c, vehicles, items, lam))
    neigh_yields = np.sort(np.array(neigh_yields))[::-1]

    # (B) separation costs per used section
    used = [nm for nm, a in c_star.assignments.items() if a is not None]
    sep = {nm: separation_cost(net, vehicles, items, lam, nm, y_star, 6, "C") for nm in used}

    # (C) closure gap & clearing deviation profit vs beta0 floor
    beta0_yield = _beta0_in_yield_units(net, c_star, vehicles, items, lam)
    best_gain = max([0.0] + [y - y_star for y in neigh_yields])
    # clearing deviation: recompute max deviation profit under p=sep
    sec_by = {s.name: s for s in net.sections}
    max_dev = 0.0
    for nm, a in c_star.assignments.items():
        if a is None:
            continue
        s = sec_by[nm]
        price = sep.get(nm, 0.0)
        carried = [items[x] for x in a.items]
        inc = (useful_displacement(net, s, carried)
               - price * blockseconds(s, a.v_kmh, net.beta0_s)
               - lam * s.length_km * vehicles[a.vehicle].g(a.v_kmh))
        for veh2 in vehicles.values():
            ceil = capability_ceiling(veh2, s)
            for i in range(1, 21):
                v2 = ceil * i / 20
                dev = (useful_displacement(net, s, carried)
                       - price * blockseconds(s, v2, net.beta0_s)
                       - lam * s.length_km * veh2.g(v2))
                max_dev = max(max_dev, dev - inc)

    # (D) yield surface over (upstream speed, downstream speed) at fixed items
    ceil_up = capability_ceiling(vehicles["H"], sec_by["AB_fast"])
    ceil_dn = capability_ceiling(vehicles["H"], sec_by["BC"])
    vs_up = np.linspace(60, ceil_up, 40)
    vs_dn = np.linspace(60, ceil_dn, 40)
    VU, VD = np.meshgrid(vs_up, vs_dn)
    Z = np.zeros_like(VU)
    for i in range(VU.shape[0]):
        for j in range(VU.shape[1]):
            cfg = Configuration({
                "AB_fast": Assignment("AB_fast", "H", VU[i, j], ("x1", "x2")),
                "AB_slow": None,
                "BC": Assignment("BC", "H", VD[i, j], ("x1", "x2")),
            })
            Z[i, j] = configuration_yield(net, cfg, vehicles, items, lam)

    # ---- render ----
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(range(1, len(neigh_yields) + 1), neigh_yields, "o-", color=C_H, ms=3, lw=1)
    ax1.axhline(y_star, ls="--", color=C_R, lw=1.4)
    ax1.set_xlabel("neighbour rank")
    ax1.set_ylabel("yield")
    ax1.set_title("A  Neighbours vs optimum")

    ax2 = fig.add_subplot(1, 4, 2)
    names = list(sep)
    ax2.bar(names, [sep[n] for n in names], color=[C_R, C_H][:len(names)], width=0.55)
    ax2.set_ylabel(r"separation cost $\varsigma(\sigma)$")
    ax2.set_title("B  Separation cost by section")

    ax3 = fig.add_subplot(1, 4, 3)
    xs = [0, 1]
    vals = [best_gain, max_dev]
    ax3.axhspan(0, beta0_yield, color=C_ACC, alpha=0.12)
    ax3.axhline(beta0_yield, ls="--", color=C_GREY, lw=1.4)
    ax3.scatter(xs, vals, color=C_H, s=110, zorder=3)
    for x, v in zip(xs, vals):
        ax3.annotate(f"{v:.3g}", (x, v), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=8)
    ax3.annotate(r"$\beta_0$ floor", (1, beta0_yield), xytext=(0, 4),
                 textcoords="offset points", ha="right", fontsize=7, color=C_GREY)
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["closure\ngap", "clearing\ndev."])
    ax3.set_ylabel("yield units")
    ax3.set_ylim(-0.15 * beta0_yield, 1.3 * beta0_yield)
    ax3.set_xlim(-0.6, 1.6)
    ax3.set_title(r"C  Gaps below $\beta_0$ floor")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.plot_surface(VU, VD, Z, cmap=CMAP, linewidth=0, antialiased=True, alpha=0.9)
    a = c_star.assignments
    zopt = y_star
    ax4.scatter([a["AB_fast"].v_kmh if a["AB_fast"] else ceil_up],
                [a["BC"].v_kmh if a["BC"] else ceil_dn], [zopt],
                color=C_R, s=40, depthshade=False)
    ax4.set_xlabel("upstream v")
    ax4.set_ylabel("downstream v")
    ax4.set_zlabel("yield")
    ax4.set_title("D  Yield surface")
    ax4.view_init(elev=22, azim=-60)

    _finish(fig, os.path.join(FIGDIR, "panel_1_equivalence.png"))


# ======================================================================
# Panel 2: Forced optimal speed
# ======================================================================

def panel_2():
    net = Network(["P", "Q"], [Section("PQ", "P", "Q", 150.0, 300.0)],
                  {"P": 0.0, "Q": 150.0}, beta0_s=120.0)
    sec = net.sections[0]
    veh = Vehicle("H", 300.0, 1, 0.0, 1.0)
    delta = 150.0
    ceil = capability_ceiling(veh, sec)
    speeds = np.linspace(20, ceil, 300)
    lambdas = [0.0, 1e-7, 1e-6, 1e-5, 1e-4]

    # (A) ratio vs speed for several lambda
    ratio_curves = {}
    vstars = []
    for lam in lambdas:
        r = np.array([delta / blocksecond_cost(sec, veh, v, net.beta0_s, lam) for v in speeds])
        ratio_curves[lam] = r
        vst, _ = section_optimal_speed(sec, veh, delta, net.beta0_s, lam, grid=3000)
        vstars.append(vst)

    # (C) denominator decomposition at lam=1e-6
    lam_c = 1e-6
    n_bs = np.array([blockseconds(sec, v, net.beta0_s) for v in speeds])
    e_bs = np.array([lam_c * sec.length_km * veh.g(v) for v in speeds])

    # (D) surface ratio over (speed, lambda)
    lam_grid = np.linspace(0, 1e-4, 40)
    SP, LM = np.meshgrid(speeds, lam_grid)
    Z = np.zeros_like(SP)
    for i in range(SP.shape[0]):
        for j in range(SP.shape[1]):
            Z[i, j] = delta / blocksecond_cost(sec, veh, SP[i, j], net.beta0_s, LM[i, j])

    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax1 = fig.add_subplot(1, 4, 1)
    cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(lambdas)))
    for k, lam in enumerate(lambdas):
        ax1.plot(speeds, ratio_curves[lam], color=cols[k], lw=1.4)
        vi = int(np.argmax(ratio_curves[lam]))
        ax1.plot(speeds[vi], ratio_curves[lam][vi], "o", color=cols[k], ms=4)
    ax1.set_xlabel("speed (km/h)")
    ax1.set_ylabel("yield ratio")
    ax1.set_title("A  Ratio vs speed")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(lambdas, vstars, "o-", color=C_H, lw=1.4, ms=5)
    ax2.axhline(ceil, ls="--", color=C_GREY, lw=1.2)
    ax2.set_xscale("symlog", linthresh=1e-7)
    ax2.set_xlabel(r"energy weight $\lambda$")
    ax2.set_ylabel(r"$v^\star$ (km/h)")
    ax2.set_title(r"B  $v^\star$ vs $\lambda$")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.stackplot(speeds, n_bs, e_bs, colors=[C_H, C_R], alpha=0.85,
                  labels=["block-seconds", "energy-wear"])
    ax3.set_xlabel("speed (km/h)")
    ax3.set_ylabel("cost")
    ax3.legend(loc="upper center", fontsize=7, frameon=False)
    ax3.set_title("C  Cost decomposition")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.plot_surface(SP, LM, Z, cmap=CMAP, linewidth=0, antialiased=True, alpha=0.92)
    # ridge of maxima
    ridge_v = speeds[np.argmax(Z, axis=1)]
    ax4.plot(ridge_v, lam_grid, np.max(Z, axis=1), color=C_R, lw=2)
    ax4.set_xlabel("speed")
    ax4.set_ylabel(r"$\lambda$")
    ax4.set_zlabel("ratio")
    ax4.set_title(r"D  Ratio surface + ridge")
    ax4.view_init(elev=24, azim=-52)

    _finish(fig, os.path.join(FIGDIR, "panel_2_forced_speed.png"))


# ======================================================================
# Panel 3: Comparative-advantage sorting
# ======================================================================

def panel_3():
    net, sections = nbg_berlin()
    H = Vehicle("H", 300.0, 1, 0.0, 1e-7)
    R = Vehicle("R", 160.0, 1, 0.0, 1e-7)
    lam = 1e-5
    labels = [s.name for s in sections]

    dH, dR, nH, nR, vlims = [], [], [], [], []
    for sec in sections:
        x = Item("x", sec.tail, "Berlin")
        dH.append(yield_density(net, sec, H, [x], net.beta0_s, lam))
        dR.append(yield_density(net, sec, R, [x], net.beta0_s, lam))
        vH = capability_ceiling(H, sec)
        vR = capability_ceiling(R, sec)
        nH.append(blockseconds(sec, vH, net.beta0_s))
        nR.append(blockseconds(sec, vR, net.beta0_s))
        vlims.append(sec.vmax_kmh)
    dH, dR = np.array(dH), np.array(dR)

    # (D) density surface over (section speed limit, vehicle capability)
    seclims = np.linspace(120, 320, 40)
    caps = np.linspace(120, 320, 40)
    SL, CP = np.meshgrid(seclims, caps)
    Z = np.zeros_like(SL)
    ref = sections[0]
    for i in range(SL.shape[0]):
        for j in range(SL.shape[1]):
            s = Section("s", "Nuremberg", "Erfurt", ref.length_km, SL[i, j])
            v = Vehicle("v", CP[i, j], 1, 0.0, 1e-7)
            x = Item("x", "Nuremberg", "Berlin")
            Z[i, j] = yield_density(net, s, v, [x], net.beta0_s, lam)

    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    xpos = np.arange(len(labels))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.bar(xpos - 0.2, dH, width=0.4, color=C_H, label="high-cap")
    ax1.bar(xpos + 0.2, dR, width=0.4, color=C_R, label="regional")
    ax1.set_xticks(xpos); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("yield density D")
    ax1.legend(fontsize=7, frameon=False)
    ax1.set_title("A  Density by section")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.bar(xpos - 0.2, nH, width=0.4, color=C_H)
    ax2.bar(xpos + 0.2, nR, width=0.4, color=C_R)
    ax2.set_xticks(xpos); ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("block-seconds")
    ax2.set_title("B  Block-seconds used")

    ax3 = fig.add_subplot(1, 4, 3)
    adv = dH - dR
    ax3.scatter(vlims, adv, c=[C_H if a > 1e-9 else C_GREY for a in adv], s=70, zorder=3)
    # jitter annotations vertically to avoid overlap where limits coincide
    seen = {}
    for k, lb in enumerate(labels):
        key = round(vlims[k])
        off = seen.get(key, 0)
        seen[key] = off + 1
        ax3.annotate(lb, (vlims[k], adv[k]), fontsize=6,
                     xytext=(5, 4 + 10 * off), textcoords="offset points")
    ax3.axhline(0, ls="--", color=C_GREY, lw=1)
    ax3.set_xlabel("section speed limit (km/h)")
    ax3.set_ylabel(r"$D_H - D_R$")
    ax3.set_title("C  Advantage vs limit")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.plot_surface(SL, CP, Z, cmap=CMAP, linewidth=0, antialiased=True, alpha=0.9)
    ax4.set_xlabel("section limit")
    ax4.set_ylabel("vehicle cap.")
    ax4.set_zlabel("density")
    ax4.set_title("D  Density surface")
    ax4.view_init(elev=24, azim=-58)

    _finish(fig, os.path.join(FIGDIR, "panel_3_sorting.png"))


# ======================================================================
# Panel 4: Liveness
# ======================================================================

def panel_4():
    coords = {"F": -30.0, "A": 0.0, "B": 100.0, "C": 200.0, "D": 300.0, "E": 60.0}
    sections = [
        Section("FA", "F", "A", 30.0, 160.0),
        Section("AB", "A", "B", 100.0, 300.0),
        Section("EB", "E", "B", 45.0, 160.0),
        Section("BC", "B", "C", 100.0, 300.0),
        Section("CD", "C", "D", 100.0, 300.0),
    ]
    net = Network(list(coords), sections, coords, beta0_s=120.0)
    items = [Item("i1", "F", "D"), Item("i2", "A", "D"), Item("i3", "E", "D"),
             Item("i4", "B", "D"), Item("i5", "C", "D")]
    locs = {x.name: x.loc for x in items}
    tgt = {x.name: x.target for x in items}

    history = []      # residual per item per round
    settled_ct = []
    pressure = []     # total residual (undrained pressure proxy)
    max_rounds = 40
    for rnd in range(max_rounds):
        res = {nm: net.rho(locs[nm], tgt[nm]) for nm in locs}
        history.append(dict(res))
        settled_ct.append(sum(1 for r in res.values() if r == 0.0))
        pressure.append(sum(res.values()))
        if all(r == 0.0 for r in res.values()):
            break
        load = {s.name: 0 for s in net.sections}
        order = sorted(locs, key=lambda nm: -res[nm])
        new = dict(locs)
        for nm in order:
            if res[nm] == 0.0:
                continue
            b = locs[nm]
            best, bp = None, 0.0
            for s in net.sections_from(b):
                if load[s.name] >= 1:
                    continue
                prog = net.rho(b, tgt[nm]) - net.rho(s.head, tgt[nm])
                if prog > bp:
                    bp, best = prog, s
            if best is not None:
                new[nm] = best.head
                load[best.name] += 1
        locs = new

    rounds = np.arange(len(history))
    item_names = [x.name for x in items]
    res_mat = np.array([[history[r][nm] for nm in item_names] for r in range(len(history))])

    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax1 = fig.add_subplot(1, 4, 1)
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(item_names)))
    for k, nm in enumerate(item_names):
        ax1.plot(rounds, res_mat[:, k], "o-", color=cols[k], ms=3, lw=1.2)
    ax1.set_xlabel("round")
    ax1.set_ylabel("residual distance (km)")
    ax1.set_title("A  Item residuals")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.plot(rounds, pressure, "o-", color=C_R, lw=1.6, ms=4)
    ax2.fill_between(rounds, pressure, color=C_R, alpha=0.15)
    ax2.set_xlabel("round")
    ax2.set_ylabel("total undrained pressure (km)")
    ax2.set_title("B  Pressure decay")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.step(rounds, settled_ct, where="post", color=C_ACC, lw=1.8)
    ax3.axhline(len(item_names), ls="--", color=C_GREY, lw=1.2)
    ax3.set_xlabel("round")
    ax3.set_ylabel("items settled")
    ax3.set_ylim(0, len(item_names) + 0.5)
    ax3.set_title("C  Settled count")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    IX, RX = np.meshgrid(np.arange(len(item_names)), rounds)
    ax4.plot_surface(IX, RX, res_mat, cmap=CMAP, linewidth=0, antialiased=True, alpha=0.92)
    ax4.set_xlabel("item")
    ax4.set_ylabel("round")
    ax4.set_zlabel("residual (km)")
    ax4.set_xticks(np.arange(len(item_names)))
    ax4.set_xticklabels(item_names, fontsize=6)
    ax4.set_title("D  Residual surface")
    ax4.view_init(elev=26, azim=-46)

    _finish(fig, os.path.join(FIGDIR, "panel_4_liveness.png"))


# ======================================================================
# Panel 5: Quantum coincidence
# ======================================================================

def panel_5():
    net, vehicles, items = corridor_ABC()
    lam = 1e-3
    y_star, c_star = brute_force_optimum_delivering(net, vehicles, items, lam, "C", 6)
    beta0 = _beta0_in_yield_units(net, c_star, vehicles, items, lam)

    trials = [(1, 1, 1), (1, 3, 1), (1, 1, 5), (2, 4, 6), (1, 5, 2)]
    slack, band, split, labels = [], [], [], []
    for (mp, ma, me) in trials:
        slack.append(max(mp, ma, me) * beta0)
        band.append(abs(ma - me) * beta0)
        split.append(abs(ma - me) * beta0)
        labels.append(f"{mp}/{ma}/{me}")

    # (D) slack surface over (alg, econ) multipliers at fixed phys=1
    ma_grid = np.linspace(1, 6, 40)
    me_grid = np.linspace(1, 6, 40)
    MA, ME = np.meshgrid(ma_grid, me_grid)
    SLACK = np.maximum(np.maximum(1.0, MA), ME) * beta0

    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    xpos = np.arange(len(trials))
    maxconst = [max(t) for t in trials]

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(maxconst, np.array(slack) / beta0, c=C_H, s=60, zorder=3)
    lo, hi = min(maxconst), max(maxconst)
    ax1.plot([lo, hi], [lo, hi], ls="--", color=C_GREY, lw=1.2)
    ax1.set_xlabel(r"$\max$ multiplier")
    ax1.set_ylabel(r"slack / $\beta_0$")
    ax1.set_title("A  Slack tracks max")

    ax2 = fig.add_subplot(1, 4, 2)
    order = np.argsort(split)
    ax2.plot(np.array(split)[order] / beta0, np.array(band)[order] / beta0,
             "o-", color=C_R, lw=1.4, ms=5)
    ax2.set_xlabel(r"$|\beta^{alg}-\beta^{econ}|/\beta_0$")
    ax2.set_ylabel(r"price band / $\beta_0$")
    ax2.set_title("B  Band vs split")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.bar(xpos - 0.2, np.array(slack) / beta0, width=0.4, color=C_H, label="slack")
    ax3.bar(xpos + 0.2, np.array(band) / beta0, width=0.4, color=C_R, label="band")
    ax3.set_xticks(xpos); ax3.set_xticklabels(labels, fontsize=7)
    ax3.set_xlabel("trial (phys/alg/econ)")
    ax3.set_ylabel(r"$/\beta_0$")
    ax3.legend(fontsize=7, frameon=False)
    ax3.set_title("C  Degradation per trial")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.plot_surface(MA, ME, SLACK / beta0, cmap=CMAP, linewidth=0, antialiased=True, alpha=0.9)
    ax4.plot(ma_grid, me_grid, np.maximum(np.maximum(1.0, ma_grid), me_grid),
             color=C_R, lw=2)  # diagonal min ridge
    ax4.set_xlabel("alg mult.")
    ax4.set_ylabel("econ mult.")
    ax4.set_zlabel(r"slack / $\beta_0$")
    ax4.set_title("D  Slack surface")
    ax4.view_init(elev=24, azim=-56)

    _finish(fig, os.path.join(FIGDIR, "panel_5_quantum.png"))


def main():
    panel_1()
    panel_2()
    panel_3()
    panel_4()
    panel_5()
    print("All 5 panels written to", os.path.relpath(FIGDIR))


if __name__ == "__main__":
    main()
