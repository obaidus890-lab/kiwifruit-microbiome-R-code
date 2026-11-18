# -*- coding: utf-8 -*-
"""Research analysis helper. See README for pipeline and methodology."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import mannwhitneyu, spearmanr

from analysis_configuration import COMP_ORDER, COMP_LABEL, FIGS, TABLES, COLORS, AAAS, panel

STRAT_COLOR = {"Colonizer": "#3C5488", "Transmitter": "#00A087",
               "Transient": "#B09C85"}
STRATS = ["Colonizer", "Transmitter", "Transient"]
AG = ["S", "L", "F"]
GEPHI = os.path.join("e:\\Article\\Microbiome", "result", "gephi")


def read(path):
    return pd.read_csv(path, sep="\t")


def mw2p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    u, p1 = mannwhitneyu(a, b)
    return min(1.0, 2 * min(p1, 1 - p1))


def panel_a(ax, ds):
    comp = read(os.path.join(TABLES, "tab_sec15_strategy_comp_%s.tsv" % ds))
    comp = comp.set_index("compartment").reindex(COMP_ORDER)
    x = np.arange(len(comp))
    bot = np.zeros(len(comp))
    for s in STRATS:
        ax.bar(x, comp[s.lower()], bottom=bot, color=STRAT_COLOR[s],
               width=0.62, label=s, edgecolor="white", linewidth=0.4)
        bot += comp[s.lower()].values
    ax.set_xticks(x)
    ax.set_xticklabels([COMP_LABEL[c] for c in COMP_ORDER], rotation=30,
                       ha="right")
    ax.set_ylabel("Proportion of genera")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5, ncol=3, bbox_to_anchor=(0.5, 1.06),
              loc="lower center", handlelength=1.2, columnspacing=1)
    panel(ax, "a")


def panel_b(ax, ds):
    site = read(os.path.join(TABLES, "tab_sec15_strategy_site_%s.tsv" % ds))
    sites = ["WH", "SH", "SX"]
    w = 0.24
    x = np.arange(len(sites))
    for i, s in enumerate(STRATS):
        ax.bar(x + (i - 1) * w, site[s.lower()], width=w,
               color=STRAT_COLOR[s], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Proportion of genera")
    ax.set_ylim(0, 1.0)
    from scipy.stats import chi2_contingency
    tab = [[int(round(site.loc[i, s.lower()] * site.loc[i, "n_genera"]))
            for s in STRATS] for i in range(len(sites))]
    p = chi2_contingency(np.array(tab))[1]
    ax.set_title("site chi-square %s" %
                 ("P < 0.001" if p < 0.001 else "P = %.3f" % p),
                 fontsize=7, fontweight="normal")
    panel(ax, "b")


def panel_c(ax, ds):
    m = read(os.path.join(TABLES, "tab_sec15_strategy_icamp_%s.tsv" % ds))
    ax.scatter(m["colonizer_frac"], m["deterministic_frac"],
               c="#3C5488", s=26, edgecolor="k", linewidth=0.4, zorder=3)
    for _, r in m.iterrows():
        ax.annotate(r["compartment"], (r["colonizer_frac"],
                                       r["deterministic_frac"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    r, p = spearmanr(m["colonizer_frac"], m["deterministic_frac"])
    ax.set_xlabel("Colonizer proportion")
    ax.set_ylabel("Deterministic fraction (iCAMP)")
    ax.text(0.04, 0.95, "Spearman rho = %.2f\nP = %.2f" % (r, p),
            transform=ax.transAxes, fontsize=7, va="top")
    panel(ax, "c")


def panel_d(ax, ds):
    topo = read(os.path.join(TABLES, "tab_sec15_net_topo_%s.tsv" % ds))
    ss = topo[topo.scope_kind == "site"].copy()
    ss["group"] = ss["scope"].str.split(".").str[0] \
        .map(lambda c: "Aboveground" if c in AG else "Belowground")
    pars = [("n_edges", "Edges"), ("mean_degree", "Mean degree"),
            ("modularity", "Modularity"), ("pos_edge_ratio",
                                           "Positive edge ratio")]
    xpos = np.arange(len(pars))
    for i, (col, lab) in enumerate(pars):
        a = ss[ss.group == "Aboveground"][col].values
        b = ss[ss.group == "Belowground"][col].values
        mx = np.nanmax(np.concatenate([a, b]))
        a, b = a / mx, b / mx
        rng = np.random.default_rng(7)
        ab, bb = a[~np.isnan(a)], b[~np.isnan(b)]
        ax.scatter(np.full(len(bb), i - 0.12) +
                   rng.uniform(-0.06, 0.06, len(bb)), bb, s=14,
                   c="#7E6148", edgecolor="k", linewidth=0.3, zorder=3)
        ax.scatter(np.full(len(ab), i + 0.12) +
                   rng.uniform(-0.06, 0.06, len(ab)),
                   ab, s=14, c="#00A087", edgecolor="k", linewidth=0.3,
                   zorder=3)
        p = mw2p(a, b)
        ax.text(i, 1.08, "P=%.3f" % p, ha="center", fontsize=6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([p[1] for p in pars], fontsize=6.5, rotation=25,
                       ha="right")
    ax.set_ylabel("Normalized value (per-parameter max = 1)")
    ax.set_ylim(0, 1.25)
    panel(ax, "d")


def draw_net(ax, ds, name, tag, letter):
    nodes = pd.read_csv(os.path.join(GEPHI, "%s_%s_nodes.csv" % (ds, name)))
    edges = pd.read_csv(os.path.join(GEPHI, "%s_%s_edges.csv" % (ds, name)))
    g = nx.Graph()
    for _, r in nodes.iterrows():
        g.add_node(r["Id"], **r.to_dict())
    for _, r in edges.iterrows():
        g.add_edge(r["Source"], r["Target"], weight=r["Weight"],
                   sign=r["Sign"])
    pos = nx.spring_layout(g, seed=42, k=1.6 / np.sqrt(max(g.number_of_nodes(),
                                                           1)))
    mod_colors = {m: AAAS[i % len(AAAS)]
                  for i, m in enumerate(sorted(nodes["Module"].unique()))}
    ncol = [mod_colors[nodes.set_index("Id").loc[n, "Module"]]
            for n in g.nodes()]
    nsize = [min(28 + 2.2 * nodes.set_index("Id").loc[n, "Degree"], 220)
             for n in g.nodes()]
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.35, alpha=0.5,
                           edge_color=[("#E64B35" if d["sign"] == "negative"
                                        else "#999999")
                                       for _, _, d in g.edges(data=True)])
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=ncol, node_size=nsize,
                           linewidths=0.4, edgecolors="k", alpha=0.9)
    ks = nodes[nodes["Keystone"]]
    nx.draw_networkx_nodes(g, pos, nodelist=list(ks["Id"]), ax=ax,
                           node_size=[min(28 + 2.2 * r["Degree"], 220)
                                      for _, r in ks.iterrows()],
                           node_color=[mod_colors[r["Module"]]
                                       for _, r in ks.iterrows()],
                           edgecolors="#DC0000", linewidths=1.3)
    top = ks.sort_values("Degree", ascending=False).head(5)
    for _, r in top.iterrows():
        lab = str(r["Id"])[:18] + ("..." if len(str(r["Id"])) > 18 else "")
        ax.annotate(lab, pos[r["Id"]], fontsize=5,
                    xytext=(3, 3), textcoords="offset points",
                    color="#404040",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.75))
    ax.set_title("%s (%s)" % (tag, name), fontsize=7.5)
    ax.set_axis_off()
    if letter:
        panel(ax, letter)


def panel_f(ax, ds):
    key = read(os.path.join(TABLES, "tab_sec15_keystone_%s.tsv" % ds))
    key = key[key["scope"].isin(COMP_ORDER)]
    hub_col = {"Network hub": "#DC0000", "Module hub": "#E64B35",
               "Connector hub": "#F39B7F", "Peripheral": "#B0B0B0"}
    for ht, sub in key.groupby("hub_type"):
        ax.scatter(sub["Pi"], sub["Zi"], s=14, c=hub_col[ht],
                   edgecolor="none" if ht == "Peripheral" else "k",
                   linewidth=0.3, label="%s (n=%d)" % (ht, len(sub)),
                   alpha=0.85)
    ax.axvline(0.62, ls="--", lw=0.7, c="#606060")
    ax.axhline(2.5, ls="--", lw=0.7, c="#606060")
    top = key[(key.Zi > 2.5) | (key.Pi > 0.62)].sort_values(
        "Zi", ascending=False).head(5)
    for _, r in top.iterrows():
        lab = str(r["genus"])[:22] + ("..." if len(str(r["genus"])) > 22
                                      else "")
        ax.annotate("%s (%s)" % (lab, r["scope"]), (r["Pi"], r["Zi"]),
                    fontsize=5.5, xytext=(3, 2),
                    textcoords="offset points")
    ax.set_xlabel("Among-module connectivity Pi")
    ax.set_ylabel("Within-module connectivity Zi")
    ax.legend(fontsize=6, loc="upper left", markerscale=1.1,
              handlelength=1, frameon=True, framealpha=0.85,
              edgecolor="none")
    panel(ax, "f")


def panel_g(ax, ds):
    rob = read(os.path.join(TABLES, "tab_sec15_robust_%s.tsv" % ds))
    for scope, col in [("AG", "#00A087"), ("BG", "#7E6148")]:
        sub = rob[rob.scope == scope]
        if len(sub) == 0:
            continue
        ax.plot(sub.frac, sub.random_mean, c=col, ls=":", lw=1.0,
                label="%s random" % scope)
        ax.fill_between(sub.frac,
                        sub.random_mean - sub.random_sd,
                        sub.random_mean + sub.random_sd,
                        color=col, alpha=0.12, linewidth=0)
        ax.plot(sub.frac, sub.targeted, c=col, ls="-", lw=1.3,
                label="%s targeted (degree)" % scope)
        ax.plot(sub.frac, sub.keystone_first, c=col, ls="--", lw=1.3,
                label="%s keystone-first" % scope)
    ax.set_xlabel("Fraction of nodes removed")
    ax.set_ylabel("Relative largest connected component")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), handlelength=1.8,
              columnspacing=1.2)
    panel(ax, "g")


def panel_h(ax, ds):
    gs = read(os.path.join(TABLES, "tab_sec15_net_genspec_%s.tsv" % ds))
    gs = gs.set_index("scope").reindex(COMP_ORDER)
    x = np.arange(len(gs))
    ax.bar(x, gs["prop_generalist"], color="#3C5488", width=0.62,
           label="Generalist", edgecolor="white", linewidth=0.4)
    ax.bar(x, 1 - gs["prop_generalist"], bottom=gs["prop_generalist"],
           color="#B09C85", width=0.62, label="Specialist",
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([COMP_LABEL[c] for c in COMP_ORDER], rotation=30,
                       ha="right")
    ax.set_ylabel("Proportion of edges")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=6.5, ncol=2, bbox_to_anchor=(0.5, 1.06),
              loc="lower center", handlelength=1.2, columnspacing=1)
    panel(ax, "h")


def panel_i(ax, ds):
    ns = read(os.path.join(TABLES, "tab_sec15_net_nodestat_%s.tsv" % ds))
    key = ns[ns["keystone"]]
    agg = key.groupby("genus").agg(n_key=("scope", "count"),
                                   mean_deg=("degree", "mean"))
    genera = list(agg.sort_values(["n_key", "mean_deg"],
                                  ascending=False).head(10).index)
    ylab = {g: i for i, g in enumerate(genera)}
    pts = key[key["genus"].isin(genera)]
    im = ax.scatter([COMP_ORDER.index(s) for s in pts["scope"]],
                    [ylab[g] for g in pts["genus"]], c=pts["degree"],
                    cmap="YlOrRd", vmin=0, vmax=pts["degree"].max(),
                    s=64, edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_yticks(range(len(genera)))
    ax.set_yticklabels([g[:22] + ("..." if len(g) > 22 else "")
                        for g in genera], fontstyle="italic", fontsize=6.5)
    ax.set_xticks(range(len(COMP_ORDER)))
    ax.set_xticklabels([COMP_LABEL[c] for c in COMP_ORDER], rotation=30,
                       ha="right")
    ax.set_xlim(-0.6, len(COMP_ORDER) - 0.4)
    ax.set_ylim(len(genera) - 0.5, -0.5)
    ax.grid(alpha=0.25, lw=0.4)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Degree", size=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Top 10 key taxa", fontsize=7.5, fontweight="normal")
    panel(ax, "i")


def make_j(ds):
    ns = read(os.path.join(TABLES, "tab_sec15_net_nodestat_%s.tsv" % ds))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, c in zip(axes.ravel(), COMP_ORDER):
        sub = ns[ns["scope"] == c]
        neg = sub["n_neg"] > sub["n_pos"]
        ax.scatter(sub.loc[neg, "degree"], sub.loc[neg, "log10_neg_pos"],
                   s=9, c="#E64B35", alpha=0.75, edgecolor="none",
                   label="more negative", zorder=3)
        ax.scatter(sub.loc[~neg, "degree"], sub.loc[~neg, "log10_neg_pos"],
                   s=9, c="#3C5488", alpha=0.75, edgecolor="none",
                   label="more positive", zorder=2)
        ax.axhline(0, lw=0.6, c="#808080", ls="--")
        ax.set_title(COMP_LABEL[c], fontsize=7, fontweight="normal")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Degree", fontsize=7)
        ax.set_ylabel("log10(neg/pos)", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="lower right", framealpha=0.85)
    panel(axes[0, 0], "j")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.10,
                        hspace=0.45, wspace=0.35)
    return fig


def make_k(ds):
    '\u5b9a\u6b96\u7b56\u7565 x \u7f51\u7edc\u5173\u8054\uff1a\u8282\u70b9\u7b56\u7565\u6784\u6210 + \u8282\u70b9\u5ea6\u6309\u7b56\u7565\u5bf9\u6bd4\u3002'
    st = read(os.path.join(TABLES, "tab_sec15_net_strategy_%s.tsv" % ds))
    scopes = list(COMP_ORDER) + ["AG", "BG"]
    st = st.set_index("scope").reindex(scopes)
    labs = [COMP_LABEL[c] for c in COMP_ORDER] + ["AG", "BG"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x = np.arange(len(st))
    bot = np.zeros(len(st))
    for s in STRATS:
        vals = st[s.lower() + "_frac"].values
        ax1.bar(x, vals, bottom=bot, color=STRAT_COLOR[s], width=0.62,
                label=s, edgecolor="white", linewidth=0.4)
        bot += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("Node proportion")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=6.5, ncol=3, bbox_to_anchor=(0.5, 1.08),
               loc="lower center", handlelength=1.2, columnspacing=1)
    panel(ax1, "k")

    # Analysis note: see README and original laboratory scripts for methodological details.
    data, pos, cols = [], [], []
    for i, sc in enumerate(scopes):
        nf = os.path.join(GEPHI, "%s_%s_nodes.csv" % (ds, sc))
        if not os.path.exists(nf):
            continue
        nodes = pd.read_csv(nf)
        for j, s in enumerate(STRATS):
            deg = nodes.loc[nodes.Strategy == s, "Degree"].values
            if len(deg):
                data.append(deg)
                pos.append(i + (j - 1) * 0.27)
                cols.append(STRAT_COLOR[s])
    bp = ax2.boxplot(data, positions=pos, widths=0.22, patch_artist=True,
                     showfliers=False, medianprops=dict(color="k", lw=0.8))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for d, p in zip(data, pos):
        rng = np.random.default_rng(11)
        ax2.scatter(np.full(len(d), p) + rng.uniform(-0.05, 0.05, len(d)),
                    d, s=5, c="k", alpha=0.18, edgecolor="none", zorder=2)
    # Analysis note: see README and original laboratory scripts for methodological details.
    nc = pd.concat([pd.read_csv(os.path.join(GEPHI, "%s_%s_nodes.csv"
                                             % (ds, sc)))
                    for sc in scopes
                    if os.path.exists(os.path.join(
                        GEPHI, "%s_%s_nodes.csv" % (ds, sc)))])
    p_ct = mw2p(nc.loc[nc.Strategy == "Colonizer", "Degree"],
                nc.loc[nc.Strategy == "Transient", "Degree"])
    ax2.set_xticks(range(len(scopes)))
    ax2.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("Node degree")
    ax2.set_title("Mann-Whitney Colonizer vs Transient P %s" %
                  ("< 0.001" if p_ct < 0.001 else "= %.3f" % p_ct),
                  fontsize=7, fontweight="normal")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.24,
                        wspace=0.28)
    return fig


ADJ = {
    "a": dict(left=0.13, right=0.98, top=0.86, bottom=0.24),
    "b": dict(left=0.13, right=0.98, top=0.90, bottom=0.12),
    "c": dict(left=0.16, right=0.97, top=0.96, bottom=0.14),
    "d": dict(left=0.14, right=0.97, top=0.96, bottom=0.26),
    "f": dict(left=0.12, right=0.97, top=0.96, bottom=0.14),
    "g": dict(left=0.14, right=0.97, top=0.97, bottom=0.32),
    "h": dict(left=0.13, right=0.98, top=0.86, bottom=0.24),
    "i": dict(left=0.24, right=0.88, top=0.94, bottom=0.20),
}


def main():
    for ds, figno in [("16S", 7), ("ITS", 8)]:
        specs = [("a", (3.7, 3.6), panel_a), ("b", (3.4, 3.3), panel_b),
                 ("c", (3.4, 3.3), panel_c), ("d", (3.9, 3.5), panel_d),
                 ("f", (4.8, 3.7), panel_f), ("g", (4.3, 4.1), panel_g),
                 ("h", (3.7, 3.4), panel_h), ("i", (4.6, 3.8), panel_i)]
        for letter, size, fn in specs:
            fig, ax = plt.subplots(figsize=size)
            fn(ax, ds)
            fig.subplots_adjust(**ADJ[letter])
            out = os.path.join(FIGS, "fig_sec5_%s_%s.png" % (ds, letter))
            fig.savefig(out, dpi=300)
            plt.close(fig)
            print("->", out)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 4.0))
        draw_net(ax1, ds, "BG", "Belowground", "e")
        draw_net(ax2, ds, "AG", "Aboveground", "")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02,
                            wspace=0.05)
        out = os.path.join(FIGS, "fig_sec5_%s_e.png" % ds)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("->", out)
        fig = make_j(ds)
        out = os.path.join(FIGS, "fig_sec5_%s_j.png" % ds)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("->", out)
        fig = make_k(ds)
        out = os.path.join(FIGS, "fig_sec5_%s_k.png" % ds)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("->", out)


if __name__ == "__main__":
    main()
