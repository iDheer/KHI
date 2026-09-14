"""All figures used in the report (static PNGs for the PDF)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # validated all-pairs (light)
OTHER = "#b5b4ad"
THEME_ORDER = ["Technology", "Education", "Ethics & Society", "Environment"]
THEME_COLOR = dict(zip(THEME_ORDER, SERIES))
THEME_MARKER = {"Technology": "o", "Education": "s", "Ethics & Society": "^", "Environment": "D"}
LIKERT_COLORS = ["#a8302f", "#f0a19f", "#d9d8d3", "#86b6ef", "#1c5cab"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#a8302f", "#e89593", "#f0efec", "#86b6ef", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 220,
})


def _save(fig, out: Path, name: str):
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def community_colors(n: int) -> list:
    return SERIES[:n] + [OTHER] * max(0, n - len(SERIES))


# ----------------------------------------------------------------------------- data
def response_distribution(raw: pd.DataFrame, X: pd.DataFrame, out: Path):
    order = [-2, -1, 0, 1, 2]
    names = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
    shares = X.apply(lambda c: c.value_counts(normalize=True)).T.reindex(columns=order).fillna(0) * 100
    fig, axes = plt.subplots(1, 4, figsize=(11, 5.6), sharex=True)
    for ax, (code, theme) in zip(axes, zip("TESV", THEME_ORDER)):
        sub = shares.loc[[c for c in shares.index if c.startswith(code)]][::-1]
        y = np.arange(len(sub))
        left = -(sub[-2] + sub[-1] + sub[0] / 2)
        for val, color in zip(order, LIKERT_COLORS):
            ax.barh(y, sub[val], left=left, color=color, height=0.78, edgecolor=SURFACE, linewidth=1)
            left = left + sub[val]
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(y, sub.index, color=INK2, fontsize=8)
        ax.set_title(theme, color=THEME_COLOR[theme])
        ax.set_xlim(-60, 100)
        ax.set_xticks([-50, 0, 50, 100], ["50%", "0", "50%", "100%"])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LIKERT_COLORS]
    fig.legend(handles, names, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Answer distribution per statement (neutral centred on 0; n = 87 retained respondents)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, out, "fig01_response_distribution.png")


def consensus_plot(items: pd.DataFrame, out: Path):
    df = items.sort_values("consensus_TW")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    x = np.arange(len(df))
    for theme in THEME_ORDER:
        m = df["theme"].to_numpy() == theme
        ax.scatter(x[m], df["consensus_TW"].to_numpy()[m], s=34, color=THEME_COLOR[theme],
                   marker=THEME_MARKER[theme], edgecolor=SURFACE, linewidth=1, label=theme, zorder=3)
    ax.set_xticks(x, df.index, rotation=90, fontsize=7, color=INK2)
    ax.set_ylabel("Tastle–Wierman consensus")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(-1, len(df))
    ax.legend(ncol=4, loc="upper left")
    ax.set_title("Consensus per statement (0 = maximally split, 1 = unanimous), sorted", loc="left")
    fig.tight_layout()
    _save(fig, out, "fig02_consensus.png")


# ----------------------------------------------------------------------------- construction
def construction_diagnostics(S: pd.DataFrame, sweep: pd.DataFrame, k: int, out: Path):
    tri = S.to_numpy()[np.triu_indices(len(S), 1)]
    tri = tri[np.isfinite(tri)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    ax = axes[0]
    ax.hist(tri, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_title("Pairwise similarity (item-centred cosine)", loc="left")
    ax.set_xlabel("similarity")
    ax.set_ylabel("respondent pairs")
    for ax, col, title in [(axes[1], "modularity", "Best Louvain modularity vs k"),
                           (axes[2], "stability_ari", "Partition stability (mean ARI) vs k")]:
        ax.plot(sweep["k"], sweep[col], color=SERIES[0], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE)
        ax.axvline(k, color=SERIES[1], lw=1.2, ls="--")
        ax.text(k + 0.2, ax.get_ylim()[1], f"chosen k = {k}", color=INK2, va="top", fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("k (nearest neighbours)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[1].set_ylim(sweep["modularity"].min() - 0.02, sweep["modularity"].max() + 0.05)
    for _, r in sweep.iterrows():
        axes[1].annotate(int(r["communities"]), (r["k"], r["modularity"]), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, out, "fig03_construction.png")


# ----------------------------------------------------------------------------- respondent network
def respondent_network(G: nx.Graph, labels: dict, cent: pd.DataFrame, names: list, out: Path, highlight=5):
    pos = nx.spring_layout(G, weight="weight", seed=7, k=0.35, iterations=300)
    n_comm = max(labels.values()) + 1
    colors = community_colors(n_comm)
    fig, ax = plt.subplots(figsize=(9, 7))
    widths = [0.3 + 2.2 * max(w, 0) for _, _, w in G.edges(data="weight")]
    same = [labels[u] == labels[v] for u, v in G.edges()]
    ecol = [colors[labels[u]] if s else "#c9c8c1" for (u, v), s in zip(G.edges(), same)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=ecol, alpha=0.35)
    sizes = 60 + 900 * cent.loc[list(G.nodes()), "betweenness"] / cent["betweenness"].max()
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=[colors[labels[n]] for n in G.nodes()],
                           edgecolors=SURFACE, linewidths=1.2)
    top = cent.sort_values("betweenness", ascending=False).index[:highlight]
    for n in top:
        ax.annotate(f"#{n}", pos[n], xytext=(7, 7), textcoords="offset points", fontsize=8.5,
                    color=INK, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=colors[c],
                      markeredgecolor=SURFACE, label=f"{names[c]} (n={sum(1 for v in labels.values() if v == c)})")
               for c in range(n_comm)]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5)
    ax.set_title("Respondent opinion network: kNN backbone, colour = Louvain community, "
                 "size = betweenness (top brokers labelled)", loc="left")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, out, "fig04_respondent_network.png")


def centrality_plots(G: nx.Graph, cent: pd.DataFrame, labels: dict, X: pd.DataFrame, names: list, out: Path):
    n_comm = max(labels.values()) + 1
    colors = community_colors(n_comm)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    deg = cent["degree"]
    bins = np.arange(deg.min(), deg.max() + 2) - 0.5
    axes[0].hist(deg, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
    axes[0].set_title("Degree distribution", loc="left")
    axes[0].set_xlabel("degree")
    axes[0].set_ylabel("respondents")
    mean_score = X.mean(axis=1)
    for c in range(n_comm):
        idx = [n for n in cent.index if labels[n] == c]
        axes[1].scatter(cent.loc[idx, "strength"], cent.loc[idx, "betweenness"], s=26, color=colors[c],
                        edgecolor=SURFACE, linewidth=0.8, label=names[c])
        axes[2].scatter(mean_score.loc[idx], cent.loc[idx, "degree"], s=26, color=colors[c],
                        edgecolor=SURFACE, linewidth=0.8)
    axes[1].set_title("Strength vs betweenness", loc="left")
    axes[1].set_xlabel("strength (sum of similarities)")
    axes[1].set_ylabel("betweenness")
    axes[1].legend(fontsize=7, loc="upper left")
    axes[2].set_title("Overall agreement vs degree", loc="left")
    axes[2].set_xlabel("respondent mean score (−2…+2)")
    axes[2].set_ylabel("degree")
    for ax in axes:
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig05_centrality.png")


def null_model_plot(null_q: np.ndarray, observed_q: float, sw: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), gridspec_kw={"width_ratios": [2, 1, 1]})
    ax = axes[0]
    ax.hist(null_q, bins=25, color=OTHER, edgecolor=SURFACE, linewidth=0.8, label="shuffled answers")
    ax.axvline(observed_q, color=SERIES[1], lw=2, label=f"observed Q = {observed_q:.3f}")
    ax.set_title("Modularity: real data vs 200 column-shuffled datasets", loc="left")
    ax.set_xlabel("best Louvain modularity")
    ax.set_ylabel("count")
    ax.legend(loc="upper left", fontsize=8)
    for ax, key, title in [(axes[1], "C", "Transitivity"), (axes[2], "L", "Mean shortest path")]:
        vals = [sw[key], sw[f"{key}_rand"]]
        ax.bar([0, 1], vals, color=[SERIES[0], OTHER], width=0.6, edgecolor=SURFACE)
        ax.errorbar(1, sw[f"{key}_rand"], yerr=sw[f"{key}_rand_sd"], color=INK2, capsize=3)
        ax.set_xticks([0, 1], ["network", "rewired"])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(title, loc="left")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out, "fig06_null_models.png")


def community_profiles(X: pd.DataFrame, labels: pd.Series, tests: pd.DataFrame, names: list, theme_of,
                       out: Path, top: int = 16):
    comms = sorted(labels.unique())
    theme_means = pd.DataFrame({names[c]: [X.loc[labels.index[labels == c],
                                                 [k for k in X.columns if theme_of(k) == t]].stack().mean()
                                           for t in THEME_ORDER] for c in comms}, index=THEME_ORDER)
    item_means = tests.head(top)[[f"mean_C{c + 1}" for c in comms]]
    item_means.columns = [names[c] for c in comms]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.6})
    for ax, M, title in [(axes[0], theme_means, "Mean score per theme"),
                         (axes[1], item_means, f"Top {top} most discriminating statements (Kruskal–Wallis η²)")]:
        ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(M.shape[1]), M.columns, rotation=20, ha="right", fontsize=8)
        ylabels = M.index if M is theme_means else [f"{i}  (η²={tests.loc[i, 'eta2']:.2f})" for i in M.index]
        ax.set_yticks(range(M.shape[0]), ylabels, fontsize=8, color=INK2)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.iat[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 1.25 else INK)
        ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-2, 2))
    fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.45, pad=0.2, aspect=40,
                 label="mean Likert score (−2 strongly disagree … +2 strongly agree)")
    _save(fig, out, "fig07_community_profiles.png")


def theme_ari_plot(M: pd.DataFrame, info: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.imshow(M.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(M)), M.columns, fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(len(M)), M.index, fontsize=8)
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M.iat[i, j] > 0.6 else INK)
    ax.set_title("ARI between theme-specific partitions", loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax = axes[1]
    y = np.arange(len(info))
    ax.barh(y, info["modularity"], color=[THEME_COLOR[t] for t in info["theme"]], height=0.6, edgecolor=SURFACE)
    for i, r in info.iterrows():
        ax.text(r["modularity"] + 0.005, i, f"Q={r['modularity']:.2f}, {int(r['communities'])} comm., "
                f"stab={r['stability_ari']:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, info["theme"], fontsize=8)
    ax.set_xlim(0, 0.75)
    ax.invert_yaxis()
    ax.set_title("Per-theme respondent networks", loc="left")
    ax.set_xlabel("modularity")
    fig.tight_layout()
    _save(fig, out, "fig08_theme_networks.png")


# ----------------------------------------------------------------------------- statement network
def item_network(G: nx.Graph, labels: dict, themes: dict, negatives: list, out: Path):
    connected = G.subgraph([n for n in G if G.degree(n) > 0])
    pos = nx.kamada_kawai_layout(connected, weight=None)
    pos = nx.spring_layout(connected, pos=pos, weight="weight", seed=3, k=0.9, iterations=60)
    isolated = sorted(n for n in G if G.degree(n) == 0)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    for i, n in enumerate(isolated):
        pos[n] = np.array([xs.min() + (i + 0.5) * (xs.max() - xs.min()) / len(isolated), ys.min() - 0.35])
    sizes_by_comm = pd.Series(labels).value_counts()
    big = [c for c in sizes_by_comm.index if sizes_by_comm[c] > 1]
    # colour each detected community by its dominant theme, so off-colour nodes are cross-theme members
    comm_color = {}
    for c in big:
        dominant = pd.Series([themes[n] for n in labels if labels[n] == c]).value_counts().index[0]
        comm_color[c] = THEME_COLOR[dominant] if THEME_COLOR[dominant] not in comm_color.values() else OTHER
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, mode in zip(axes, ["theme", "community"]):
        nx.draw_networkx_edges(G, pos, ax=ax, width=[0.2 + 2.5 * (w - 0.3) for _, _, w in G.edges(data="weight")],
                               edge_color="#c9c8c1", alpha=0.55)
        for n in G.nodes():
            col = THEME_COLOR[themes[n]] if mode == "theme" else comm_color.get(labels[n], OTHER)
            ax.scatter(*pos[n], s=300, marker=THEME_MARKER[themes[n]], color=col, edgecolor=SURFACE,
                       linewidth=1.2, zorder=3)
            ax.text(pos[n][0], pos[n][1], n, fontsize=5.5, ha="center", va="center", color="white",
                    fontweight="bold", zorder=4)
        ax.axhline(ys.min() - 0.2, color=GRID, lw=0.8)
        ax.text(xs.min(), ys.min() - 0.24, "isolated (no significant positive correlation with ρ ≥ 0.3)",
                fontsize=7.5, color=MUTED, va="top")
        ax.axis("off")
    axes[0].set_title("Statement network coloured by questionnaire theme", loc="left")
    axes[1].set_title("…coloured by Louvain community (colour of its majority theme)", loc="left")
    handles = [Line2D([], [], marker=THEME_MARKER[t], ls="", markersize=8, markerfacecolor=THEME_COLOR[t],
                      markeredgecolor=SURFACE, label=t) for t in THEME_ORDER]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig09_item_network.png")


def item_correlation_heatmap(R: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    M = R.to_numpy().copy()
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap=DIVERGING, vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(len(R)), R.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(R)), R.index, fontsize=5)
    for b in (15, 30, 45):
        ax.axhline(b - 0.5, color=INK2, lw=0.8)
        ax.axvline(b - 0.5, color=INK2, lw=0.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman ρ")
    ax.set_title("Statement–statement correlations (blocks: T, E, S, V)", loc="left")
    _save(fig, out, "fig10_item_correlations.png")
