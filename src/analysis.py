"""Network metrics, community detection, robustness checks and opinion profiling."""
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .network import fdr_bh, knn_graph, respondent_similarity

SEED = 42


# ----------------------------------------------------------------------------- metrics
def global_metrics(G: nx.Graph) -> dict:
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    degrees = np.array([d for _, d in G.degree()])
    strength = np.array([d for _, d in G.degree(weight="weight")])
    dist = nx.Graph((u, v, {"d": 1 - w}) for u, v, w in giant.edges(data="weight"))
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "giant_component_size": giant.number_of_nodes(),
        "mean_degree": degrees.mean(),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(degrees.max()),
        "mean_strength": strength.mean(),
        "avg_clustering": nx.average_clustering(G),
        "avg_weighted_clustering": nx.average_clustering(G, weight="weight"),
        "transitivity": nx.transitivity(G),
        "avg_shortest_path": nx.average_shortest_path_length(giant),
        "diameter": nx.diameter(giant),
        "avg_weighted_path_(1-sim)": nx.average_shortest_path_length(dist, weight="d"),
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
    }


def node_centralities(G: nx.Graph) -> pd.DataFrame:
    dist = {(u, v): 1 - w for u, v, w in G.edges(data="weight")}
    nx.set_edge_attributes(G, dist, "distance")
    df = pd.DataFrame({
        "degree": dict(G.degree()),
        "strength": dict(G.degree(weight="weight")),
        "clustering": nx.clustering(G, weight="weight"),
        "betweenness": nx.betweenness_centrality(G, weight="distance", seed=SEED),
        "closeness": nx.closeness_centrality(G, distance="distance"),
        "eigenvector": nx.eigenvector_centrality_numpy(G, weight="weight"),
    })
    df.index.name = "node"
    return df


def small_world(G: nx.Graph, reps: int = 50) -> dict:
    """Compare clustering / path length with degree-preserving rewired graphs."""
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    H0 = nx.Graph(giant.edges())
    C, L = nx.transitivity(H0), nx.average_shortest_path_length(H0)
    rng = np.random.default_rng(SEED)
    Cs, Ls = [], []
    for _ in range(reps):
        H = H0.copy()
        nx.double_edge_swap(H, nswap=10 * H.number_of_edges(), max_tries=10 ** 6,
                            seed=int(rng.integers(1e9)))
        if not nx.is_connected(H):
            H = H.subgraph(max(nx.connected_components(H), key=len))
        Cs.append(nx.transitivity(H))
        Ls.append(nx.average_shortest_path_length(H))
    Cr, Lr = np.mean(Cs), np.mean(Ls)
    return {"C": C, "L": L, "C_rand": Cr, "L_rand": Lr, "C_rand_sd": np.std(Cs),
            "L_rand_sd": np.std(Ls), "sigma": (C / Cr) / (L / Lr)}


# ----------------------------------------------------------------------------- communities
def louvain(G: nx.Graph, runs: int = 100, resolution: float = 1.0):
    """Run Louvain with many seeds; keep the best-modularity partition and report stability."""
    partitions = []
    for s in range(runs):
        comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=s)
        partitions.append((nx.community.modularity(G, comms, weight="weight"), comms))
    best_q, best = max(partitions, key=lambda t: t[0])
    best = sorted(best, key=len, reverse=True)
    labels = {n: c for c, members in enumerate(best) for n in members}
    nodes = list(G.nodes())
    ref = [labels[n] for n in nodes]
    aris = []
    for _, comms in partitions:
        lab = {n: c for c, m in enumerate(comms) for n in m}
        aris.append(adjusted_rand_score(ref, [lab[n] for n in nodes]))
    return labels, {"modularity": best_q, "n_communities": len(best),
                    "sizes": [len(c) for c in best], "stability_mean_ari": float(np.mean(aris)),
                    "modularity_range": (min(p[0] for p in partitions), best_q)}


def labels_to_list(labels: dict, order) -> list:
    return [labels[n] for n in order]


def k_sweep(S: pd.DataFrame, ks=range(2, 16)) -> pd.DataFrame:
    rows = []
    for k in ks:
        G = knn_graph(S, k)
        lab, info = louvain(G, runs=20)
        rows.append({"k": k, "edges": G.number_of_edges(), "components": nx.number_connected_components(G),
                     "density": nx.density(G), "modularity": info["modularity"],
                     "communities": info["n_communities"], "stability_ari": info["stability_mean_ari"]})
    return pd.DataFrame(rows)


def resolution_sweep(G: nx.Graph, gammas=(0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0)) -> pd.DataFrame:
    rows = []
    for g in gammas:
        _, info = louvain(G, runs=30, resolution=g)
        rows.append({"resolution": g, "communities": info["n_communities"], "modularity": info["modularity"],
                     "stability_ari": info["stability_mean_ari"], "sizes": info["sizes"]})
    return pd.DataFrame(rows)


def permutation_null(X: pd.DataFrame, k: int, reps: int = 200) -> np.ndarray:
    """Modularity obtained when each statement's answers are shuffled across respondents.

    Keeps every statement's answer distribution but destroys any link between a person's answers.
    """
    rng = np.random.default_rng(SEED)
    qs = []
    for _ in range(reps):
        Xp = X.apply(lambda c: pd.Series(rng.permutation(c.to_numpy()), index=c.index))
        G = knn_graph(respondent_similarity(Xp), k)
        _, info = louvain(G, runs=5)
        qs.append(info["modularity"])
    return np.array(qs)


def construction_robustness(X: pd.DataFrame, reference: dict, graphs: dict) -> pd.DataFrame:
    """ARI between the main partition and partitions obtained from alternative constructions."""
    nodes = list(reference)
    rows = []
    for name, G in graphs.items():
        lab, info = louvain(G, runs=50)
        common = [n for n in nodes if n in lab]
        rows.append({"variant": name, "edges": G.number_of_edges(), "communities": info["n_communities"],
                     "modularity": info["modularity"],
                     "ARI_vs_main": adjusted_rand_score([reference[n] for n in common], [lab[n] for n in common]),
                     "NMI_vs_main": normalized_mutual_info_score([reference[n] for n in common],
                                                                 [lab[n] for n in common])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- opinion profiling
def tastle_wierman(col: pd.Series) -> float:
    """Tastle & Wierman (2007) ordinal consensus in [0, 1]; 1 = everybody gives the same answer."""
    x = col.dropna()
    if x.empty:
        return np.nan
    p = x.value_counts(normalize=True)
    mu = (p.index.to_numpy() * p.to_numpy()).sum()
    width = 4.0
    return 1 + float((p.to_numpy() * np.log2(1 - np.abs(p.index.to_numpy() - mu) / width)).sum())


def item_summary(X: pd.DataFrame, statements: dict, theme_of) -> pd.DataFrame:
    df = pd.DataFrame(index=X.columns)
    df["theme"] = [theme_of(c) for c in X.columns]
    df["statement"] = [statements[c] for c in X.columns]
    df["n"] = X.notna().sum()
    df["mean"] = X.mean()
    df["sd"] = X.std()
    df["pct_agree"] = (X > 0).sum() / X.notna().sum() * 100
    df["pct_disagree"] = (X < 0).sum() / X.notna().sum() * 100
    df["consensus_TW"] = X.apply(tastle_wierman)
    return df


def community_item_tests(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Kruskal-Wallis test per statement across communities, with eta-squared and FDR q."""
    rows = []
    k = labels.nunique()
    for c in X.columns:
        groups = [X.loc[labels.index[labels == g], c].dropna() for g in sorted(labels.unique())]
        n = sum(len(g) for g in groups)
        H, p = stats.kruskal(*groups)
        rows.append({"item": c, "H": H, "p": p, "eta2": max((H - k + 1) / (n - k), 0),
                     **{f"mean_C{g + 1}": grp.mean() for g, grp in enumerate(groups)}})
    out = pd.DataFrame(rows).set_index("item")
    out["q"] = fdr_bh(out["p"].to_numpy())
    return out.sort_values("eta2", ascending=False)


def theme_partitions(X: pd.DataFrame, theme_codes: dict, k: int) -> tuple[dict, pd.DataFrame]:
    """Build a respondent network per theme and compare the resulting communities."""
    parts, info_rows = {}, []
    for code, name in theme_codes.items():
        cols = [c for c in X.columns if c.startswith(code)]
        Xt = X[cols].dropna(thresh=12)
        G = knn_graph(respondent_similarity(Xt, min_overlap=12), k)
        lab, info = louvain(G, runs=50)
        parts[name] = lab
        info_rows.append({"theme": name, "respondents": len(Xt), "edges": G.number_of_edges(),
                          "modularity": info["modularity"], "communities": info["n_communities"],
                          "stability_ari": info["stability_mean_ari"]})
    names = list(parts)
    M = pd.DataFrame(np.eye(len(names)), names, names)
    for a, b in itertools.combinations(names, 2):
        common = sorted(set(parts[a]) & set(parts[b]))
        M.loc[a, b] = M.loc[b, a] = adjusted_rand_score([parts[a][n] for n in common],
                                                        [parts[b][n] for n in common])
    return parts, pd.DataFrame(info_rows), M
