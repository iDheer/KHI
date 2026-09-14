"""End-to-end pipeline: raw survey CSV -> cleaned data -> networks -> metrics, tables and figures.

Usage:  python run_pipeline.py
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src import analysis as A
from src import visualize as V
from src.network import (DEFAULT_K, item_correlations, item_graph, knn_graph, respondent_similarity,
                         threshold_graph)
from src.preprocess import THEMES, prepare, theme_of

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Survey_Results_UC.csv"
OUT = ROOT / "outputs"
FIG, TAB = OUT / "figures", OUT / "tables"
K = DEFAULT_K


def name_communities(X: pd.DataFrame, labels: pd.Series) -> list:
    """Give the respondent communities descriptive names based on their opinion profile."""
    comms = sorted(labels.unique())
    if len(comms) != 3:
        return [f"Community {c + 1}" for c in comms]
    means = {c: X.loc[labels == c] for c in comms}
    moderate = min(comms, key=lambda c: means[c].stack().mean())
    rest = [c for c in comms if c != moderate]
    regulator = max(rest, key=lambda c: means[c][["T12", "T13"]].stack().mean())
    names = {moderate: "Moderate Pragmatists", regulator: "Tech-Cautious Regulators"}
    names[next(c for c in rest if c != regulator)] = "Progressive Idealists"
    return [names[c] for c in comms]


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main():
    for d in (FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    metrics = {}

    # 1. Data preparation --------------------------------------------------------------------
    raw, X_all, X, quality, statements = prepare(DATA)
    quality.to_csv(TAB / "respondent_quality.csv")
    X.to_csv(OUT / "cleaned_likert_matrix.csv")
    items = A.item_summary(X, statements, theme_of)
    items.to_csv(TAB / "statement_summary.csv")
    metrics["data"] = {
        "raw_rows": len(raw), "statements": raw.shape[1],
        "id_range": [int(raw.index.min()), int(raw.index.max())],
        "empty_rows": int((quality["answered"] == 0).sum()),
        "dropped_partial_rows": int(((quality["answered"] > 0) & ~quality["kept"]).sum()),
        "retained": len(X), "cells_missing_in_retained": int(X.isna().sum().sum()),
        "no_comments_in_retained": int(quality.loc[quality["kept"], "no_comments"].sum()),
        "overall_mean": float(X.stack().mean()),
        "share_agree": float((X.stack() > 0).mean()),
        "share_strongly_agree": float((X.stack() == 2).mean()),
        "share_disagree": float((X.stack() < 0).mean()),
        "theme_means": X.T.groupby(theme_of).apply(lambda d: d.stack().mean()).to_dict(),
    }
    print(f"[data] {len(raw)} rows -> {len(X)} respondents x {X.shape[1]} statements")

    # 2. Respondent network ------------------------------------------------------------------
    S = respondent_similarity(X, centre="item")
    sweep = A.k_sweep(S)
    sweep.to_csv(TAB / "k_sweep.csv", index=False)
    G = knn_graph(S, K)
    nx.write_graphml(G, OUT / "respondent_network.graphml")
    metrics["respondent_network"] = A.global_metrics(G)
    tri = S.to_numpy()[np.triu_indices(len(S), 1)]
    metrics["similarity"] = {"mean": float(np.nanmean(tri)), "sd": float(np.nanstd(tri)),
                             "p05": float(np.nanpercentile(tri, 5)), "p95": float(np.nanpercentile(tri, 95)),
                             "edge_weight_mean": float(np.mean([w for *_, w in G.edges(data="weight")]))}

    labels, comm_info = A.louvain(G, runs=100)
    L = pd.Series(labels).loc[X.index]
    names = name_communities(X, L)
    comm_info["names"] = names
    metrics["communities"] = comm_info
    metrics["resolution_sweep"] = A.resolution_sweep(G).to_dict(orient="records")

    cent = A.node_centralities(G)
    cent["community"] = [names[labels[n]] for n in cent.index]
    cent["mean_score"] = X.mean(axis=1)
    cent.sort_values("betweenness", ascending=False).to_csv(TAB / "respondent_centrality.csv")
    metrics["centrality_corr"] = {
        "degree_vs_mean_score_spearman": float(cent["degree"].corr(cent["mean_score"], method="spearman")),
        "betweenness_vs_mean_score_spearman": float(cent["betweenness"].corr(cent["mean_score"], method="spearman")),
    }
    tests = A.community_item_tests(X, L)
    tests.to_csv(TAB / "community_statement_tests.csv")
    prof = X.groupby(L).agg(["mean"]).T.droplevel(1).T
    prof.index = [names[c] for c in prof.index]
    prof.T.to_csv(TAB / "community_statement_means.csv")
    metrics["community_profiles"] = {
        names[c]: {"size": int((L == c).sum()),
                   "mean_score": float(X.loc[L == c].stack().mean()),
                   "share_strongly_agree": float((X.loc[L == c].stack() == 2).mean()),
                   "share_disagree": float((X.loc[L == c].stack() < 0).mean()),
                   "theme_means": X.loc[L == c].T.groupby(theme_of).apply(lambda d: d.stack().mean()).to_dict(),
                   "intra_edge_share": float(np.mean([labels[u] == labels[v] for u, v in G.edges()
                                                      if labels[u] == c or labels[v] == c]))}
        for c in sorted(L.unique())}
    metrics["significant_statements_q05"] = int((tests["q"] < 0.05).sum())

    # 3. Validation: null models and robustness ----------------------------------------------
    null_q = A.permutation_null(X, K, reps=200)
    metrics["permutation_null"] = {"mean": float(null_q.mean()), "sd": float(null_q.std()),
                                   "max": float(null_q.max()),
                                   "z": float((comm_info["modularity"] - null_q.mean()) / null_q.std()),
                                   "p_empirical": float((np.sum(null_q >= comm_info["modularity"]) + 1) / (len(null_q) + 1))}
    sw = A.small_world(G)
    metrics["small_world"] = sw
    variants = {
        "kNN k=6": knn_graph(S, 6),
        "kNN k=10": knn_graph(S, 10),
        "Top-10% threshold": threshold_graph(S, 0.90),
        "Person-centred cosine (k=8)": knn_graph(respondent_similarity(X, "person"), K),
        "Raw cosine, no centring (k=8)": knn_graph(respondent_similarity(X, "none"), K),
    }
    robust = A.construction_robustness(X, labels, variants)
    robust.to_csv(TAB / "construction_robustness.csv", index=False)
    metrics["robustness"] = robust.to_dict(orient="records")

    parts, theme_info, theme_ari = A.theme_partitions(X, THEMES, K)
    theme_info.to_csv(TAB / "theme_networks.csv", index=False)
    theme_ari.to_csv(TAB / "theme_partition_ari.csv")
    metrics["theme_networks"] = theme_info.to_dict(orient="records")
    metrics["theme_partition_ari"] = theme_ari.to_dict()

    # 4. Statement network -------------------------------------------------------------------
    R, P = item_correlations(X)
    R.to_csv(TAB / "statement_spearman.csv")
    IG, negatives = item_graph(R, P)
    nx.write_graphml(IG, OUT / "statement_network.graphml")
    ilabels, iinfo = A.louvain(IG, runs=100)
    nodes = list(IG.nodes())
    theme_codes = [n[0] for n in nodes]
    within = [R.iat[i, j] for i in range(60) for j in range(i + 1, 60) if R.columns[i][0] == R.columns[j][0]]
    between = [R.iat[i, j] for i in range(60) for j in range(i + 1, 60) if R.columns[i][0] != R.columns[j][0]]
    istrength = pd.Series(dict(IG.degree(weight="weight"))).sort_values(ascending=False)
    ibetween = pd.Series(nx.betweenness_centrality(IG))
    item_nodes = pd.DataFrame({"theme": [theme_of(n) for n in nodes], "community": [ilabels[n] for n in nodes],
                               "degree": dict(IG.degree()), "strength": istrength, "betweenness": ibetween,
                               "statement": [statements[n] for n in nodes]}, index=nodes)
    item_nodes.sort_values("strength", ascending=False).to_csv(TAB / "statement_network_nodes.csv")
    metrics["statement_network"] = {
        **{k: v for k, v in A.global_metrics(IG.subgraph([n for n in IG if IG.degree(n) > 0])).items()},
        "isolated_statements": [n for n in IG if IG.degree(n) == 0],
        "significant_negative_pairs": negatives,
        "communities": {str(c): sorted(n for n in nodes if ilabels[n] == c) for c in set(ilabels.values())},
        "modularity": iinfo["modularity"], "stability_ari": iinfo["stability_mean_ari"],
        "NMI_vs_themes": normalized_mutual_info_score(theme_codes, [ilabels[n] for n in nodes]),
        "ARI_vs_themes": adjusted_rand_score(theme_codes, [ilabels[n] for n in nodes]),
        "mean_rho_within_theme": float(np.mean(within)), "mean_rho_between_themes": float(np.mean(between)),
        "top_strength": istrength.head(8).round(2).to_dict(),
        "top_betweenness": ibetween.sort_values(ascending=False).head(6).round(3).to_dict(),
    }

    # 5. Figures -----------------------------------------------------------------------------
    V.response_distribution(raw, X, FIG)
    V.consensus_plot(items, FIG)
    V.construction_diagnostics(S, sweep, K, FIG)
    V.respondent_network(G, labels, cent, names, FIG)
    V.centrality_plots(G, cent, labels, X, names, FIG)
    V.null_model_plot(null_q, comm_info["modularity"], sw, FIG)
    V.community_profiles(X, L, tests, names, theme_of, FIG)
    V.theme_ari_plot(theme_ari, theme_info, FIG)
    V.item_network(IG, ilabels, {n: theme_of(n) for n in nodes}, negatives, FIG)
    V.item_correlation_heatmap(R, FIG)

    (OUT / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2))
    print(json.dumps(to_jsonable(metrics), indent=2))


if __name__ == "__main__":
    main()
