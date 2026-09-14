"""Similarity measures and graph construction for respondent and statement networks."""
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

MIN_OVERLAP = 30  # minimum co-answered statements for a respondent pair to be compared
DEFAULT_K = 8     # neighbours per respondent in the kNN backbone (see k-sweep in analysis)


def respondent_similarity(X: pd.DataFrame, centre: str = "item", min_overlap: int = MIN_OVERLAP) -> pd.DataFrame:
    """Pairwise-complete cosine similarity between respondents.

    centre="item"   subtract each statement's class mean, so two people are similar when they
                    deviate from the class consensus in the same direction (main method).
    centre="person" additionally remove each respondent's own mean (removes acquiescence bias).
    centre="none"   raw Likert vectors (sensitivity check only; dominated by general agreement).
    """
    Z = X.copy()
    if centre in ("item", "person"):
        Z = Z - Z.mean(axis=0)
    if centre == "person":
        Z = Z.sub(Z.mean(axis=1), axis=0)

    M = Z.notna().to_numpy(float)
    Z0 = Z.fillna(0).to_numpy()
    dot = Z0 @ Z0.T
    sq = Z0 ** 2
    norm_i = sq @ M.T          # |z_i|^2 restricted to the items j also answered
    norm_j = M @ sq.T
    with np.errstate(divide="ignore", invalid="ignore"):
        S = dot / np.sqrt(norm_i * norm_j)
    overlap = M @ M.T
    S[(overlap < min_overlap) | ~np.isfinite(S)] = np.nan
    np.fill_diagonal(S, np.nan)
    return pd.DataFrame(S, index=X.index, columns=X.index)


def knn_graph(S: pd.DataFrame, k: int = DEFAULT_K) -> nx.Graph:
    """Symmetric k-nearest-neighbour graph: keep i-j if j is in i's top-k or vice versa.

    Only positive similarities become edges; weight = cosine similarity.
    """
    G = nx.Graph()
    G.add_nodes_from(S.index)
    values = S.to_numpy()
    nodes = S.index.to_numpy()
    for a, row in enumerate(values):
        valid = np.where(np.isfinite(row) & (row > 0))[0]
        top = valid[np.argsort(row[valid])[::-1][:k]]
        for b in top:
            G.add_edge(nodes[a], nodes[b], weight=float(row[b]))
    return G


def threshold_graph(S: pd.DataFrame, quantile: float = 0.90) -> nx.Graph:
    """Alternative construction: keep the strongest (1 - quantile) share of all pairs."""
    tri = S.to_numpy()[np.triu_indices(len(S), 1)]
    cut = np.nanquantile(tri, quantile)
    G = nx.Graph()
    G.add_nodes_from(S.index)
    for i, a in enumerate(S.index):
        for j in range(i + 1, len(S)):
            w = S.iat[i, j]
            if np.isfinite(w) and w >= cut:
                G.add_edge(a, S.index[j], weight=float(w))
    return G


def item_correlations(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation between statements (pairwise complete) and two-sided p-values."""
    cols = X.columns
    n = len(cols)
    R = np.eye(n)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair = X[[cols[i], cols[j]]].dropna()
            rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            if not np.isfinite(rho):
                rho, p = 0.0, 1.0
            R[i, j] = R[j, i] = rho
            P[i, j] = P[j, i] = p
    return pd.DataFrame(R, cols, cols), pd.DataFrame(P, cols, cols)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def item_graph(R: pd.DataFrame, P: pd.DataFrame, alpha: float = 0.05, min_rho: float = 0.30):
    """Statement network: positive, FDR-significant correlations with rho >= min_rho.

    Returns the positive graph and the list of significant negative pairs (reported separately).
    """
    cols = R.columns
    iu = np.triu_indices(len(cols), 1)
    q = fdr_bh(P.to_numpy()[iu])
    G = nx.Graph()
    G.add_nodes_from(cols)
    negatives = []
    for (i, j), qv in zip(zip(*iu), q):
        rho = R.iat[i, j]
        if qv >= alpha:
            continue
        if rho >= min_rho:
            G.add_edge(cols[i], cols[j], weight=float(rho))
        elif rho <= -min_rho:
            negatives.append((cols[i], cols[j], float(rho)))
    return G, negatives
