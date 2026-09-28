"""Multi-pass candidate generation. Every pass runs independently per country label
(open set: whatever labels appear in the query file) and the results are unioned.

A pass returns a polars frame (qi, pi, <pass>_score, <pass>_rank) with qi/pi = row indices
into the query (S1) and pool (S2+S3 concatenated) frames.
"""
import time
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

N_THREADS = 14


def _topk(A, BT, K):
    """Top-K of A @ BT per row of A; BT is the pre-transposed (features x n) CSR pool matrix."""
    C = sp_matmul_topn(A, BT, top_n=K, n_threads=N_THREADS, sort=True).tocsr()
    rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr))
    # rank within row (C is sorted descending per row)
    rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], np.diff(C.indptr))
    return rows, C.indices.astype(np.int64), C.data.astype(np.float32), rank.astype(np.int16)


def text_of(df, kind):
    if kind == 'comb':
        return (df['name_core'] + ' ' + df['addr_norm']).to_list()
    if kind == 'name':
        return df['name_core'].to_list()
    if kind == 'name_nospace':  # catches domain-style names: allpropertysolutions.com
        return df['name_core'].str.replace_all(' ', '').to_list()
    if kind == 'addr':
        return df['addr_norm'].to_list()
    if kind == 'nameaddr':  # whole core name as ONE token: exact-name agreement stays rare/high-IDF
        return ('N_' + df['name_core'].str.replace_all(' ', '_') + ' ' + df['addr_norm']).to_list()
    raise ValueError(kind)


PASSES = {
    # name: (text kind, vectorizer kwargs, K, reverse)
    'comb': ('comb', dict(token_pattern=r'\S+', max_df=20000), 100, False),
    'addr': ('addr', dict(token_pattern=r'\S+', max_df=20000), 50, False),
    'name': ('name', dict(token_pattern=r'\S+', max_df=20000), 50, False),
    'rev_comb': ('comb', dict(token_pattern=r'\S+', max_df=20000), 5, True),
    'name_ch': ('name_nospace', dict(analyzer='char', ngram_range=(4, 4), max_df=20000, min_df=2), 30, False),
}


def run_pass(name, q, p, spec=None, verbose=True):
    kind, vkw, K, reverse = spec or PASSES[name]
    out = []
    for ctry in q['country_n'].unique().sort().to_list():
        t = time.time()
        qm = q.filter(pl.col('country_n') == ctry).select('idx_q', 'name_core', 'addr_norm')
        pm = p.filter(pl.col('country_n') == ctry).select('idx_p', 'name_core', 'addr_norm')
        if qm.height == 0 or pm.height == 0:
            continue
        v = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **vkw)
        # fit IDF on the pool (+queries) of this country only; country itself is never a feature
        v.fit(text_of(pm, kind) + text_of(qm, kind))
        A, B = v.transform(text_of(qm, kind)), v.transform(text_of(pm, kind))
        if reverse:   # pool records query the S1 side: "which S1 does this candidate look like?"
            r, c, s, k = _topk(B, A.T.tocsr(), K)
            r, c = c, r
        else:
            r, c, s, k = _topk(A, B.T.tocsr(), K)
        out.append(pl.DataFrame({'qi': qm['idx_q'].to_numpy()[r], 'pi': pm['idx_p'].to_numpy()[c],
                                 f'{name}_score': s, f'{name}_rank': k}))
        if verbose:
            print(f'  pass {name} [{ctry}] q={qm.height} p={pm.height} pairs={len(r)} {time.time() - t:.0f}s', flush=True)
    return pl.concat(out)


def generate(q, p, passes, verbose=True):
    """q, p: normalized frames. Adds idx_q/idx_p. Returns unioned candidate frame with per-pass cols."""
    q = q.with_row_index('idx_q'); p = p.with_row_index('idx_p')
    res = None
    for name in passes:
        f = run_pass(name, q, p, verbose=verbose)
        res = f if res is None else res.join(f, on=['qi', 'pi'], how='full', coalesce=True)
    return res
