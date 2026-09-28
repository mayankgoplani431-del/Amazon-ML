"""Step 3: candidate pruning. A light LightGBM ranker over blocking scores + a few cheap string
similarities keeps the top-N candidates per S1. Its output is the FINAL candidate set
(candidate_pairs.tsv) that the matching model scores."""
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from candidates import CFG
PASS_COLS = list(CFG)   # one score/rank column pair per blocking pass


def cheap_features(c: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame) -> pl.DataFrame:
    qi, pi = c['qi'].to_numpy(), c['pi'].to_numpy()
    g = lambda df, col, idx: df[col].gather(idx).to_list()
    nq, nc = g(q, 'name_core', qi), g(p, 'name_core', pi)
    aq, ac = g(q, 'addr_norm', qi), g(p, 'addr_norm', pi)
    f1q, f1c = q['num1'].gather(qi), p['num1'].gather(pi)
    kw = dict(workers=-1, dtype=np.float32)
    feats = {
        'c_n_tset': cpdist(nq, nc, scorer=fuzz.token_set_ratio, **kw),
        'c_n_ratio': cpdist(nq, nc, scorer=fuzz.ratio, **kw),
        'c_a_tset': cpdist(aq, ac, scorer=fuzz.token_set_ratio, **kw),
        'c_a_ratio': cpdist(aq, ac, scorer=fuzz.ratio, **kw),
        'c_num1_eq': np.where((f1q != '').to_numpy() & (f1c != '').to_numpy(), (f1q == f1c).to_numpy(), np.nan).astype(np.float32),
        'c_a_empty': (p['addr_norm'].gather(pi) == '').to_numpy().astype(np.float32),
    }
    out = c.with_columns([pl.Series(k, v) for k, v in feats.items()])
    for n in PASS_COLS:
        out = out.with_columns(pl.col(f'{n}_score').fill_null(0.0), pl.col(f'{n}_rank').fill_null(-1))
    # rank of this pair within its S1 group by the combined name+addr score, and group size
    out = out.with_columns(
        pl.col('comb_score').rank('ordinal', descending=True).over('qi').alias('g_comb_rank'),
        pl.len().over('qi').alias('g_size'),
        (pl.col('c_n_tset') + pl.col('c_a_tset')).rank('ordinal', descending=True).over('qi').alias('g_sim_rank'),
    )
    return out


def feat_cols():
    return ([f'{n}_score' for n in PASS_COLS] + [f'{n}_rank' for n in PASS_COLS] +
            ['n_passes', 'c_n_tset', 'c_n_ratio', 'c_a_tset', 'c_a_ratio', 'c_num1_eq', 'c_a_empty',
             'g_comb_rank', 'g_size', 'g_sim_rank'])


PARAMS = dict(objective='binary', learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1, verbose=-1, num_threads=14, seed=0)


def train(df: pl.DataFrame, rounds=300):
    X = df.select(feat_cols()).to_numpy()
    return lgb.train(PARAMS, lgb.Dataset(X, df['label'].to_numpy()), rounds)


def predict(model, df):
    return model.predict(df.select(feat_cols()).to_numpy(), num_threads=14).astype(np.float32)


def prune(df: pl.DataFrame, score_col='s0', top_n=20):
    return df.filter(pl.col(score_col).rank('ordinal', descending=True).over('qi') <= top_n)
