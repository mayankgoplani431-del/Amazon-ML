"""Entity-level decision model (one row per S1).

Pair probabilities alone decide poorly *whether an S1 has any match* and *how many* it has.
Two small LightGBM models on per-S1 aggregates of the pair scores predict
  p_empty = P(S1 has no true match)       (binary)
  n_hat   = E[number of true matches]     (poisson)
and the expected-F0.5 subset rule uses them instead of prod(1-p) and sum(p).
Trained out-of-fold on train (same S1 folds as the pair model), fold-averaged on test.
"""
import os, pickle
import numpy as np
import polars as pl
import lightgbm as lgb
import pipeline as P
import decide as D

PAR = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=14, seed=0)


def aggregates(s: pl.DataFrame, col: str, all_qi: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 summary of pair scores, before and after exclusivity."""
    ex = D.exclusive(s, col).select('qi', 'pi', pl.lit(1, pl.Int8).alias('_ex'))
    s = s.join(ex, on=['qi', 'pi'], how='left').with_columns(pl.col('_ex').fill_null(0))
    s = s.with_columns(pl.col(col).rank('ordinal', descending=True).over('qi').alias('_r'))
    top = lambda k: pl.col(col).filter(pl.col('_r') == k).first().fill_null(0).alias(f'top{k}')
    a = s.group_by('qi').agg(
        pl.len().alias('n_cand'), top(1), top(2), top(3), top(4), top(5),
        pl.col(col).sum().alias('sum_p'),
        (pl.col(col) * pl.col('_ex')).sum().alias('sum_p_ex'),
        pl.col('_ex').sum().alias('n_ex'),
        (pl.col(col) > 0.5).sum().alias('n_05'), (pl.col(col) > 0.8).sum().alias('n_08'),
        (pl.col(col) > 0.95).sum().alias('n_095'),
        ((pl.col(col) > 0.5) & (pl.col('_ex') == 1)).sum().alias('n_05_ex'),
        (1 - pl.col(col)).clip(1e-6, 1).log().sum().alias('log_p0'),
        (1 - pl.col(col) * pl.col('_ex')).clip(1e-6, 1).log().sum().alias('log_p0_ex'),
        pl.col(col).filter(pl.col('_r') == 1).first().alias('_t1'),
        pl.col('_ex').filter(pl.col('_r') == 1).first().fill_null(0).alias('top1_survives_ex'),
    ).drop('_t1')
    return all_qi.join(a, on='qi', how='left').fill_null(0)


FEATS = ['n_cand', 'top1', 'top2', 'top3', 'top4', 'top5', 'sum_p', 'sum_p_ex', 'n_ex', 'n_05', 'n_08', 'n_095',
         'n_05_ex', 'log_p0', 'log_p0_ex', 'top1_survives_ex']


def fit_predict(agg: pl.DataFrame, ntrue: pl.DataFrame = None, name='entity', rounds=400):
    """Train (OOF by S1 fold) when ntrue given, else load models and fold-average. Returns qi, p_empty, n_hat."""
    path = os.path.join(P.MODELS, f'{name}.pkl')
    X = agg.select(FEATS).to_numpy()
    if ntrue is not None:
        d = agg.join(ntrue, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
        y_e = (d['ntrue'] == 0).to_numpy().astype(np.float32)
        y_n = d['ntrue'].to_numpy().astype(np.float32)
        fold = P.fold_of(pl.col('qi'))
        fo = agg.select(fold.alias('f'))['f'].to_numpy()
        pe, pn, models = np.zeros(len(X)), np.zeros(len(X)), []
        for k in range(P.NF):
            tr, va = fo != k, fo == k
            me = lgb.train({**PAR, 'objective': 'binary'}, lgb.Dataset(X[tr], y_e[tr]), rounds)
            mn = lgb.train({**PAR, 'objective': 'poisson'}, lgb.Dataset(X[tr], y_n[tr]), rounds)
            pe[va], pn[va] = me.predict(X[va]), mn.predict(X[va])
            models.append((me, mn))
        pickle.dump(models, open(path, 'wb'))
    else:
        models = pickle.load(open(path, 'rb'))
        pe = np.mean([me.predict(X) for me, _ in models], axis=0)
        pn = np.mean([mn.predict(X) for _, mn in models], axis=0)
    return agg.select('qi').with_columns(pl.Series('p_empty', pe.astype(np.float32)), pl.Series('n_hat', pn.astype(np.float32)))


def decide(s: pl.DataFrame, col: str, ent: pl.DataFrame, empty_scale=1.0, n_blend=1.0, exclusive=True):
    """Expected-F0.5 subset selection using entity-level p_empty / n_hat.
    E[F_k] = 1.25 * S_k / (k + 0.25 * N),  N = n_blend * n_hat + (1 - n_blend) * sum_p ;  E[F_0] = empty_scale * p_empty."""
    base = D.exclusive(s, col) if exclusive else s
    g = (base.join(ent, on='qi', how='left')
             .sort(['qi', col], descending=[False, True])
             .with_columns(pl.col(col).cum_sum().over('qi').alias('_cs'),
                           pl.int_range(1, pl.len() + 1).over('qi').alias('_k'),
                           pl.col(col).sum().over('qi').alias('_T')))
    g = g.with_columns((n_blend * pl.col('n_hat') + (1 - n_blend) * pl.col('_T')).clip(lower_bound=0.5).alias('_N'))
    g = g.with_columns((1.25 * pl.col('_cs') / (pl.col('_k') + 0.25 * pl.col('_N'))).alias('_ef'))
    g = g.with_columns(pl.col('_ef').max().over('qi').alias('_efmax'))
    ks = (g.filter(pl.col('_ef') == pl.col('_efmax')).group_by('qi')
            .agg(pl.col('_k').min().alias('_ks'), pl.col('_efmax').first().alias('_best'), pl.col('p_empty').first().alias('_pe')))
    g = g.join(ks, on='qi')
    keep = (pl.col('_k') <= pl.col('_ks')) & (pl.col('_best') > empty_scale * pl.col('_pe'))
    return g.filter(keep).drop([c for c in g.columns if c.startswith('_')] + ['p_empty', 'n_hat'])
