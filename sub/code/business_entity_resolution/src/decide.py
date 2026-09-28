"""Entity-level decision logic + fast vectorized macro-F0.5 scorer.

Structural prior (verified on train GT): every S2/S3 record belongs to at most ONE S1 entity,
so each candidate is first assigned only to its best-scoring S1 ("exclusivity").
"""
import numpy as np
import polars as pl


def exclusive(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Keep, for every candidate pi, only its highest-scoring S1."""
    return df.filter(pl.col(col) == pl.col(col).max().over('pi')).unique(subset=['pi'], keep='first')


def threshold(df: pl.DataFrame, col: str, t: float, rel: float = 0.0) -> pl.DataFrame:
    """Predict pairs with score >= t and >= rel * best score of that S1."""
    return df.filter((pl.col(col) >= t) & (pl.col(col) >= rel * pl.col(col).max().over('qi')))


def expected_f(df: pl.DataFrame, col: str, miss: float = 0.0, empty_bias: float = 1.0) -> pl.DataFrame:
    """Per S1, choose the top-k set maximising the plug-in expected F0.5:
         E[F_k] = 1.25 * sum_{i<=k} p_i / (k + 0.25 * (sum_all p + miss)),   E[F_0] = empty_bias * prod(1 - p_i)
    """
    g = (df.sort(['qi', col], descending=[False, True])
           .with_columns(pl.col(col).cum_sum().over('qi').alias('_cs'),
                         pl.int_range(1, pl.len() + 1).over('qi').alias('_k'),
                         pl.col(col).sum().over('qi').alias('_T'),
                         (1 - pl.col(col)).log().sum().over('qi').exp().alias('_p0')))
    g = g.with_columns((1.25 * pl.col('_cs') / (pl.col('_k') + 0.25 * (pl.col('_T') + miss))).alias('_ef'))
    g = g.with_columns(pl.col('_ef').max().over('qi').alias('_efmax'))
    kstar = g.filter(pl.col('_ef') == pl.col('_efmax')).group_by('qi').agg(pl.col('_k').min().alias('_kstar'),
                                                                           pl.col('_efmax').first(), pl.col('_p0').first())
    g = g.join(kstar.select('qi', '_kstar', pl.col('_p0').alias('_p0b')), on='qi')
    keep = (pl.col('_k') <= pl.col('_kstar')) & (pl.col('_efmax') > empty_bias * pl.col('_p0b'))
    return g.filter(keep).drop([c for c in g.columns if c.startswith('_')])


def macro_f05(pred: pl.DataFrame, ntrue: pl.DataFrame, cand: pl.DataFrame = None) -> dict:
    """pred: (qi, pi, label) predicted pairs; ntrue: (qi, ntrue) for EVERY evaluated S1;
    cand: optional (qi, pi, label) final candidate set for blocking stats."""
    pg = pred.group_by('qi').agg(pl.col('label').sum().alias('tp'), pl.len().alias('np'))
    x = ntrue.join(pg, on='qi', how='left').with_columns(pl.col('tp').fill_null(0), pl.col('np').fill_null(0))
    x = x.with_columns(
        pl.when(pl.col('ntrue') == 0).then((pl.col('np') == 0).cast(pl.Float64))
          .when((pl.col('np') == 0) | (pl.col('tp') == 0)).then(0.0)
          .otherwise(1.25 * pl.col('tp') / (pl.col('np') + 0.25 * pl.col('ntrue'))).alias('f'))
    tp, npred, nt = x['tp'].sum(), x['np'].sum(), x['ntrue'].sum()
    sing = x.filter(pl.col('ntrue') == 0)
    sc = dict(n_s1=x.height, macro_f05=float(x['f'].mean()), pair_precision=tp / max(npred, 1),
              pair_recall=tp / max(nt, 1), singleton_acc=float((sing['np'] == 0).mean()) if sing.height else float('nan'),
              false_pos=int(npred - tp), false_neg=int(nt - tp))
    if cand is not None:
        cg = cand.group_by('qi').agg(pl.col('label').sum().alias('hit'), pl.len().alias('n'))
        y = ntrue.join(cg, on='qi', how='left').fill_null(0)
        sz = y['n'].to_numpy()
        sc.update(blocking_recall=y['hit'].sum() / max(nt, 1), avg_cands=float(sz.mean()),
                  p95_cands=float(np.percentile(sz, 95)), max_cands=int(sz.max()))
    return sc


if __name__ == '__main__':
    # README example + singleton convention, via the vectorized scorer
    pred = pl.DataFrame({'qi': [1, 1, 1, 3], 'pi': [10, 11, 12, 30], 'label': [1, 0, 1, 0]})
    nt = pl.DataFrame({'qi': [1, 2, 3], 'ntrue': [2, 0, 0]})
    sc = macro_f05(pred, nt)
    assert abs(sc['macro_f05'] - (0.7142857 + 1 + 0) / 3) < 1e-6, sc
    df = pl.DataFrame({'qi': [1, 1, 2, 2], 'pi': [10, 11, 10, 12], 'p': [0.9, 0.2, 0.5, 0.95]})
    ex = exclusive(df, 'p')
    assert sorted(ex['pi'].to_list()) == [10, 11, 12] and ex.filter(pl.col('pi') == 10)['qi'][0] == 1
    ef = expected_f(df, 'p')
    print('ok', sc['macro_f05'], ef.sort('qi').rows())
