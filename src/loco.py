"""Leave-one-country-out proxy for the unseen test country (France):
train the matcher on one country only, evaluate macro F0.5 on the other, compare with in-domain OOF.
Usage: python loco.py <exp_name> <train_country> <eval_country> [drop_feature,...]
"""
import sys, os, time
import numpy as np
import polars as pl
import lightgbm as lgb
import pipeline as P, decide as D
from common import load_norm
from run_train import sweep, apply_rule
from evaluate import log_experiment


def main(exp, tr_c, ev_c, drop=()):
    t = time.time()
    ctry = load_norm('train', 1).select(pl.int_range(pl.len()).cast(pl.UInt32).alias('qi'), 'country_n')
    tr, ev = [], []
    for c in P.iter_parts('train'):
        c = c.with_columns(pl.col('qi').cast(pl.UInt32)).join(ctry, on='qi')
        tr.append(c.filter((pl.col('country_n') == tr_c) & ((pl.col('qi').hash(seed=11) % 1000) < 300)).drop('country_n'))
        ev.append(c.filter(pl.col('country_n') == ev_c).drop('country_n'))
    tr, ev = pl.concat(tr), pl.concat(ev)
    cols = P.fcols(tr, drop)
    params = dict(P.P1)
    if os.environ.get('RANKNORM') == '1':
        tr = P.country_rank_norm(tr.with_columns(pl.lit('a').alias('country_n')), cols).drop('country_n')
        ev = P.country_rank_norm(ev.with_columns(pl.lit('b').alias('country_n')), cols).drop('country_n')
    if os.environ.get('MONO') == '1':
        params['monotone_constraints'] = P.monotone(cols)
        params['monotone_constraints_method'] = 'advanced'
    m = lgb.train(params, lgb.Dataset(tr.select(cols).to_numpy(), tr['label'].to_numpy(), feature_name=cols), 1000)
    s = ev.select('qi', 'pi', 'label').with_columns(pl.Series('p', m.predict(ev.select(cols).to_numpy(), num_threads=14)))
    ev_q = ctry.filter(pl.col('country_n') == ev_c).select('qi')
    # ground truth count for every evaluated S1 (incl. pairs lost in blocking)
    tc = os.path.join(P.CACHE, 'truth_counts.parquet')
    if not os.path.exists(tc):
        import candidates
        q, p = candidates.load_split('train', ['entity_id'])
        P.labels(q, p).group_by('qi').agg(pl.len().alias('ntrue')).with_columns(pl.col('qi').cast(pl.UInt32)).write_parquet(tc)
        del q, p
    gt = pl.read_parquet(tc)
    ntrue = ev_q.join(gt, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
    best, _ = sweep(s, 'p', ntrue, None, None)
    sc = D.macro_f05(apply_rule(s, 'p', best[1]), ntrue)
    imp = sorted(zip(m.feature_importance('gain'), cols), reverse=True)[:15]
    txt = log_experiment(exp, sc, dict(train=tr_c, eval=ev_c, drop=list(drop), decision=best[1], ranknorm=os.environ.get('RANKNORM'), mono=os.environ.get('MONO')),
                         notes='LOCO proxy for unseen country; top gain features: ' + ', '.join(c for _, c in imp))
    print(txt, f'\n{time.time() - t:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], tuple(x for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else []) if x))
