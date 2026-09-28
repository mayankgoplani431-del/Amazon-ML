"""Experiment (c): soft meta-model over model probabilities vs hard-AND voting (S5 logic), on train OOF.

Base OOF probabilities (each fold scored by the model trained on the other fold):
  E02 stage1, S3 stage1+stage2  : re-scored from the legacy feature parts (work/cache/feat_train, legacy rows only)
  E06 stage1                    : work_e06/cache/oof_stage1.parquet
Hard-AND  = each model's own tuned decision rule, intersected (what S5 does on test).
Soft meta = LightGBM on [p02, p3, p06 (+ agreement stats)], trained on fold k, evaluated on fold 1-k; its expected-F0.5
            rule is tuned on the training fold only. No label of the evaluated fold is used anywhere.
Scored with plain macro F0.5 and density-stressed F0.5 (alpha 1.9: FP on ownerless records x1.9).
Usage (ER_WORK = legacy work dir):  python exp_meta.py <work_e06_dir> <out_json>
"""
import os, sys, json, pickle, time
import numpy as np
import polars as pl
import lightgbm as lgb
import pipeline as P, decide as D
from e05 import _leg_view, _oof_or_avg
from run_train import sweep, apply_rule

t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
ARCH = os.path.join(P.WORK, 'models_archive')


def legacy_oof():
    """Pass 1: E02 + S3 stage-1 OOF on legacy rows; pass 2: S3 stage-2 with S3 stage-1 group stats (streamed)."""
    m02, c02 = pickle.load(open(os.path.join(ARCH, 'E02', 'stage1.pkl'), 'rb'))
    m31, c31 = pickle.load(open(os.path.join(ARCH, 'S3', 'stage1.pkl'), 'rb'))
    ren = lambda c: c.drop([x for x in c.columns if x.startswith('s0_')] + ['s0']).rename(
        {'s0L': 's0', **{x: x[2:] for x in c.columns if x.startswith('L_s0_')}})
    out = []
    for c in P.iter_parts('train'):
        c = _leg_view(ren(c.filter(pl.col('in_leg') == 1)))
        fo = c['fold'].to_numpy()
        out.append(c.select('qi', 'pi', 'label', 'fold').with_columns(
            pl.Series('p02', _oof_or_avg(m02, c.select(c02).to_numpy(), fo, 'train')),
            pl.Series('p3s1', _oof_or_avg(m31, c.select(c31).to_numpy(), fo, 'train'))))
    s = pl.concat(out)
    say('legacy stage-1 OOF:', s.height)
    m32, c32 = pickle.load(open(os.path.join(ARCH, 'S3', 'stage2.pkl'), 'rb'))
    ctx = P.group_stats(s.select('qi', 'pi', pl.col('p3s1').alias('stage1')), 'stage1', 'p1')
    res = []
    for c in P.iter_parts('train'):
        c = _leg_view(ren(c.filter(pl.col('in_leg') == 1))).join(ctx, on=['qi', 'pi'], how='left')
        res.append(c.select('qi', 'pi').with_columns(pl.Series('p3', _oof_or_avg(m32, c.select(c32).to_numpy(), c['fold'].to_numpy(), 'train'))))
    say('S3 stage-2 OOF done')
    return s.join(pl.concat(res), on=['qi', 'pi'], how='left')


def main(w06, out_json):
    gt = pl.read_parquet(os.path.join(P.CACHE, 'truth_counts.parquet'))
    n_s1 = pl.read_parquet(os.path.join(P.CACHE, 'train_s1.parquet'), columns=['entity_id']).height
    ntrue = pl.DataFrame({'qi': pl.arange(0, n_s1, eager=True).cast(pl.UInt32)}).join(gt, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
    cache = os.path.join(P.CACHE, 'exp_meta_base.parquet')
    if os.path.exists(cache):
        base = pl.read_parquet(cache)
    else:
        leg = legacy_oof().with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
        e06 = pl.read_parquet(os.path.join(w06, 'cache', 'oof_stage1.parquet')).select(
            pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('label').alias('label06'), pl.col('stage1').alias('p06'))
        base = leg.join(e06, on=['qi', 'pi'], how='full', coalesce=True).with_columns(
            pl.coalesce('label', 'label06').cast(pl.Int8).alias('label'), P.fold_of(pl.col('qi')).alias('fold')).drop('label06')
        base.write_parquet(cache)
    say('base pairs (union of candidate sets):', base.height, '| positives', int(base['label'].sum()))
    owned = base.filter(pl.col('label') == 1)['pi'].unique()   # all owned pool records reachable here
    rules = {'p02': json.load(open(os.path.join(ARCH, 'E02', 'decision_stage1.json'))),
             'p3': json.load(open(os.path.join(ARCH, 'S3', 'decision.json'))),
             'p06': json.load(open(os.path.join(w06, 'models', 'decision_stage1.json')))}
    votes = {k: apply_rule(base.filter(pl.col(k).is_not_null()).select('qi', 'pi', 'label', k), k, r).select('qi', 'pi') for k, r in rules.items()}
    k = ['qi', 'pi']
    hard_and = votes['p02'].join(votes['p3'], on=k, how='semi').join(votes['p06'], on=k, how='semi')
    nv = pl.concat([v.with_columns(pl.lit(1).alias('v')) for v in votes.values()]).group_by(k).agg(pl.col('v').sum().alias('n_votes'))
    base = base.join(nv, on=k, how='left').with_columns(pl.col('n_votes').fill_null(0))
    feats = ['p02', 'p3', 'p06', 'n_votes']
    base = base.with_columns(pl.min_horizontal('p02', 'p3', 'p06').alias('pmin'), pl.mean_horizontal('p02', 'p3', 'p06').alias('pmean'))
    feats += ['pmin', 'pmean']
    par = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, verbose=-1, num_threads=14, seed=0)
    res = {'n_s1': n_s1}
    meta_pred = []
    for f in (0, 1):
        tr, ev = base.filter(pl.col('fold') != f), base.filter(pl.col('fold') == f)
        m = lgb.train(par, lgb.Dataset(tr.select(feats).to_numpy(), tr['label'].to_numpy()), 300)
        tr = tr.with_columns(pl.Series('pm', m.predict(tr.select(feats).to_numpy())))
        nt_tr = ntrue.filter(P.fold_of(pl.col('qi')) != f)
        smp = (pl.col('qi').hash(seed=5) % 10) < 3   # rule tuned on a 30% S1 sample of the training fold (memory: full sweep exhausted 16 GB)
        rule = sweep(tr.filter(smp).select('qi', 'pi', 'label', 'pm'), 'pm', nt_tr.filter(smp), None, None, owned=owned, alpha=1.9)[0][1]
        ev = ev.with_columns(pl.Series('pm', m.predict(ev.select(feats).to_numpy())))
        meta_pred.append(apply_rule(ev.select('qi', 'pi', 'label', 'pm'), 'pm', rule).select('qi', 'pi'))
        say(f'fold {f}: meta rule tuned on other fold = {rule}')
    meta = pl.concat(meta_pred)
    lab = base.select('qi', 'pi', 'label')
    def score(pred, tag):
        pr = pred.join(lab, on=k, how='left').with_columns(pl.col('label').fill_null(0))
        out = {}
        for f in (0, 1):
            nt = ntrue.filter(P.fold_of(pl.col('qi')) == f)
            pf = pr.filter(P.fold_of(pl.col('qi')) == f)
            a, b = D.macro_f05(pf, nt), D.macro_f05(pf, nt, owned=owned, alpha=1.9)
            out[f] = dict(plain=round(a['macro_f05'], 5), stressed=round(b['macro_f05'], 5), precision=round(a['pair_precision'], 5),
                          recall=round(a['pair_recall'], 5), fp=a['false_pos'], fn=a['false_neg'], pairs=pf.height)
        avg = {m: round((out[0][m] + out[1][m]) / 2, 5) for m in ('plain', 'stressed', 'precision', 'recall')}
        say(f'{tag:10s} {avg}  folds {out}')
        res[tag] = dict(avg=avg, folds=out)
    for kk, v in votes.items():
        score(v, kk + '_alone')
    score(hard_and, 'hard_AND')
    score(meta, 'soft_meta')
    # disputed pairs: at least one but not all three models vote yes
    disp = base.filter((pl.col('n_votes') >= 1) & (pl.col('n_votes') <= 2)).select('qi', 'pi', 'label', 'n_votes')
    in_meta = disp.join(meta, on=k, how='semi')
    res['disputed'] = dict(n=disp.height, true_share=round(float(disp['label'].mean()), 4),
                           meta_keeps=in_meta.height, meta_keep_precision=round(float(in_meta['label'].mean()) if in_meta.height else 0, 4),
                           hard_and_keeps=0, by_votes={int(r['n_votes']): dict(n=r['n'], true_share=round(r['t'], 4))
                                                       for r in disp.group_by('n_votes').agg(pl.len().alias('n'), pl.col('label').mean().alias('t')).iter_rows(named=True)})
    say('disputed pairs:', res['disputed'])
    json.dump(res, open(out_json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
