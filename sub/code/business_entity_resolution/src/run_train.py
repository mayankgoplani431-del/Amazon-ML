"""Train + validate on the training split with 2-fold (grouped by S1) out-of-fold predictions.

Every experiment prints/saves the scorecard (work/experiments/scorecards.md, log.jsonl).
Usage: python run_train.py <experiment_id> [top_n]
"""
import sys, os, json, time
import numpy as np
import polars as pl
import candidates, pipeline as P, decide as D
from evaluate import log_experiment
from common import WORK


def s1_truth(q, lab):
    return (q.select(pl.col('idx_q').alias('qi'))
             .join(lab.group_by('qi').agg(pl.len().alias('ntrue')), on='qi', how='left')
             .with_columns(pl.col('ntrue').fill_null(0)))


def sweep(scores, col, ntrue, cand, folds):
    """Decision-rule search on OOF scores. Returns best config + per-fold scorecards."""
    res = []
    ex = D.exclusive(scores, col)
    for use_ex, base in [(False, scores), (True, ex)]:
        for t in np.arange(0.05, 0.96, 0.05):
            pred = D.threshold(base, col, float(t))
            res.append((D.macro_f05(pred, ntrue)['macro_f05'], dict(rule='threshold', t=round(float(t), 2), exclusive=use_ex)))
    best = max(res, key=lambda r: r[0])
    # refine around the best threshold, then try relative-to-best and expected-F rules on top
    base = ex if best[1]['exclusive'] else scores
    for t in np.arange(best[1]['t'] - 0.05, best[1]['t'] + 0.051, 0.01):
        for rel in (0.0, 0.3, 0.5, 0.7):
            pred = D.threshold(base, col, float(t), rel)
            res.append((D.macro_f05(pred, ntrue)['macro_f05'], dict(rule='threshold', t=round(float(t), 2), rel=rel, exclusive=best[1]['exclusive'])))
    for miss in (0.0, 0.2, 0.5):
        for eb in (0.6, 0.8, 1.0, 1.2):
            pred = D.expected_f(base.filter(pl.col(col) > 0.01), col, miss, eb)
            res.append((D.macro_f05(pred, ntrue)['macro_f05'], dict(rule='expected_f', miss=miss, empty_bias=eb, exclusive=best[1]['exclusive'])))
    best = max(res, key=lambda r: r[0])
    return best, res


def apply_rule(scores, col, cfg):
    base = D.exclusive(scores, col) if cfg['exclusive'] else scores
    if cfg['rule'] == 'threshold':
        return D.threshold(base, col, cfg['t'], cfg.get('rel', 0.0))
    return D.expected_f(base.filter(pl.col(col) > 0.01), col, cfg['miss'], cfg['empty_bias'])


def report(name, scores, col, ntrue, cand, config, notes=''):
    best, res = sweep(scores, col, ntrue, cand, None)
    pred = apply_rule(scores, col, best[1])
    sc = D.macro_f05(pred, ntrue, cand)
    folds = {}
    for k in range(P.NF):
        nt_k = ntrue.filter(P.fold_of(pl.col('qi')) == k)
        folds[k] = D.macro_f05(pred.filter(P.fold_of(pl.col('qi')) == k), nt_k)['macro_f05']
    sc['fold_f05'] = folds
    txt = log_experiment(name, sc, {**config, 'decision': best[1]}, notes=notes + f' | per-fold F0.5 {folds}')
    print(txt, flush=True)
    return best[1], pred, sc


def main(exp, top_n=20):
    t = time.time()
    q, p = candidates.load_split('train')
    lab = P.labels(q, p)
    ntrue = s1_truth(q, lab)
    P.run_stage0('train', q, p, top_n=top_n)
    print(f'true pairs total {ntrue["ntrue"].sum()} (stage0 line above: union = blocking hits before pruning)', flush=True)
    P.run_features('train', q, p)
    del q, p
    cand = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 'label']) for f in P.parts('train', 'pr')])
    cfg = dict(top_n=top_n, blocking=candidates.CFG.__repr__()[:300])
    # ---- stage 1
    if not os.path.exists(os.path.join(P.MODELS, 'stage1.pkl')):
        P.train_stage('stage1')
    s1 = P.predict_stage('train', 'stage1')
    s1.write_parquet(os.path.join(P.CACHE, 'oof_stage1.parquet'))
    best1, _, _ = report(f'{exp}_stage1', s1, 'stage1', ntrue, cand, {**cfg, 'model': 'lgb stage1'})
    if os.environ.get('STAGES', '2') == '1':
        json.dump(best1, open(os.path.join(P.MODELS, 'decision_stage1.json'), 'w'))
        print(f'total {time.time() - t:.0f}s')
        return
    # ---- stage 2: competition features of stage-1 probabilities
    ctx = P.group_stats(s1.drop('label'), 'stage1', 'p1')
    if not os.path.exists(os.path.join(P.MODELS, 'stage2.pkl')):
        P.train_stage('stage2', extra=ctx)
    s2 = P.predict_stage('train', 'stage2', extra=ctx)
    s2.write_parquet(os.path.join(P.CACHE, 'oof_stage2.parquet'))
    best, pred, sc = report(f'{exp}_stage2', s2, 'stage2', ntrue, cand, {**cfg, 'model': 'lgb stage1+stage2'})
    json.dump(best, open(os.path.join(P.MODELS, 'decision.json'), 'w'))
    print(f'total {time.time() - t:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
