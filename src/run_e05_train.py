"""E05 on train: union candidates -> features -> E04 (OOF) -> E02/S3 re-scored (OOF) -> ensemble rule selection.
All numbers are 2-fold grouped out-of-fold over ALL 2.2M train S1 (the S4 rule E02∩S3 is reproduced on train too).
"""
import os, json, time
import numpy as np
import polars as pl
import candidates, pipeline as P, decide as D, e05
from run_train import sweep, apply_rule, s1_truth
from evaluate import log_experiment
from common import load_norm

t0 = time.time()
R = os.path.join(P.CACHE, 'e05')
os.makedirs(R, exist_ok=True)
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)

if not P.parts('train', 'feat'):
    q, p = candidates.load_split('train')
    e05.run_stage0_union('train', q, p)
    e05.run_features_union('train', q, p)
    del q, p
lab_q = load_norm('train', 1).select(pl.int_range(pl.len()).cast(pl.UInt32).alias('qi'), 'country_n')
gt = pl.read_parquet(os.path.join(P.CACHE, 'truth_counts.parquet'))
ntrue = lab_q.select('qi').join(gt, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
info = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 'label', 'num_first_eq', 'in_e04', 'in_leg']) for f in P.parts('train', 'feat')]) \
         .with_columns(pl.col('qi').cast(pl.UInt32))
info = info.with_columns((pl.col('num_first_eq') == 1).fill_null(False).any().over('qi').alias('_sib'))
info = info.with_columns(((pl.col('num_first_eq') == 0) & pl.col('_sib')).fill_null(False).alias('hard')).drop('_sib')
cand = info.select('qi', 'pi', 'label')
say('union candidates:', info.height, 'pairs;', round(info.height / lab_q.height, 2), 'per S1;',
    'e04-only', info.filter((pl.col('in_e04') == 1) & (pl.col('in_leg') == 0)).height,
    'legacy-only', info.filter((pl.col('in_e04') == 0) & (pl.col('in_leg') == 1)).height)

# ---------------- E04
if not os.path.exists(os.path.join(P.MODELS, 'stage1_e04.pkl')):
    e05.train_e04('stage1_e04')
s1 = e05.predict_e04('train', 'stage1_e04').with_columns(pl.col('qi').cast(pl.UInt32))
ctx1 = P.group_stats(s1.select('qi', 'pi', 'stage1_e04'), 'stage1_e04', 'p1')
if not os.path.exists(os.path.join(P.MODELS, 'stage2_e04.pkl')):
    e05.train_e04('stage2_e04', extra=ctx1)
s2 = e05.predict_e04('train', 'stage2_e04', extra=ctx1).with_columns(pl.col('qi').cast(pl.UInt32))
s2.write_parquet(os.path.join(R, 'oof_e04.parquet'))
best04, _ = sweep(s2.rename({'stage2_e04': 'p'}), 'p', ntrue, None, None)
json.dump(best04[1], open(os.path.join(P.MODELS, 'decision_e04.json'), 'w'))
v04 = apply_rule(s2.rename({'stage2_e04': 'p'}), 'p', best04[1]).select('qi', 'pi')
say('E04 decision rule', best04[1])

# ---------------- legacy models on the legacy pool (their own training conditions)
v = {}
for name in ('E02', 'S3'):
    sc = e05.score_legacy('train', name).with_columns(pl.col('qi').cast(pl.UInt32))
    sc.write_parquet(os.path.join(R, f'oof_{name}.parquet'))
    v[name] = e05.legacy_votes(sc, name)
    say(name, 'rescored on legacy pool:', sc.height, 'pairs')

# ---------------- rule comparison
def report(tag, pred):
    pred = pred.join(cand, on=['qi', 'pi'], how='left').with_columns(pl.col('label').fill_null(0))
    sc = D.macro_f05(pred, ntrue, cand)
    per = {}
    for c in ('us', 'india'):
        kq = lab_q.filter(pl.col('country_n') == c).select('qi')
        per[c] = round(D.macro_f05(pred.join(kq, on='qi'), ntrue.join(kq, on='qi'))['macro_f05'], 5)
    hq = info.filter(pl.col('hard'))
    hp = pred.join(hq.select('qi', 'pi'), on=['qi', 'pi'], how='semi')
    h_tp, h_true = int(hp['label'].sum()), int(hq['label'].sum())
    sc.update(per_country=per, hard_precision=round(h_tp / max(hp.height, 1), 4), hard_recall=round(h_tp / max(h_true, 1), 4),
              hard_pred=hp.height)
    sc['fold_f05'] = {k: round(D.macro_f05(pred.filter(P.fold_of(pl.col('qi')) == k), ntrue.filter(P.fold_of(pl.col('qi')) == k))['macro_f05'], 5) for k in range(P.NF)}
    print(log_experiment(f'E05_{tag}', sc, dict(rule=tag, e04_decision=best04[1])), flush=True)
    print(f'   per-country {per} | hard subgroup P={sc["hard_precision"]} R={sc["hard_recall"]} n={hp.height} | folds {sc["fold_f05"]}', flush=True)
    return sc

res = {}
for rule in ('e02_and_s3', 'e04', 'majority', 'e04_and_any'):
    res[rule] = report(rule, e05.ensemble(v04, v['E02'], v['S3'], rule))
for name in ('E02', 'S3'):
    res[name] = report(name + '_alone', v[name])

# ---------------- decorrelation + foreign-candidate diagnostics
k = ['qi', 'pi']
agree = lambda a, b: a.join(b, on=k, how='semi').height / max(a.height, 1)
say('agreement: E04->E02', round(agree(v04, v['E02']), 4), 'E04->S3', round(agree(v04, v['S3']), 4),
    'E02->S3', round(agree(v['E02'], v['S3']), 4))
src = s2.with_columns(pl.when((pl.col('in_e04') == 1) & (pl.col('in_leg') == 0)).then(pl.lit('e04_only'))
                        .when((pl.col('in_e04') == 0) & (pl.col('in_leg') == 1)).then(pl.lit('foreign_legacy_only'))
                        .otherwise(pl.lit('both')).alias('src'))
print(src.group_by('src').agg(pl.len(), pl.col('label').mean().alias('pos_rate'),
                              pl.col('stage2_e04').quantile(0.5).alias('p50'), pl.col('stage2_e04').quantile(0.9).alias('p90'),
                              ((pl.col('stage2_e04') > 0.1) & (pl.col('stage2_e04') < 0.9)).mean().alias('uncertain_frac')), flush=True)
json.dump({kk: {x: (vv[x] if not isinstance(vv[x], (np.floating,)) else float(vv[x])) for x in ('macro_f05', 'pair_precision', 'pair_recall', 'hard_precision', 'hard_recall')} for kk, vv in res.items()},
          open(os.path.join(R, 'rule_comparison.json'), 'w'), indent=1, default=float)
say('done')
