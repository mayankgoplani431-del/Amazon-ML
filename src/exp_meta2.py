"""Deployable soft meta-model: features that EXIST ON TEST = E06/E07 probabilities + E02/S3 yes/no votes.
Train side: votes of E02/S3 come from their OOF probabilities through their own decision rules (exp_meta_base.parquet).
1. Held-out evaluation (train fold k -> evaluate fold 1-k): soft meta vs hard-AND (E02 ∩ S3 ∩ E06 = S5 logic).
2. If --apply: fit on all OOF, tune the decision rule on OOF, score test, write <out>/matching_results.tsv and
   <out>/candidate_pairs.tsv (union of E06, E07 and S3 candidate sets = everything the models scored).
Usage (ER_WORK = legacy work dir): python exp_meta2.py <work_e06> <work_e07> <out_dir> [--apply]
"""
import os, sys, json, time
import numpy as np
import polars as pl
import lightgbm as lgb
import pipeline as P, decide as D
from run_train import sweep, apply_rule

t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
R = 'D:/Amazon ML'
K = ['qi', 'pi']
FEATS = ['p06', 'p07', 'v02', 'v3', 'v06', 'v07', 'nv']
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, verbose=-1, num_threads=14, seed=0)


def add_votes(df, rules):
    for c, v in (('p06', 'v06'), ('p07', 'v07')):
        vote = apply_rule(df.filter(pl.col(c).is_not_null()).select(K + [c]).with_columns(pl.lit(0, pl.Int8).alias('label')), c, rules[c]).select(K)
        df = df.join(vote.with_columns(pl.lit(1, pl.Int8).alias(v)), on=K, how='left').with_columns(pl.col(v).fill_null(0))
    return df.with_columns((pl.col('v02') + pl.col('v3') + pl.col('v06') + pl.col('v07')).alias('nv'))


def main(w06, w07, out, apply=False):
    ARCH = os.path.join(P.WORK, 'models_archive')
    rules = {'p02': json.load(open(os.path.join(ARCH, 'E02', 'decision_stage1.json'))),
             'p3': json.load(open(os.path.join(ARCH, 'S3', 'decision.json'))),
             'p06': json.load(open(os.path.join(w06, 'models', 'decision_stage1.json'))),
             'p07': json.load(open(os.path.join(w07, 'models', 'decision_stage1.json')))}
    base = pl.read_parquet(os.path.join(P.CACHE, 'exp_meta_base.parquet'))
    e07 = pl.read_parquet(os.path.join(w07, 'cache', 'oof_stage1.parquet')).select(
        pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('label').alias('l7'), pl.col('stage1').alias('p07'))
    base = base.join(e07, on=K, how='full', coalesce=True).with_columns(pl.coalesce('label', 'l7').cast(pl.Int8).alias('label'),
                                                                        P.fold_of(pl.col('qi')).alias('fold')).drop('l7')
    for c, v in (('p02', 'v02'), ('p3', 'v3')):
        vote = apply_rule(base.filter(pl.col(c).is_not_null()).select(K + ['label', c]), c, rules[c]).select(K)
        base = base.join(vote.with_columns(pl.lit(1, pl.Int8).alias(v)), on=K, how='left').with_columns(pl.col(v).fill_null(0))
    base = add_votes(base, rules)
    say('train base:', base.height, 'pairs')
    gt = pl.read_parquet(os.path.join(P.CACHE, 'truth_counts.parquet'))
    n_s1 = pl.read_parquet(os.path.join(P.CACHE, 'train_s1.parquet'), columns=['entity_id']).height
    ntrue = pl.DataFrame({'qi': pl.arange(0, n_s1, eager=True).cast(pl.UInt32)}).join(gt, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
    owned = base.filter(pl.col('label') == 1)['pi'].unique()
    lab = base.select(K + ['label'])
    hard = base.filter((pl.col('v02') == 1) & (pl.col('v3') == 1) & (pl.col('v06') == 1)).select(K)
    preds, rules_meta = [], []
    for f in (0, 1):
        tr, ev = base.filter(pl.col('fold') != f), base.filter(pl.col('fold') == f)
        m = lgb.train(PAR, lgb.Dataset(tr.select(FEATS).to_numpy(), tr['label'].to_numpy()), 300)
        tr = tr.select(K + ['label']).with_columns(pl.Series('pm', m.predict(tr.select(FEATS).to_numpy())))
        smp = (pl.col('qi').hash(seed=5) % 10) < 3   # 30% S1 sample for the rule sweep (memory)
        r = sweep(tr.filter(smp), 'pm', ntrue.filter((P.fold_of(pl.col('qi')) != f) & smp), None, None, owned=owned, alpha=1.9)[0][1]
        rules_meta.append(r)
        preds.append(apply_rule(ev.select(K + ['label']).with_columns(pl.Series('pm', m.predict(ev.select(FEATS).to_numpy()))), 'pm', r).select(K))
        say(f'fold {f} meta rule: {r}')
    soft = pl.concat(preds)
    res = {}
    for tag, pred in (('hard_AND_S5', hard), ('soft_meta', soft)):
        pr = pred.join(lab, on=K, how='left').with_columns(pl.col('label').fill_null(0))
        a, b = D.macro_f05(pr, ntrue), D.macro_f05(pr, ntrue, owned=owned, alpha=1.9)
        res[tag] = dict(plain=round(a['macro_f05'], 5), stressed=round(b['macro_f05'], 5), precision=round(a['pair_precision'], 5),
                        recall=round(a['pair_recall'], 5), fp=a['false_pos'], fn=a['false_neg'], pairs=pred.height)
        say(tag, res[tag])
    d_add, d_rem = soft.join(hard, on=K, how='anti').join(lab, on=K), hard.join(soft, on=K, how='anti').join(lab, on=K)
    res['diff'] = dict(added=d_add.height, added_true_share=round(float(d_add['label'].mean()), 4) if d_add.height else None,
                       removed=d_rem.height, removed_true_share=round(float(d_rem['label'].mean()), 4) if d_rem.height else None)
    say('soft vs hard differences (held-out, with labels):', res['diff'])
    # calibration set = train analogue of S4 - S5 (E02 and S3 say yes, E06 says no); its TEST true share is known
    # from the leaderboard (0.969 -> 0.971: ~57-66% true), so train/test degradation of disputed pairs can be measured
    cal = base.filter((pl.col('v02') == 1) & (pl.col('v3') == 1) & (pl.col('v06') == 0))
    add_in_cal = d_add.join(cal.select(K), on=K, how='semi')
    res['calibration'] = dict(s4_minus_s5_train_n=cal.height, s4_minus_s5_train_true_share=round(float(cal['label'].mean()), 4),
                              soft_added_within_it=add_in_cal.height,
                              soft_added_within_it_true_share=round(float(add_in_cal['label'].mean()), 4) if add_in_cal.height else None)
    say('calibration:', res['calibration'])
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, 'exp_meta2.json'), 'w'), indent=1)
    if not apply:
        return
    # ---- test: fit on all OOF, rule tuned on all OOF (density-stressed), score test pairs
    m = lgb.train(PAR, lgb.Dataset(base.select(FEATS).to_numpy(), base['label'].to_numpy()), 300)
    rule = sweep(base.select(K + ['label']).with_columns(pl.Series('pm', m.predict(base.select(FEATS).to_numpy()))), 'pm', ntrue, None, None, owned=owned, alpha=1.9)[0][1]
    ids_q = pl.read_parquet(os.path.join(w06, 'cache', 'test_s1.parquet'), columns=['entity_id']).with_row_index('qi')
    p_ids = pl.concat([pl.read_parquet(os.path.join(w06, 'cache', f'test_s{i}.parquet'), columns=['entity_id']) for i in (2, 3)]).with_row_index('pi')
    def votes(path):
        m_ = pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False).fill_null('')
        return (m_.select(pl.col('source1_entity_id').alias('entity_id'), pl.col('matched_entity_ids').str.split(',').alias('c')).explode('c')
                  .filter(pl.col('c') != '').join(ids_q, on='entity_id').join(p_ids.rename({'entity_id': 'c'}), on='c').select(K))
    t06 = pl.read_parquet(os.path.join(w06, 'cache', 'test_stage1.parquet')).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('stage1').alias('p06'))
    t07 = pl.read_parquet(os.path.join(w07, 'cache', 'test_stage1.parquet')).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('stage1').alias('p07'))
    te = t06.join(t07, on=K, how='full', coalesce=True)
    for name, v in (('E02', 'v02'), ('S3', 'v3')):
        vv = votes(os.path.join(R, f'output_{name}', 'matching_results.tsv')).with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
        te = te.join(vv.with_columns(pl.lit(1, pl.Int8).alias(v)), on=K, how='full', coalesce=True).with_columns(pl.col(v).fill_null(0))
    te = add_votes(te, rules)
    te = te.with_columns(pl.Series('pm', m.predict(te.select(FEATS).to_numpy())), pl.lit(0, pl.Int8).alias('label'))
    pred = apply_rule(te.select(K + ['label', 'pm']), 'pm', rule).select(K)
    say('test rule', rule, 'predicted pairs', pred.height)
    from run_test import write_lists
    idq = ids_q.select(pl.col('qi').cast(pl.UInt32), pl.col('entity_id').alias('source1_entity_id'))
    idp = p_ids.select(pl.col('pi').cast(pl.UInt32), pl.col('entity_id').alias('cand_id'))
    write_lists(idq, pred, idp, 'matched_entity_ids', os.path.join(out, 'matching_results.tsv'))
    say('matching written; build candidate union with vote.py/combos (E06, E07, S3 candidate files)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], '--apply' in sys.argv)
