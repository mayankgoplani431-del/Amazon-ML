"""Score-level combiner: 0.977 model (EXP-1) scores + our independent E06/E07 probabilities, held-out.
Train rows: OOF pairs of S1 NOT in half B (extra 200k + half A); tune decision on half A; report on half B (never used).
Base = the 0.977 decision on its own scores. Metric: plain and density-stressed (alpha 1.9) macro F0.5.
If --apply: fit on all non-half-B rows, score test, decide, apply S9 France removals, write S12/.
Usage: python stack_0977.py [--apply]
"""
import os, sys, json, time
import numpy as np
import polars as pl
import lightgbm as lgb
sys.path.insert(0, 'D:/Amazon ML/new')
from tune_0977 import decide, macro, DELTAS, BETAS

N = 'D:/Amazon ML/new/'
W6 = 'D:/Amazon ML/amazon-ml-2026-entity-resolution-master/amazon-ml-2026-entity-resolution-master/work_e06/cache/'
W7 = 'D:/Amazon ML/amazon-ml-2026-entity-resolution-master/amazon-ml-2026-entity-resolution-master/work_e07/cache/'
GT = 'D:/Amazon ML/student_resource/dataset/train/train_ground_truth.tsv'
t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9, verbose=-1, num_threads=14, seed=0)


def ids(split):
    q = pl.read_parquet(W6 + f'{split}_s1.parquet', columns=['entity_id']).with_row_index('qi').rename({'entity_id': 's1'})
    p = pl.concat([pl.read_parquet(W6 + f'{split}_s{i}.parquet', columns=['entity_id']) for i in (2, 3)]).with_row_index('pi').rename({'entity_id': 'cand'})
    return q.with_columns(pl.col('qi').cast(pl.UInt32)), p.with_columns(pl.col('pi').cast(pl.UInt32))


def add_ours(df, split):
    q, p = ids(split)
    df = df.join(q, on='s1', how='left').join(p, on='cand', how='left')
    for tag, w, f in (('p06', W6, 'oof_stage1.parquet' if split == 'train' else 'test_stage1.parquet'),
                      ('p07', W7, 'oof_stage1.parquet' if split == 'train' else 'test_stage1.parquet')):
        o = pl.read_parquet(w + f, columns=['qi', 'pi', 'stage1']).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32), pl.col('stage1').alias(tag))
        df = df.join(o, on=['qi', 'pi'], how='left')
    return df.with_columns(pl.col('score').rank('ordinal', descending=True).over('s1').alias('srank'),
                           (pl.col('score').max().over('s1') - pl.col('score')).alias('sgap'),
                           pl.col('p06').is_not_null().cast(pl.Int8).alias('has06'), pl.col('p07').is_not_null().cast(pl.Int8).alias('has07'))


FEATS = ['score', 'stage1_score', 'ranker_score', 'cut_rank', 'srank', 'sgap', 'p06', 'p07', 'has06', 'has07']


def main(apply=False):
    oof = pl.read_parquet(N + 'train_oof_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'})
    oof = add_ours(oof, 'train')
    say('train rows', oof.height, 'coverage p06', round(oof['has06'].mean(), 3), 'p07', round(oof['has07'].mean(), 3))
    gt = (pl.read_csv(GT, separator='\t', quote_char=None, infer_schema=False).fill_null('')
            .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    owned, nt_all = gt['cand'].unique(), gt.group_by('s1').len().rename({'len': 'ntrue'})
    nt = lambda part: part.select('s1').unique().join(nt_all, on='s1', how='left').with_columns(pl.col('ntrue').fill_null(0))
    halves = sorted(oof.filter(pl.col('held_out'))['heldout_half'].unique().to_list())
    isB = pl.col('held_out') & (pl.col('heldout_half') == halves[-1])
    isA = pl.col('held_out') & (pl.col('heldout_half') == halves[0])
    tr, A, B = oof.filter(~isB), oof.filter(isA), oof.filter(isB)
    lab = oof.select('s1', 'cand', 'label')
    m = lgb.train(PAR, lgb.Dataset(tr.select(FEATS).to_numpy(), tr['label'].to_numpy(), feature_name=FEATS), 400)
    # (half A rows are in the combiner's training set -> tune the decision on OUT-OF-FOLD combiner scores for A)
    fa = (A['s1'].hash(seed=1) % 2).to_numpy()
    pa = np.zeros(A.height)
    for k in (0, 1):
        trk = oof.filter(~isB).join(A.filter(pl.Series(fa == k)).select('s1').unique(), on='s1', how='anti')
        mk = lgb.train(PAR, lgb.Dataset(trk.select(FEATS).to_numpy(), trk['label'].to_numpy(), feature_name=FEATS), 400)
        pa[fa == k] = mk.predict(A.filter(pl.Series(fa == k)).select(FEATS).to_numpy())
    A2 = A.with_columns(pl.Series('score', pa))
    B2 = B.with_columns(pl.Series('score', m.predict(B.select(FEATS).to_numpy())))
    best, rep = {}, {}
    for c in ('India', 'US'):
        a, ntA = A2.filter(pl.col('country') == c), nt(A2.filter(pl.col('country') == c))
        grid = sorted(((macro(decide(a.select('s1', 'cand', 'score', 'country'), {c: dl}, {c: bt}), lab, ntA, owned, 1.9), dl, bt)
                       for dl in DELTAS for bt in BETAS), reverse=True)
        best[c] = grid[0]
        b, ntB = B.filter(pl.col('country') == c), nt(B.filter(pl.col('country') == c))
        b2 = B2.filter(pl.col('country') == c)
        base = decide(b.select('s1', 'cand', 'score', 'country'), {}, {})
        new = decide(b2.select('s1', 'cand', 'score', 'country'), {c: grid[0][1]}, {c: grid[0][2]})
        rep[c] = dict(base_plain=round(macro(base, lab, ntB, owned, 1.0), 5), base_stressed=round(macro(base, lab, ntB, owned, 1.9), 5),
                      comb_plain=round(macro(new, lab, ntB, owned, 1.0), 5), comb_stressed=round(macro(new, lab, ntB, owned, 1.9), 5),
                      delta=grid[0][1], beta=grid[0][2])
    imp = dict(zip(FEATS, m.feature_importance('gain').round(0).tolist()))
    say('HALF B (held out):', json.dumps(rep, indent=1), '\nfeature gain:', imp)
    json.dump(dict(half_B=rep, gain=imp), open(N + 'stack_halfB.json', 'w'), indent=1)
    if not apply:
        return
    test = add_ours(pl.read_parquet(N + 'test_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'}), 'test')
    say('test coverage p06', round(test['has06'].mean(), 3), 'p07', round(test['has07'].mean(), 3))
    test = test.with_columns(pl.Series('score', m.predict(test.select(FEATS).to_numpy())))
    delta = {c: best[c][1] for c in best}; beta = {c: best[c][2] for c in best}
    delta['France'], beta['France'] = max(delta['India'], 0.0), max(beta['India'], 1.0)
    pred = decide(test.select('s1', 'cand', 'score', 'country'), delta, beta)
    ex = lambda p: (pl.read_csv(p, separator='\t', quote_char=None, infer_schema=False).fill_null('')
                      .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    sub, s9 = ex(N + 'matching_results.tsv'), ex('D:/Amazon ML/S9_new_minus_fr/matching_results.tsv')
    pred = pred.join(sub.join(s9, on=['s1', 'cand'], how='anti'), on=['s1', 'cand'], how='anti')   # S9 France removals
    # unknown country: our models (US/India-trained) accept French word-swap distractors -> France stays exactly as S9
    fr_s1 = test.filter(pl.col('country') == 'France').select('s1').unique()
    pred = pl.concat([pred.join(fr_s1, on='s1', how='anti'), s9.join(fr_s1, on='s1', how='semi')])
    pred = pred.join(pred.group_by('cand').len().filter(pl.col('len') > 1).select('cand'), on='cand', how='anti') if False else pred
    dup = pred.group_by('cand').len().filter(pl.col('len') > 1).height
    assert dup == 0, f'{dup} candidates assigned to >1 S1'
    say(f'S12 pairs {pred.height:,}; vs S9 +{pred.join(s9, on=["s1","cand"], how="anti").height:,} -{s9.join(pred, on=["s1","cand"], how="anti").height:,}')
    os.makedirs(N + 'S12', exist_ok=True)
    order = pl.read_csv('D:/Amazon ML/student_resource/dataset/test/test_source1.tsv', separator='\t', quote_char=None, infer_schema=False, columns=['entity_id']).rename({'entity_id': 's1'})
    g = pred.sort(['s1', 'cand']).group_by('s1', maintain_order=True).agg(pl.col('cand').str.join(','))
    out = order.join(g, on='s1', how='left').with_columns(pl.col('cand').fill_null(''))
    with open(N + 'S12/matching_results.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for a, b in out.iter_rows():
            f.write(f'{a}\t{b}\n')
    import shutil; shutil.copyfile(N + 'candidate_pairs.tsv', N + 'S12/candidate_pairs.tsv')
    say('S12 written')


if __name__ == '__main__':
    main('--apply' in sys.argv)
