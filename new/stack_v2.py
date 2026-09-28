"""Combiner v2 = 0.977 scores + E06/E07 probabilities + E02/S3 votes (independent models), REMOVALS ONLY, US/India only.
Train votes: E02/S3 OOF probabilities (exp_meta_base.parquet) through their own decision rules = exactly how their test
outputs were produced. Test votes: output_E02 / output_S3 matching files.
Held-out: combiner trained on OOF rows outside half B (half A via 2-fold OOF for tuning); strictness chosen on half A under
BOTH density assumptions (alpha 1.9 and 2.5); reported on half B. Final: S13 = S12R-style removals from S9 with v2.
Usage: python stack_v2.py [--apply]
"""
import os, sys, json, time
import numpy as np
import polars as pl
import lightgbm as lgb
sys.path.insert(0, 'D:/Amazon ML/new')
sys.path.insert(0, 'D:/Amazon ML/amazon-ml-2026-entity-resolution-master/amazon-ml-2026-entity-resolution-master/src')
os.environ.setdefault('ER_WORK', 'D:/Amazon ML/work')
import stack_0977 as S
from tune_0977 import decide, macro
from run_train import apply_rule

R = 'D:/Amazon ML/'
ARCH = R + 'work/models_archive/'
t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
FEATS = S.FEATS + ['v02', 'v3']
STRICT = [(d, b) for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5) for b in (1.0, 1.2, 1.5, 2.0)]


def train_votes():
    base = pl.read_parquet(R + 'work/cache/exp_meta_base.parquet', columns=['qi', 'pi', 'p02', 'p3'])
    out = base.select('qi', 'pi')
    for c, v, rule in (('p02', 'v02', 'E02/decision_stage1.json'), ('p3', 'v3', 'S3/decision.json')):
        r = json.load(open(ARCH + rule))
        vote = apply_rule(base.filter(pl.col(c).is_not_null()).select('qi', 'pi', c).with_columns(pl.lit(0, pl.Int8).alias('label')), c, r).select('qi', 'pi')
        out = out.join(vote.with_columns(pl.lit(1, pl.Int8).alias(v)), on=['qi', 'pi'], how='left').with_columns(pl.col(v).fill_null(0))
        say('train votes', v, int(out[v].sum()))
    return out


def test_votes(q, p):
    out = None
    for name, v in (('E02', 'v02'), ('S3', 'v3')):
        m = (pl.read_csv(R + f'output_{name}/matching_results.tsv', separator='\t', quote_char=None, infer_schema=False).fill_null('')
               .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != '')
               .with_columns(pl.lit(1, pl.Int8).alias(v)))
        out = m if out is None else out.join(m, on=['s1', 'cand'], how='full', coalesce=True)
    return out.with_columns(pl.col('v02').fill_null(0), pl.col('v3').fill_null(0))


def main(apply=False):
    oof = S.add_ours(pl.read_parquet(S.N + 'train_oof_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'}), 'train')
    tv = train_votes()
    oof = oof.join(tv, on=['qi', 'pi'], how='left').with_columns(pl.col('v02').fill_null(0), pl.col('v3').fill_null(0))
    say('OOF rows', oof.height, 'v02 share', round(oof['v02'].mean(), 3), 'v3 share', round(oof['v3'].mean(), 3))
    gt = (pl.read_csv(S.GT, separator='\t', quote_char=None, infer_schema=False).fill_null('')
            .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    owned, nt_all = gt['cand'].unique(), gt.group_by('s1').len().rename({'len': 'ntrue'})
    nt = lambda part: part.select('s1').unique().join(nt_all, on='s1', how='left').with_columns(pl.col('ntrue').fill_null(0))
    halves = sorted(oof.filter(pl.col('held_out'))['heldout_half'].unique().to_list())
    isB, isA = pl.col('held_out') & (pl.col('heldout_half') == halves[-1]), pl.col('held_out') & (pl.col('heldout_half') == halves[0])
    tr, A, B = oof.filter(~isB), oof.filter(isA), oof.filter(isB)
    lab = oof.select('s1', 'cand', 'label')
    fit = lambda d, feats: lgb.train(S.PAR, lgb.Dataset(d.select(feats).to_numpy(), d['label'].to_numpy(), feature_name=feats), 400)
    fa = (A['s1'].hash(seed=1) % 2).to_numpy()
    res = {}
    for tag, feats in (('v1_E06E07', S.FEATS), ('v2_+E02S3votes', FEATS)):
        m = fit(tr, feats)
        pa = np.zeros(A.height)
        for k in (0, 1):
            trk = tr.join(A.filter(pl.Series(fa == k)).select('s1').unique(), on='s1', how='anti')
            pa[fa == k] = fit(trk, feats).predict(A.filter(pl.Series(fa == k)).select(feats).to_numpy())
        A2, B2 = A.with_columns(pl.Series('cs', pa)), B.with_columns(pl.Series('cs', m.predict(B.select(feats).to_numpy())))
        res[tag] = {'model': m}
        for c in ('India', 'US'):
            a, b = A.filter(pl.col('country') == c), B.filter(pl.col('country') == c)
            baseA, baseB = decide(a.select('s1', 'cand', 'score', 'country'), {}, {}), decide(b.select('s1', 'cand', 'score', 'country'), {}, {})
            def removals(part2, base, dl, bt):   # keep a base pair only if the combiner (at this strictness) also keeps it
                keep = decide(part2.select('s1', 'cand', pl.col('cs').alias('score'), 'country'), {c: dl}, {c: bt})
                return base.join(keep, on=['s1', 'cand'], how='semi')
            ntA, ntB = nt(a), nt(b)
            gain = lambda pr, base, ntp, al: macro(pr, lab, ntp, owned, al) - macro(base, lab, ntp, owned, al)
            scored = []
            for dl, bt in STRICT:
                prA = removals(A2.filter(pl.col('country') == c), baseA, dl, bt)
                scored.append((min(gain(prA, baseA, ntA, 1.9), gain(prA, baseA, ntA, 2.5)), dl, bt))   # robust to both densities
            scored.sort(reverse=True)
            _, dl, bt = scored[0]
            prB = removals(B2.filter(pl.col('country') == c), baseB, dl, bt)
            res[tag][c] = dict(delta=dl, beta=bt, removed_B=baseB.height - prB.height,
                               gainB_plain=round(gain(prB, baseB, ntB, 1.0), 5), gainB_a19=round(gain(prB, baseB, ntB, 1.9), 5), gainB_a25=round(gain(prB, baseB, ntB, 2.5), 5))
            say(tag, c, res[tag][c])
    json.dump({k: {c: v for c, v in d.items() if c != 'model'} for k, d in res.items()}, open(S.N + 'stack_v2_halfB.json', 'w'), indent=1)
    if not apply:
        return
    best = max(('v1_E06E07', 'v2_+E02S3votes'), key=lambda t: sum(res[t][c]['gainB_a19'] for c in ('India', 'US')))
    feats = FEATS if best.startswith('v2') else S.FEATS
    say('applying', best)
    test = S.add_ours(pl.read_parquet(S.N + 'test_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'}), 'test')
    test = test.join(test_votes(None, None), on=['s1', 'cand'], how='left').with_columns(pl.col('v02').fill_null(0), pl.col('v3').fill_null(0))
    test = test.with_columns(pl.Series('cs', res[best]['model'].predict(test.select(feats).to_numpy())))
    ex = lambda p: (pl.read_csv(p, separator='\t', quote_char=None, infer_schema=False).fill_null('')
                      .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    s9 = ex(R + 'S9_new_minus_fr/matching_results.tsv')
    keep_all = []
    for c in ('India', 'US'):
        tc = test.filter(pl.col('country') == c)
        keep_all.append(decide(tc.select('s1', 'cand', pl.col('cs').alias('score'), 'country'), {c: res[best][c]['delta']}, {c: res[best][c]['beta']}))
    keep = pl.concat(keep_all)
    fr_s1 = test.filter(pl.col('country') == 'France').select('s1').unique()
    s13 = pl.concat([s9.join(fr_s1, on='s1', how='semi'), s9.join(fr_s1, on='s1', how='anti').join(keep, on=['s1', 'cand'], how='semi')])
    s12r = ex(S.N + 'S12R/matching_results.tsv')
    say(f'S13 pairs {s13.height:,}; removed from S9 {s9.height - s13.height:,}; vs S12R +{s13.join(s12r, on=["s1","cand"], how="anti").height:,} -{s12r.join(s13, on=["s1","cand"], how="anti").height:,}')
    os.makedirs(S.N + 'S13', exist_ok=True)
    order = pl.read_csv(R + 'student_resource/dataset/test/test_source1.tsv', separator='\t', quote_char=None, infer_schema=False, columns=['entity_id']).rename({'entity_id': 's1'})
    g = s13.sort(['s1', 'cand']).group_by('s1', maintain_order=True).agg(pl.col('cand').str.join(','))
    out = order.join(g, on='s1', how='left').with_columns(pl.col('cand').fill_null(''))
    with open(S.N + 'S13/matching_results.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for a, b in out.iter_rows():
            f.write(f'{a}\t{b}\n')
    import shutil; shutil.copyfile(S.N + 'candidate_pairs.tsv', S.N + 'S13/candidate_pairs.tsv')
    say('S13 written')


if __name__ == '__main__':
    main('--apply' in sys.argv)
