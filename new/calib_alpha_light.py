"""Calibrate test distractor density (alpha) from the leaderboard, then re-optimise removal strictness.
Anchor 1: S12R (v1 combiner removals from the 0.977 decision) gained +0.0026 on the leaderboard (0.9796 -> 0.9822).
Find alpha where half-B predicts that gain (S1-share weighted: India 0.4675, US 0.3827 of test S1; France untouched).
Anchor 2 (check): S14 vs S12R = +0.0003.
Then choose v2 removal strictness on half A at the calibrated alpha, report half B, build S15 on test.
Usage: python calib_alpha.py
"""
import os, sys, json, time, shutil
import numpy as np
import polars as pl
import lightgbm as lgb
sys.path.insert(0, 'D:/Amazon ML/new')
import stack_v2 as V, stack_0977 as S
from tune_0977 import decide, macro

t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
W = {'India': 0.4675, 'US': 0.3827}
ALPHAS = [1.0, 1.9, 3.0, 5.0, 8.0]
STRICT = [(0.25, 1.2), (0.75, 1.2), (1.25, 1.5), (2.0, 2.0)]   # light grid: S13-like -> much stricter
S12R_CFG = {'India': (0.0, 1.5), 'US': (0.25, 1.0)}          # the v1 settings S12R was built with (stack_halfB.json)


def main():
    oof = S.add_ours(pl.read_parquet(S.N + 'train_oof_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'}), 'train')
    oof = oof.join(V.train_votes(), on=['qi', 'pi'], how='left').with_columns(pl.col('v02').fill_null(0), pl.col('v3').fill_null(0))
    gt = (pl.read_csv(S.GT, separator='\t', quote_char=None, infer_schema=False).fill_null('')
            .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    owned, nt_all = gt['cand'].unique(), gt.group_by('s1').len().rename({'len': 'ntrue'})
    nt = lambda part: part.select('s1').unique().join(nt_all, on='s1', how='left').with_columns(pl.col('ntrue').fill_null(0))
    lab = oof.select('s1', 'cand', 'label')
    halves = sorted(oof.filter(pl.col('held_out'))['heldout_half'].unique().to_list())
    isB, isA = pl.col('held_out') & (pl.col('heldout_half') == halves[-1]), pl.col('held_out') & (pl.col('heldout_half') == halves[0])
    tr, A, B = oof.filter(~isB), oof.filter(isA), oof.filter(isB)
    fit = lambda d, f: lgb.train(S.PAR, lgb.Dataset(d.select(f).to_numpy(), d['label'].to_numpy(), feature_name=f), 400)
    m1, m2 = fit(tr, S.FEATS), lgb.Booster(model_file=S.N + 'combiner_v2.txt')
    B1 = B.with_columns(pl.Series('cs', m1.predict(B.select(S.FEATS).to_numpy())))
    B2 = B.with_columns(pl.Series('cs', m2.predict(B.select(V.FEATS).to_numpy())))
    mA = fit(tr.join(A.select('s1').unique(), on='s1', how='anti'), V.FEATS)   # one refit without half A (light)
    A2 = A.with_columns(pl.Series('cs', mA.predict(A.select(V.FEATS).to_numpy())))
    def removal_gain(part, part2, c, dl, bt, alphas):
        p = part.filter(pl.col('country') == c)
        base = decide(p.select('s1', 'cand', 'score', 'country'), {}, {})
        keep = decide(part2.filter(pl.col('country') == c).select('s1', 'cand', pl.col('cs').alias('score'), 'country'), {c: dl}, {c: bt})
        pr, n = base.join(keep, on=['s1', 'cand'], how='semi'), nt(p)
        return {a: macro(pr, lab, n, owned, a) - macro(base, lab, n, owned, a) for a in alphas}, base.height - pr.height
    # ---- anchor 1: S12R config on half B
    g12 = {c: removal_gain(B, B1, c, *S12R_CFG[c], ALPHAS)[0] for c in W}
    pred12 = {a: sum(W[c] * g12[c][a] for c in W) for a in ALPHAS}
    say('S12R predicted LB gain by alpha:', {a: round(v, 5) for a, v in pred12.items()}, '| actual +0.0026')
    xs, ys = np.array(ALPHAS), np.array([pred12[a] for a in ALPHAS])
    a_star = float(np.interp(0.0026, ys, xs)) if ys.max() >= 0.0026 else float(xs[-1])
    say(f'calibrated alpha* = {a_star:.2f}')
    al = sorted(set([1.9, round(a_star, 2)]))
    # ---- strictness on half A at alpha*, reported on half B
    best = {}
    for c in W:
        grid = sorted(((removal_gain(A, A2, c, dl, bt, [a_star])[0][a_star], dl, bt) for dl, bt in STRICT), reverse=True)
        best[c] = grid[0]
        gB, nrem = removal_gain(B, B2, c, grid[0][1], grid[0][2], [1.0, 1.9, a_star])
        s13c = json.load(open(S.N + 'stack_v2_halfB.json'))['v2_+E02S3votes'][c]
        gB13, _ = removal_gain(B, B2, c, s13c['delta'], s13c['beta'], [1.0, 1.9, a_star])
        say(f'{c}: alpha* strictness delta={grid[0][1]} beta={grid[0][2]} removes {nrem} on B | half-B gain {{plain, 1.9, a*}}: '
            f'{[round(gB[a], 5) for a in (1.0, 1.9, a_star)]} vs S13 setting {[round(gB13[a], 5) for a in (1.0, 1.9, a_star)]}')
    json.dump(dict(alpha_star=a_star, best={c: dict(delta=best[c][1], beta=best[c][2]) for c in W}), open(S.N + 'calib_alpha.json', 'w'), indent=1)
    # ---- S15 on test: S9 base, v2 removals at alpha* strictness (US/India), + S14 confident additions, France = S9
    test = S.add_ours(pl.read_parquet(S.N + 'test_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'}), 'test')
    test = test.join(V.test_votes(None, None), on=['s1', 'cand'], how='left').with_columns(pl.col('v02').fill_null(0), pl.col('v3').fill_null(0))
    test = test.with_columns(pl.Series('cs', m2.predict(test.select(V.FEATS).to_numpy())))
    ex = lambda p: (pl.read_csv(p, separator='\t', quote_char=None, infer_schema=False).fill_null('')
                      .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    s9, s13, s14 = ex('D:/Amazon ML/S9_new_minus_fr/matching_results.tsv'), ex(S.N + 'S13/matching_results.tsv'), ex(S.N + 'S14/matching_results.tsv')
    keep = pl.concat([decide(test.filter(pl.col('country') == c).select('s1', 'cand', pl.col('cs').alias('score'), 'country'),
                             {c: best[c][1]}, {c: best[c][2]}) for c in W])
    fr = test.filter(pl.col('country') == 'France').select('s1').unique()
    base15 = pl.concat([s9.join(fr, on='s1', how='semi'), s9.join(fr, on='s1', how='anti').join(keep, on=['s1', 'cand'], how='semi')])
    adds = s14.join(s13, on=['s1', 'cand'], how='anti')                                          # S14's confident additions
    s15 = pl.concat([base15, adds.join(base15.select('cand'), on='cand', how='anti')]).unique()
    assert s15.group_by('cand').len().filter(pl.col('len') > 1).height == 0
    say(f'S15 pairs {s15.height:,} | vs S14 +{s15.join(s14, on=["s1","cand"], how="anti").height:,} -{s14.join(s15, on=["s1","cand"], how="anti").height:,}')
    os.makedirs(S.N + 'S15', exist_ok=True)
    order = pl.read_csv('D:/Amazon ML/student_resource/dataset/test/test_source1.tsv', separator='\t', quote_char=None, infer_schema=False, columns=['entity_id']).rename({'entity_id': 's1'})
    g = s15.sort(['s1', 'cand']).group_by('s1', maintain_order=True).agg(pl.col('cand').str.join(','))
    out = order.join(g, on='s1', how='left').with_columns(pl.col('cand').fill_null(''))
    with open(S.N + 'S15/matching_results.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for a, b in out.iter_rows():
            f.write(f'{a}\t{b}\n')
    shutil.copyfile(S.N + 'candidate_pairs.tsv', S.N + 'S15/candidate_pairs.tsv')
    say('S15 written')


if __name__ == '__main__':
    main()
