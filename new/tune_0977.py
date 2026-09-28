"""Held-out re-tuning of the 0.977 model's decision (EXP-1 scores) for test density + unseen country, then S11.

Inputs (D:/Amazon ML/new/): train_oof_scores.parquet, test_scores.parquet, candidate_pairs.tsv, matching_results.tsv (0.977),
../S9_new_minus_fr/matching_results.tsv (S9 = 0.977 minus the validated France house-number rule).
1. Re-implement decision.json exactly; check it reproduces the 0.977 file from test_scores.
2. Tune per labelled country on HALF A of the held-out training S1 only: logit shift delta (stricter = higher) and
   empty weight beta, scored with density-stressed macro F0.5 (FP on ownerless candidates x1.9: test has ~1.9x
   the distractors of train). Report plain + stressed on HALF B (never used for any choice).
3. France (no labels): delta chosen label-free so France's predicted empty-rate / matches-per-S1 profile moves to the
   US+India test profile (validated on India-as-unseen: +0.0065 at this conservative pick), never below India's delta.
4. Apply S9's France removals (leaderboard-validated) on top; write S11/.
Usage: python tune_0977.py
"""
import os, json, math, time
import numpy as np
import polars as pl

N = os.environ.get('TUNE_DIR', 'D:/Amazon ML/new/')
GT = 'D:/Amazon ML/student_resource/dataset/train/train_ground_truth.tsv'
MISS = {'India': 0.0, 'US': 0.2, 'France': 0.0}
DELTAS = [round(x, 2) for x in np.arange(-0.5, 2.51, 0.25)]
BETAS = [1.0, 1.2, 1.5]
t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)


def decide(df, delta, beta):
    """df: s1, cand, score, country. delta/beta: dict per country. Implements decision.json + one owner."""
    d = df.with_columns(pl.col('country').replace_strict(delta, default=0.0).alias('_d'),
                        pl.col('country').replace_strict(beta, default=1.0).alias('_b'),
                        pl.col('country').replace_strict(MISS, default=0.0).alias('_m'))
    lg = (pl.col('score').clip(1e-6, 1 - 1e-6) / (1 - pl.col('score').clip(1e-6, 1 - 1e-6))).log()
    d = d.with_columns((1 / (1 + (-(lg - pl.col('_d'))).exp())).alias('s'))
    d = (d.sort(['s1', 's'], descending=[False, True])
          .with_columns(pl.col('s').cum_sum().over('s1').alias('_tp'), pl.int_range(1, pl.len() + 1).over('s1').alias('_n'),
                        (pl.col('s').sum().over('s1') * (1 + pl.col('_m'))).alias('_T'),
                        (1 - pl.col('s')).clip(1e-9, 1).log().sum().over('s1').exp().alias('_p0')))
    d = d.with_columns((1.25 * pl.col('_tp') / (0.25 * pl.col('_T') + pl.col('_n'))).alias('_ef'))
    d = d.with_columns(pl.col('_ef').max().over('s1').alias('_best'))
    nb = d.filter(pl.col('_ef') == pl.col('_best')).group_by('s1').agg(pl.col('_n').min().alias('_nb'))
    d = d.join(nb, on='s1').filter((pl.col('_n') <= pl.col('_nb')) & (pl.col('_best') > pl.col('_b') * pl.col('_p0')))
    return d.sort('s', descending=True).unique(subset=['cand'], keep='first').select('s1', 'cand')


def macro(pred, lab, ntrue, owned, alpha=1.0):
    p = pred.join(lab, on=['s1', 'cand'], how='left').with_columns(pl.col('label').fill_null(0))
    p = p.with_columns(((pl.col('label') == 0) & ~pl.col('cand').is_in(owned.implode())).alias('_un'))
    g = p.group_by('s1').agg(pl.col('label').sum().alias('tp'), (pl.len() + (alpha - 1) * pl.col('_un').sum()).alias('np'))
    x = ntrue.join(g, on='s1', how='left').with_columns(pl.col('tp').fill_null(0), pl.col('np').fill_null(0))
    f = (pl.when(pl.col('ntrue') == 0).then((pl.col('np') == 0).cast(pl.Float64))
           .when((pl.col('np') == 0) | (pl.col('tp') == 0)).then(0.0)
           .otherwise(1.25 * pl.col('tp') / (pl.col('np') + 0.25 * pl.col('ntrue'))))
    return float(x.select(f.mean()).item())


def profile(pred, s1s):
    k = s1s.join(pred.group_by('s1').len(), on='s1', how='left').with_columns(pl.col('len').fill_null(0))['len']
    return float((k == 0).mean()), float(k.mean())


def main():
    oof = pl.read_parquet(N + 'train_oof_scores.parquet')
    test = pl.read_parquet(N + 'test_scores.parquet')
    ren = {'source1_entity_id': 's1', 'candidate_entity_id': 'cand'}
    oof, test = oof.rename(ren), test.rename(ren)
    say('oof', oof.shape, 'test', test.shape, 'oof cols', oof.columns)
    gt = (pl.read_csv(GT, separator='\t', quote_char=None, infer_schema=False).fill_null('')
            .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    owned = gt['cand'].unique()
    # ---- 1. reproduce the 0.977 file
    base = decide(test.select('s1', 'cand', 'score', 'country'), {}, {})
    sub = (pl.read_csv(N + 'matching_results.tsv', separator='\t', quote_char=None, infer_schema=False).fill_null('')
             .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    diff = base.join(sub, on=['s1', 'cand'], how='anti').height + sub.join(base, on=['s1', 'cand'], how='anti').height
    say(f'reproduction: {base.height:,} pairs vs submitted {sub.height:,}; differing pairs {diff:,}')
    # ---- 2. held-out tuning (half A) / reporting (half B)
    ev = oof.filter(pl.col('held_out'))
    halves = sorted(ev['heldout_half'].unique().to_list())
    A, B = ev.filter(pl.col('heldout_half') == halves[0]), ev.filter(pl.col('heldout_half') == halves[-1])
    nt_all = gt.group_by('s1').len().rename({'len': 'ntrue'})
    def nt(part):
        return part.select('s1').unique().join(nt_all, on='s1', how='left').with_columns(pl.col('ntrue').fill_null(0))
    lab = oof.select('s1', 'cand', 'label')
    best = {}
    for c in ('India', 'US'):
        a, ntA = A.filter(pl.col('country') == c), None
        ntA = nt(a)
        grid = []
        for dl in DELTAS:
            for bt in BETAS:
                grid.append((macro(decide(a.select('s1', 'cand', 'score', 'country'), {c: dl}, {c: bt}), lab, ntA, owned, 1.9), dl, bt))
        grid.sort(reverse=True)
        best[c] = grid[0]
        say(f'{c}: half-A best stressed F0.5 {grid[0][0]:.5f} at delta={grid[0][1]} beta={grid[0][2]} (baseline delta=0,beta=1: '
            f'{[g for g in grid if g[1] == 0 and g[2] == 1.0][0][0]:.5f})')
    rep = {}
    for c in ('India', 'US'):
        b = B.filter(pl.col('country') == c); ntB = nt(b)
        for tag, dl, bt in (('baseline', 0.0, 1.0), ('tuned', best[c][1], best[c][2])):
            pr = decide(b.select('s1', 'cand', 'score', 'country'), {c: dl}, {c: bt})
            rep[f'{c}_{tag}'] = dict(plain=round(macro(pr, lab, ntB, owned, 1.0), 5), stressed=round(macro(pr, lab, ntB, owned, 1.9), 5))
    say('HALF B (held out):', json.dumps(rep, indent=1))
    # ---- 3. France: label-free profile matching, never looser than India
    delta = {c: best[c][1] for c in best}; beta = {c: best[c][2] for c in best}
    dom = decide(test.filter(pl.col('country') != 'France').select('s1', 'cand', 'score', 'country'), delta, beta)
    target = profile(dom, test.filter(pl.col('country') != 'France').select('s1').unique())
    fr = test.filter(pl.col('country') == 'France').select('s1', 'cand', 'score', 'country'); frs = fr.select('s1').unique()
    floor = max(delta['India'], 0.0)   # unknown country: never looser than India's tuned setting nor than the 0.977 file itself
    curve = [(dl, *profile(decide(fr, {'France': dl}, {'France': max(beta['India'], 1.0)}), frs)) for dl in DELTAS if dl >= floor]
    e0, k0 = curve[0][1], curve[0][2]
    se, sk = max(abs(target[0] - e0), 1e-4), max(abs(target[1] - k0), 1e-3)
    delta['France'] = min(curve, key=lambda r: abs(r[1] - target[0]) / se + abs(r[2] - target[1]) / sk)[0]
    beta['France'] = max(beta['India'], 1.0)
    say(f'in-domain test profile {target}; France curve {curve}; France delta={delta["France"]}')
    # ---- 4. final decision + S9 France removals
    pred = decide(test.select('s1', 'cand', 'score', 'country'), delta, beta)
    s9 = (pl.read_csv('D:/Amazon ML/S9_new_minus_fr/matching_results.tsv', separator='\t', quote_char=None, infer_schema=False).fill_null('')
            .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    s9_removed = sub.join(s9, on=['s1', 'cand'], how='anti')
    pred = pred.join(s9_removed, on=['s1', 'cand'], how='anti')
    say(f'S11 pairs {pred.height:,} (0.977: {sub.height:,}; S9: {s9.height:,}); vs S9 +{pred.join(s9, on=["s1","cand"], how="anti").height:,} '
        f'-{s9.join(pred, on=["s1","cand"], how="anti").height:,}')
    os.makedirs(N + 'S11', exist_ok=True)
    ids = pl.read_csv('D:/Amazon ML/student_resource/dataset/test/test_source1.tsv', separator='\t', quote_char=None, infer_schema=False, columns=['entity_id'])
    g = pred.sort(['s1', 'cand']).group_by('s1', maintain_order=True).agg(pl.col('cand').str.join(','))
    out = ids.rename({'entity_id': 's1'}).join(g, on='s1', how='left').with_columns(pl.col('cand').fill_null(''))
    with open(N + 'S11/matching_results.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for a, b in out.iter_rows():
            f.write(f'{a}\t{b}\n')
    import shutil
    shutil.copyfile(N + 'candidate_pairs.tsv', N + 'S11/candidate_pairs.tsv')
    json.dump(dict(delta=delta, beta=beta, half_B=rep, reproduction_diff=diff), open(N + 'S11/tuning.json', 'w'), indent=1)
    say('S11 written')


if __name__ == '__main__':
    main()
