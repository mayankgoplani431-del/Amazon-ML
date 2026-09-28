"""Label-free go/no-go for a new submission vs the current best: profile the ADDED and REMOVED pairs.

Estimated precision of a pair set from its house-number-mismatch rate m (mixture of two reference profiles
measured on train with labels): prec ~ (m_fp - m) / (m_fp - m_true), where m_true = mismatch rate of TRUE train
pairs and m_fp = mismatch rate of the model's train FALSE POSITIVES (OOF + its decision rule), per country.
F0.5 break-even (audit_consensus.md): adding a set helps if its precision > ~0.77, removing helps if < ~0.77.
France has no train reference: its rows are reported with the US+India pooled reference (indicative only).
Usage: python diffprof.py <best_matching.tsv> <new_matching.tsv>
"""
import os, sys, json
import polars as pl
import candidates, pipeline as P
from diag import COLS, pair_frame, FLAGS, lean
from run_train import apply_rule
from common import read_tsv, load_gt


def pairs_of(path, q, p):
    m = read_tsv(path).select(pl.col('source1_entity_id').alias('entity_id'), pl.col('matched_entity_ids').str.split(',').alias('ids'))
    return (m.explode('ids').filter(pl.col('ids') != '').join(q.select('qi', 'entity_id'), on='entity_id')
             .join(p.select('pi', pl.col('entity_id').alias('ids')), on='ids').select('qi', 'pi'))


def rates(d):
    return d.group_by('country_n').agg(pl.len().alias('pairs'), *[pl.col(f).mean().round(4) for f in FLAGS]).sort('country_n')


def main(best, new):
    q, p = lean('test')
    a, b = pairs_of(best, q, p), pairs_of(new, q, p)
    added, removed = b.join(a, on=['qi', 'pi'], how='anti'), a.join(b, on=['qi', 'pi'], how='anti')
    qq, pp = q.drop('entity_id'), p.drop('entity_id')
    # train references
    tq, tp = lean('train')
    gt = (load_gt().join(tq.select(pl.col('entity_id').alias('s1'), 'qi'), on='s1')
                   .join(tp.select(pl.col('entity_id').alias('cand'), 'pi'), on='cand').select('qi', 'pi'))
    gt = gt.sample(min(gt.height, 1_000_000), seed=0)
    tq, tp = tq.drop('entity_id'), tp.drop('entity_id')
    col = 'stage1' if os.environ.get('STAGES', '2') == '1' else 'stage2'   # same switch as run_test.py
    oof = pl.read_parquet(os.path.join(P.CACHE, f'oof_{col}.parquet'))
    rule = json.load(open(os.path.join(P.MODELS, 'decision_stage1.json' if col == 'stage1' else 'decision.json')))
    fp = apply_rule(oof, col, rule).filter(pl.col('label') == 0).select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
    rt, rf = rates(pair_frame(tq, tp, gt)), rates(pair_frame(tq, tp, fp))
    ref = {r['country_n']: (r['num1_diff'], f['num1_diff']) for r, f in zip(rt.iter_rows(named=True), rf.iter_rows(named=True))}
    pooled = (sum(v[0] for v in ref.values()) / len(ref), sum(v[1] for v in ref.values()) / len(ref))
    print(f'reference TRUE train pairs:\n{rt}\nreference model FALSE POSITIVES on train:\n{rf}')
    print(f'\n{best} -> {new}: +{added.height} added, -{removed.height} removed')
    for tag, s in (('ADDED', added), ('REMOVED', removed)):
        r = rates(pair_frame(qq, pp, s))
        print(f'-- {tag}\n{r}')
        for row in r.iter_rows(named=True):
            mt, mf = ref.get(row['country_n'], pooled)
            prec = (mf - row['num1_diff']) / max(mf - mt, 1e-6)
            verdict = ('helps' if prec > 0.77 else 'HURTS') if tag == 'ADDED' else ('helps' if prec < 0.77 else 'HURTS')
            print(f'   {row["country_n"]:8s} n={row["pairs"]:>8}  est. precision {min(max(prec, 0), 1):.2f}  -> {tag.lower()} {verdict}'
                  + ('  (France: pooled reference, indicative)' if row['country_n'] not in ref else ''))


if __name__ == '__main__':
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    main(sys.argv[1], sys.argv[2])
