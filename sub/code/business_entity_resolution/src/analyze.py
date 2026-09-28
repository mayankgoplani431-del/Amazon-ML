"""Error analysis on out-of-fold predictions: false positives by cause, false negatives split into
blocking failures (never a candidate) vs scoring failures (candidate, rejected by model/decision).
Usage: python analyze.py <oof parquet> <score col>   -> work/reports/errors_<col>.txt
"""
import sys, os, json
import polars as pl
import candidates, pipeline as P, decide as D
from run_train import apply_rule, s1_truth
from common import WORK


def main(path, col):
    s = pl.read_parquet(path)
    q, p = candidates.load_split('train')
    lab = P.labels(q, p)
    rule = json.load(open(os.path.join(P.MODELS, 'decision.json'))) if col == 'stage2' else \
        {'rule': 'expected_f', 'miss': 0.2, 'empty_bias': 1.2, 'exclusive': True}
    pred = apply_rule(s, col, rule)
    cols = ['idx_q', 'entity_id', 'business_name', 'business_address', 'country_n', 'name_core', 'addr_norm', 'addr_nums']
    Q = q.select(cols).rename({'idx_q': 'qi'})
    Pp = p.select([c if c != 'idx_q' else c for c in ['idx_p'] + cols[1:]]).rename({'idx_p': 'pi'})
    out = []
    fp = pred.filter(pl.col('label') == 0).join(Q, on='qi').join(Pp, on='pi', suffix='_c')
    ntrue = s1_truth(q, lab)
    fp = fp.join(ntrue, on='qi')
    out.append(f'FALSE POSITIVES: {fp.height}  (on singleton S1: {(fp["ntrue"] == 0).sum()})')
    out.append('by country: ' + str(fp['country_n'].value_counts().rows()))
    cause = fp.with_columns(
        pl.when(pl.col('name_core') == pl.col('name_core_c')).then(pl.lit('same name'))
          .when(pl.col('addr_norm') == pl.col('addr_norm_c')).then(pl.lit('same address'))
          .when(pl.col('addr_norm_c') == '').then(pl.lit('cand no address'))
          .when(pl.col('addr_nums').str.split(' ').list.first() != pl.col('addr_nums_c').str.split(' ').list.first()).then(pl.lit('house number differs'))
          .otherwise(pl.lit('other')).alias('cause'))
    out.append('by cause: ' + str(cause['cause'].value_counts().sort('count', descending=True).rows()))
    # was the true owner of this candidate a different S1? (steal) or is the candidate a distractor?
    owner = lab.rename({'qi': 'owner'}).drop('label')
    st = fp.join(owner, on='pi', how='left')
    out.append(f'candidate truly belongs to another S1: {st["owner"].is_not_null().sum()}, distractor (no S1): {st["owner"].is_null().sum()}')
    for r in cause.sort(col, descending=True).head(40).select(col, 'cause', 'business_name', 'business_address', 'business_name_c', 'business_address_c').rows():
        out.append(f'  {r[0]:.3f} [{r[1]}] {r[2]} | {r[3]}  <->  {r[4]} | {r[5]}')
    # false negatives
    cand = s.select('qi', 'pi').with_columns(pl.lit(1).alias('in_cand'))
    fn = lab.join(pred.select('qi', 'pi', pl.lit(1).alias('pred')), on=['qi', 'pi'], how='left').filter(pl.col('pred').is_null())
    fn = fn.join(cand, on=['qi', 'pi'], how='left').join(s.select('qi', 'pi', col), on=['qi', 'pi'], how='left')
    fn = fn.join(Q, on='qi').join(Pp, on='pi', suffix='_c')
    blk = fn.filter(pl.col('in_cand').is_null())
    scr = fn.filter(pl.col('in_cand').is_not_null())
    out.append(f'\nFALSE NEGATIVES: {fn.height}  blocking failures: {blk.height}  scoring failures: {scr.height}')
    out.append('blocking failures by country: ' + str(blk['country_n'].value_counts().rows()))
    out.append('blocking failures cand addr empty: %.3f  cand name non-ascii: %.3f' % (
        (blk['addr_norm_c'] == '').mean(), blk['business_name_c'].str.contains(r'[^\x00-\x7f]').mean()))
    out.append('scoring failures by country: ' + str(scr['country_n'].value_counts().rows()))
    for tag, df in [('BLOCKING', blk), ('SCORING', scr)]:
        out.append(f'-- sample {tag} failures')
        for r in df.sample(min(30, df.height), seed=0).select(col, 'business_name', 'business_address', 'business_name_c', 'business_address_c').rows():
            out.append(f'  {r[0] if r[0] is not None else -1:.3f} {r[1]} | {r[2]}  <->  {r[3]} | {r[4]}')
    txt = '\n'.join(out)
    open(os.path.join(WORK, 'reports', f'errors_{col}.txt'), 'w', encoding='utf-8').write(txt)
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
