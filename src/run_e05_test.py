"""E05 on test: union candidates (E04 top-20 ∪ legacy top-20 ∪ S3's actual candidate file) -> E04 + E02 + S3 votes
-> ensemble rule selected on train OOF (models/e05_rule.json) -> output + hard go/no-go checks.
Usage: python run_e05_test.py <out_dir> <s3_candidate_pairs.tsv> <s4_matching_results.tsv>
"""
import os, sys, json, time
import polars as pl
import candidates, pipeline as P, e05
from run_train import apply_rule
from run_test import write_lists

t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
out_dir, s3_cand, s4_match = sys.argv[1], sys.argv[2], sys.argv[3]
rd = lambda f: pl.read_csv(f, separator='\t', quote_char=None, infer_schema=False).fill_null('')
explode = lambda df, col: df.rename({col: 'cand_id'}).with_columns(pl.col('cand_id').str.split(',')).explode('cand_id').filter(pl.col('cand_id') != '')

q, p = candidates.load_split('test')
ids_q = q.select(pl.col('idx_q').alias('qi'), pl.col('entity_id').alias('source1_entity_id'))
p_ids = p.select(pl.col('idx_p').alias('pi'), pl.col('entity_id').alias('cand_id'))
if not P.parts('test', 'feat'):
    extra = (explode(rd(s3_cand), 'candidate_entity_ids').join(ids_q, on='source1_entity_id').join(p_ids, on='cand_id')
             .select(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32)))
    say('S3 candidate pairs forced into legacy pool:', extra.height)
    e05.run_stage0_union('test', q, p, extra_pairs=extra)
    e05.run_features_union('test', q, p)
ctry = q.select(pl.col('idx_q').alias('qi'), 'country_n')
del q, p

s1 = e05.predict_e04('test', 'stage1_e04')
ctx1 = P.group_stats(s1.select('qi', 'pi', 'stage1_e04'), 'stage1_e04', 'p1')
s2 = e05.predict_e04('test', 'stage2_e04', extra=ctx1)
v04 = apply_rule(s2.rename({'stage2_e04': 'p'}), 'p', json.load(open(os.path.join(P.MODELS, 'decision_e04.json')))).select('qi', 'pi')
v = {n: e05.legacy_votes(e05.score_legacy('test', n), n) for n in ('E02', 'S3')}
rule = json.load(open(os.path.join(P.MODELS, 'e05_rule.json')))['rule']
pred = e05.ensemble(v04, v['E02'], v['S3'], rule)
cand = pl.concat([pl.read_parquet(f, columns=['qi', 'pi']) for f in P.parts('test', 'pr')])
say('rule', rule, 'predicted pairs', pred.height, 'candidates', cand.height)

# ---- GO/NO-GO #1 (hard fail, before anything else): every match is a candidate of the same S1
bad = pred.join(cand, on=['qi', 'pi'], how='anti').height
assert bad == 0, f'NO-GO: {bad} matched pairs are not in candidate_pairs for their S1'
say('check 1 PASS: matches_not_in_candidates == 0 (per S1)')

os.makedirs(out_dir, exist_ok=True)
write_lists(ids_q, cand, p_ids, 'candidate_entity_ids', os.path.join(out_dir, 'candidate_pairs.tsv'))
write_lists(ids_q, pred, p_ids, 'matched_entity_ids', os.path.join(out_dir, 'matching_results.tsv'))

# ---- sanity: re-scored S3 votes vs the actually submitted S3 predictions
s3_actual = explode(rd(os.path.join(os.path.dirname(s3_cand), 'matching_results.tsv')), 'matched_entity_ids').join(ids_q, on='source1_entity_id').join(p_ids, on='cand_id').select('qi', 'pi')
say('re-scored S3 vs submitted S3: overlap', round(v['S3'].join(s3_actual, on=['qi', 'pi'], how='semi').height / max(s3_actual.height, 1), 4))

# ---- per-country stats + diff vs S4 with house-number profile
from common import load_norm
nums = lambda src: None
s4 = explode(rd(s4_match), 'matched_entity_ids').join(ids_q, on='source1_entity_id').join(p_ids, on='cand_id').select('qi', 'pi')
feat = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 'num_first_eq']) for f in P.parts('test', 'feat')])
def prof(df, tag):
    d = df.join(feat, on=['qi', 'pi'], how='left').join(ctry, on='qi')
    print(tag, d.group_by('country_n').agg(pl.len(), (pl.col('num_first_eq') == 0).mean().round(3).alias('num_mismatch')).sort('country_n').rows(), flush=True)
prof(pred, 'E05 predictions      ')
prof(pred.join(s4, on=['qi', 'pi'], how='anti'), 'ADDED vs S4          ')
prof(s4.join(pred, on=['qi', 'pi'], how='anti'), 'REMOVED vs S4        ')
for n, vv in (('E04', v04), ('E02', v['E02']), ('S3', v['S3'])):
    d = vv.join(ctry, on='qi')
    agree = d.join(pred, on=['qi', 'pi'], how='semi').group_by('country_n').len().rename({'len': 'in_final'})
    print(f'{n} agreement with final by country', d.group_by('country_n').len().join(agree, on='country_n').with_columns((pl.col('in_final') / pl.col('len')).round(3).alias('frac')).sort('country_n').rows(), flush=True)
say('done')
