"""Inference on the test split with the models trained by run_train.py.
Writes output/matching_results.tsv and output/candidate_pairs.tsv.
Usage: python run_test.py [out_dir] [top_n]
"""
import sys, os, json, time
import polars as pl
import candidates, pipeline as P
from run_train import apply_rule


def write_lists(ids_q: pl.DataFrame, pairs: pl.DataFrame, p_ids: pl.DataFrame, col: str, path: str):
    """One row per S1 (file order), comma-joined S2/S3 ids, empty when none. No quoting, tab-separated."""
    g = (pairs.select('qi', 'pi').unique().join(p_ids, on='pi')
              .sort(['qi', 'cand_id']).group_by('qi', maintain_order=True).agg(pl.col('cand_id').str.join(',').alias(col)))
    out = ids_q.join(g, on='qi', how='left').sort('qi').select('source1_entity_id', pl.col(col).fill_null(''))
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'source1_entity_id\t{col}\n')
        for a, b in out.iter_rows():
            f.write(f'{a}\t{b}\n')
    return out


def main(out_dir, top_n=20):
    t = time.time()
    q, p = candidates.load_split('test')
    if not P.parts('test', 'cand'):
        candidates.run('test')
    P.run_stage0('test', q, p, top_n=top_n)
    need = ['entity_id', 'country_n', 'name_norm', 'name_core', 'name_alias', 'addr_norm', 'addr_nums', 'addr_state', 'addr_post']
    q, p = q.select(['idx_q'] + need), p.select(['idx_p'] + need)   # drop raw text before the memory-heavy feature step
    P.run_features('test', q, p)
    if os.path.exists(os.path.join(P.CACHE, 'FEATURES_ONLY')):   # orchestration: stop after test features
        print('features only: done', flush=True)
        return
    ids_q = q.select(pl.col('idx_q').alias('qi'), pl.col('entity_id').alias('source1_entity_id'))
    p_ids = p.select(pl.col('idx_p').alias('pi'), pl.col('entity_id').alias('cand_id'))
    del q, p
    s1 = P.predict_stage('test', 'stage1')
    s1.write_parquet(os.path.join(P.CACHE, 'test_stage1.parquet'))   # raw scores: decision rules can be re-applied later
    if os.environ.get('STAGES', '2') == '1':   # stage-1 model + its own tuned decision rule
        rule = json.load(open(os.path.join(P.MODELS, 'decision_stage1.json')))
        pred = apply_rule(s1, 'stage1', rule)
    else:
        ctx = P.group_stats(s1, 'stage1', 'p1')
        s2 = P.predict_stage('test', 'stage2', extra=ctx)
        s2.write_parquet(os.path.join(P.CACHE, 'test_stage2.parquet'))
        ent_cfg = os.path.join(P.MODELS, 'entity_decision.json')
        if os.path.exists(ent_cfg):   # entity-level empty/count model + expected-F0.5 rule
            import entity as E
            rule = json.load(open(ent_cfg))
            allq = ids_q.select(pl.col('qi').cast(pl.UInt32))
            s2c = s2.with_columns(pl.col('qi').cast(pl.UInt32))
            ent = E.fit_predict(E.aggregates(s2c, 'stage2', allq), None, 'entity_stage2')
            pred = E.decide(s2c, 'stage2', ent, rule['empty_scale'], rule['n_blend'])
        else:
            rule = json.load(open(os.path.join(P.MODELS, 'decision.json')))
            pred = apply_rule(s2, 'stage2', rule)
    cand = pl.concat([pl.read_parquet(f, columns=['qi', 'pi']) for f in P.parts('test', 'pr')])
    os.makedirs(out_dir, exist_ok=True)
    write_lists(ids_q, cand, p_ids, 'candidate_entity_ids', os.path.join(out_dir, 'candidate_pairs.tsv'))
    m = write_lists(ids_q, pred, p_ids, 'matched_entity_ids', os.path.join(out_dir, 'matching_results.tsv'))
    print(f'rule={rule} predicted pairs={pred.height} S1 with matches={(m["matched_entity_ids"] != "").sum()}/{m.height} '
          f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'),
         int(sys.argv[2]) if len(sys.argv) > 2 else 20)
