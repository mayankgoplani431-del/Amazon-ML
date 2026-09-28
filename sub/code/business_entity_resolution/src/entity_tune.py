"""Train the entity-level model on stage-2 OOF scores and pick its decision parameters (validation only)."""
import os, json
import polars as pl
import pipeline as P, decide as D, entity as E
from evaluate import log_experiment

s = pl.read_parquet(os.path.join(P.CACHE, 'oof_stage2.parquet')).with_columns(pl.col('qi').cast(pl.UInt32))
gt = pl.read_parquet(os.path.join(P.CACHE, 'truth_counts.parquet'))
n = int(s['qi'].max()) + 1
allq = pl.DataFrame({'qi': pl.arange(0, 2206821, eager=True).cast(pl.UInt32)})
nt = allq.join(gt, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
ent = E.fit_predict(E.aggregates(s.drop('label'), 'stage2', allq), nt, 'entity_stage2')
best = (0, None)
for es in [0.9, 1.0, 1.1, 1.2, 1.3, 1.45]:
    for nb in [0.5, 0.8, 1.0]:
        f = D.macro_f05(E.decide(s, 'stage2', ent, es, nb), nt)['macro_f05']
        print(f'empty_scale={es} n_blend={nb}: {f:.5f}', flush=True)
        if f > best[0]:
            best = (f, dict(empty_scale=es, n_blend=nb))
pred = E.decide(s, 'stage2', ent, **best[1])
cand = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 'label']) for f in P.parts('train', 'pr')]).with_columns(pl.col('qi').cast(pl.UInt32))
sc = D.macro_f05(pred, nt, cand)
sc['fold_f05'] = {k: D.macro_f05(pred.filter(P.fold_of(pl.col('qi')) == k), nt.filter(P.fold_of(pl.col('qi')) == k))['macro_f05'] for k in range(P.NF)}
print(log_experiment('E03a_stage2_entity', sc, dict(decision=best[1], model='stage1+stage2+entity')))
json.dump(best[1], open(os.path.join(P.MODELS, 'entity_decision.json'), 'w'))
