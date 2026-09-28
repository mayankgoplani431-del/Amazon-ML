"""Adds ONE blocking pass to an existing blocking union without redoing the others (memory: one TF-IDF kind only).
Runs candidates.run with just that pass into cand_<pass>_<split>/, then merges part i into cand_<split>/part i
(both runs iterate countries and 100k-S1 chunks in the same order, so part i covers the same S1 rows; asserted).
Usage (ER_WORK = work dir whose cand_<split> parts get extended):  python add_pass.py <split> [pass_name]
"""
import os, sys, glob
import polars as pl
import candidates
from candidates import CFG
from common import CACHE


def main(split, name='comb_ch3'):
    extra_dir = os.path.join(CACHE, f'cand_{name}_{split}')
    if not glob.glob(os.path.join(extra_dir, 'part_*.parquet')):
        candidates.run(split, {name: CFG[name]}, out_dir=extra_dir)
    base_parts = sorted(glob.glob(os.path.join(CACHE, f'cand_{split}', 'part_*.parquet')))
    extra_parts = sorted(glob.glob(os.path.join(extra_dir, 'part_*.parquet')))
    assert len(base_parts) == len(extra_parts), (len(base_parts), len(extra_parts))
    score_cols = [f'{n}_score' for n in CFG]
    added = total_true_new = 0
    for fb, fe in zip(base_parts, extra_parts):
        b = pl.read_parquet(fb)
        if f'{name}_score' in b.columns:   # already merged (rerun-safe)
            continue
        e = pl.read_parquet(fe).drop('n_passes')
        assert set(e['qi'].unique().to_list()) <= set(b['qi'].unique().to_list()), f'{fb}: chunks do not align'
        m = b.drop('n_passes').join(e, on=['qi', 'pi'], how='full', coalesce=True)
        m = m.with_columns(pl.sum_horizontal([pl.col(c).is_not_null() for c in score_cols if c in m.columns]).alias('n_passes'))
        added += m.height - b.height
        m.write_parquet(fb + '.tmp')
        os.replace(fb + '.tmp', fb)
        print(f'{os.path.basename(fb)}: +{m.height - b.height} new pairs', flush=True)
    print(f'{split}: {added} new candidate pairs merged', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
