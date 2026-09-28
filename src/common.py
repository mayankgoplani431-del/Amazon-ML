"""Shared paths and I/O helpers. Override locations with ER_DATA / ER_WORK env vars."""
import os
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get('ER_DATA', os.path.join(ROOT, 'data'))      # folder containing train/ and test/ (NOT committed)
WORK = os.environ.get('ER_WORK', os.path.join(ROOT, 'work'))      # caches, models, reports (NOT committed)
CACHE = os.path.join(WORK, 'cache')
os.makedirs(CACHE, exist_ok=True)


def read_tsv(path):
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False).fill_null('')


def load_norm(split, src):
    return pl.read_parquet(os.path.join(CACHE, f'{split}_s{src}.parquet'))


def load_gt():
    """Long-format ground truth: (s1, cand) positive pairs, plus the list of all S1 ids."""
    gt = read_tsv(os.path.join(DATA, 'train', 'train_ground_truth.tsv'))
    pairs = (gt.with_columns(pl.col('matched_entity_ids').str.split(',').alias('cand'))
               .explode('cand').filter(pl.col('cand') != '')
               .select(pl.col('source1_entity_id').alias('s1'), 'cand'))
    return pairs
