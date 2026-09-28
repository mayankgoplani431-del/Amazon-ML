"""Disk hygiene: delete intermediates that later stages no longer read. All are regenerable.

  cand_{split} : raw blocking union      -> needed only to rebuild c0 (kept: slow to rebuild, ~40 min)
  c0_{split}   : cheap features, unpruned -> needed only to re-prune; deleted once pr_{split} exists (~5 min to rebuild)
  pr_{split}   : final candidate set      -> kept (candidate_pairs.tsv + features are built from it)
  feat_{split} : pair features            -> kept while models are trained/applied
Run standalone to see sizes and free space: python cleanup.py [--all-intermediate]
"""
import os, sys, glob, shutil
from common import CACHE, WORK


def size_gb(path):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, '**', '*'), recursive=True) if os.path.isfile(f)) / 1e9


def drop(name, split):
    p = os.path.join(CACHE, f'{name}_{split}')
    if os.path.isdir(p):
        gb = size_gb(p)
        shutil.rmtree(p, ignore_errors=True)
        print(f'cleanup: removed {name}_{split} ({gb:.1f} GB)', flush=True)


def after_prune(split):
    if glob.glob(os.path.join(CACHE, f'pr_{split}', 'part_*.parquet')):
        drop('c0', split)


def misc():
    for f in glob.glob(os.path.join(CACHE, '*.pkl')) + glob.glob(os.path.join(CACHE, '*.tmp')):
        os.remove(f)
    for d in glob.glob(os.path.join(os.path.dirname(__file__), '__pycache__')):
        shutil.rmtree(d, ignore_errors=True)


def report():
    for d in sorted(glob.glob(os.path.join(CACHE, '*'))):
        print(f'  {os.path.basename(d):28s} {size_gb(d) if os.path.isdir(d) else os.path.getsize(d) / 1e9:6.2f} GB')
    print(f'  free on disk: {shutil.disk_usage(WORK).free / 1e9:.1f} GB')


if __name__ == '__main__':
    for s in ('train', 'test'):
        after_prune(s)
        if '--all-intermediate' in sys.argv:
            drop('cand', s)
    misc()
    report()
