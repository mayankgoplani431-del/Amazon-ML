"""Step 1: normalize all source files once and cache as parquet (work/cache/{split}_s{i}.parquet)."""
import os, sys, time
from multiprocessing import Pool
import polars as pl
from common import DATA, CACHE, read_tsv
from normalize import normalize


def _chunk(df):
    return normalize(df)


def run(split, src, workers=6):
    out = os.path.join(CACHE, f'{split}_s{src}.parquet')
    if os.path.exists(out):
        return
    t = time.time()
    df = read_tsv(os.path.join(DATA, split, f'{split}_source{src}.tsv'))
    n = df.height
    step = (n + workers * 4 - 1) // (workers * 4)
    with Pool(workers) as p:
        parts = p.map(_chunk, [df.slice(i, step) for i in range(0, n, step)])
    res = pl.concat(parts).with_row_index('idx')
    res.write_parquet(out + '.tmp')
    os.replace(out + '.tmp', out)
    print(split, src, res.shape, f'{time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    import translit
    if not os.path.exists(translit.PATH):
        translit.build()   # learned from train ground truth only
    for split in sys.argv[1:] or ['train', 'test']:
        for src in (1, 2, 3):
            run(split, src)
