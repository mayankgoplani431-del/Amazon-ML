"""Step 2: multi-pass blocking over ALL Source-1 records of a split, written in chunks.

Passes (independent, unioned; per country label, open set):
  comb     word TF-IDF on name_core + addr_norm, S1 -> pool top-K
  addr     word TF-IDF on addr_norm,             S1 -> pool top-K
  name_ch  char-4gram TF-IDF on space-less core name (typos, domain-style names), S1 -> pool top-K
  rev      word TF-IDF name+addr, pool -> S1 top-K ("which S1 does this candidate look like most?")
  nameaddr TF-IDF where the whole core name is one token + address tokens (exact name + address), both directions
  k_name_num  exact key (country, core name, first house number), size-capped
  k_name      exact key (country, core name) for near-unique names (covers candidates without address)
Output: work/cache/cand_{split}/part_*.parquet with qi, pi, per-pass score/rank, n_passes.
"""
import os, sys, time, glob
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from common import CACHE, load_norm
from blocking import _topk, text_of

W = dict(token_pattern=r'\S+', max_df=20000)
CFG = {
    'comb': dict(kind='comb', vkw=W, K=50),
    'addr': dict(kind='addr', vkw=W, K=25),
    'name_ch': dict(kind='name_nospace', vkw=dict(analyzer='char', ngram_range=(4, 4), max_df=20000, min_df=2), K=12),
    'rev': dict(kind='comb', vkw=W, K=5, reverse=True),
    'nameaddr': dict(kind='nameaddr', vkw=W, K=20),
    'rev_na': dict(kind='nameaddr', vkw=W, K=3, reverse=True),
    'k_name_num': dict(key=['name_core', 'num1'], max_q=20, max_p=200),
    'k_name': dict(key=['name_core'], max_q=3, max_p=30),
    'k_sorted_num': dict(key=['name_sorted', 'num1'], max_q=20, max_p=200),   # word-order transpositions
    'k_nosp_num': dict(key=['name_nosp', 'num1'], max_q=20, max_p=200),       # glued / domain-style names
}
NAME_KEYS = [pl.col('name_core').str.split(' ').list.sort().list.join(' ').alias('name_sorted'),
             pl.col('name_core').str.replace_all(' ', '').alias('name_nosp')]
NUM1 = pl.col('addr_nums').str.split(' ').list.first().fill_null('').alias('num1')


def key_pass(name, c, qm, pm):
    """Exact-key join; keys shared by too many records are skipped (not discriminative)."""
    k = c['key']
    qk = qm.filter(pl.all_horizontal([pl.col(x) != '' for x in k])).select(['idx_q'] + k)
    pk = pm.filter(pl.all_horizontal([pl.col(x) != '' for x in k])).select(['idx_p'] + k)
    qk = qk.filter(pl.len().over(k) <= c['max_q'])
    pk = pk.with_columns(pl.len().over(k).alias('_np')).filter(pl.col('_np') <= c['max_p'])
    j = qk.join(pk, on=k)
    return j.select(pl.col('idx_q').alias('qi'), pl.col('idx_p').alias('pi'),
                    (1.0 / pl.col('_np')).cast(pl.Float32).alias(f'{name}_score'),
                    pl.lit(0, pl.Int16).alias(f'{name}_rank'))
CHUNK = 100_000


def load_split(split, cols=None):
    q = load_norm(split, 1).drop('idx').with_row_index('idx_q')
    p = pl.concat([load_norm(split, 2), load_norm(split, 3)]).drop('idx').with_row_index('idx_p')
    if cols:
        q, p = q.select(['idx_q'] + cols), p.select(['idx_p'] + cols)
    else:
        num1 = pl.col('addr_nums').str.split(' ').list.first().fill_null('').alias('num1')
        q, p = q.with_columns(num1), p.with_columns(num1)
    return q, p


def run(split, cfg=CFG, out_dir=None):
    out_dir = out_dir or os.path.join(CACHE, f'cand_{split}')
    os.makedirs(out_dir, exist_ok=True)
    q, p = load_split(split, ['country_n', 'name_core', 'addr_norm', 'addr_nums'])
    q, p = q.with_columns(NUM1, *NAME_KEYS), p.with_columns(NUM1, *NAME_KEYS)
    part = 0
    for ctry in q['country_n'].unique().sort().to_list():
        qm = q.filter(pl.col('country_n') == ctry)
        pm = p.filter(pl.col('country_n') == ctry)
        if pm.height == 0:
            continue
        qidx, pidx = qm['idx_q'].to_numpy(), pm['idx_p'].to_numpy()
        mats = {}
        for name, c in cfg.items():
            if 'key' in c:
                continue
            key = (c['kind'], str(c['vkw']))
            if key not in mats:
                t = time.time()
                v = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, **c['vkw'])
                v.fit(text_of(pm, c['kind']) + text_of(qm, c['kind']))
                A, B = v.transform(text_of(qm, c['kind'])), v.transform(text_of(pm, c['kind']))
                del v
                # store the pool side pre-transposed for forward passes; S1 side transposed for reverse
                mats[key] = (A, B, B.T.tocsr(), A.T.tocsr() if any(
                    x.get('reverse') and x['kind'] == c['kind'] and x['vkw'] == c['vkw'] for x in cfg.values()) else None)
                print(f'[{ctry}] vectorized {key[0]} {time.time() - t:.0f}s', flush=True)
        # reverse passes over the whole pool of this country (pool rows query S1 rows) + exact-key passes
        side = {}
        for name, c in cfg.items():
            t = time.time()
            if 'key' in c:
                side[name] = key_pass(name, c, qm, pm)
            elif c.get('reverse'):
                A, B, BT, AT = mats[(c['kind'], str(c['vkw']))]
                fr = []
                for s in range(0, B.shape[0], CHUNK * 4):
                    r, cc, sc, rk = _topk(B[s:s + CHUNK * 4], AT, c['K'])
                    fr.append(pl.DataFrame({'qi': qidx[cc], 'pi': pidx[r + s], f'{name}_score': sc, f'{name}_rank': rk}))
                side[name] = pl.concat(fr)
            else:
                continue
            print(f'[{ctry}] side pass {name}: {side[name].height} pairs {time.time() - t:.0f}s', flush=True)
        for k in mats:   # pool-side matrix only needed transposed from here on
            mats[k] = (mats[k][0], None, mats[k][2], None)
        for s in range(0, qm.height, CHUNK):
            t = time.time()
            res = None
            for name, c in cfg.items():
                if c.get('reverse') or 'key' in c:
                    continue
                A, B, BT, AT = mats[(c['kind'], str(c['vkw']))]
                r, cc, sc, rk = _topk(A[s:s + CHUNK], BT, c['K'])
                f = pl.DataFrame({'qi': qidx[r + s], 'pi': pidx[cc], f'{name}_score': sc, f'{name}_rank': rk})
                res = f if res is None else res.join(f, on=['qi', 'pi'], how='full', coalesce=True)
            chunk_q = pl.Series(qidx[s:s + CHUNK]).implode()
            for name, fr in side.items():
                res = res.join(fr.filter(pl.col('qi').is_in(chunk_q)), on=['qi', 'pi'], how='full', coalesce=True)
            score_cols = [f'{n}_score' for n in cfg]
            res = res.with_columns(pl.sum_horizontal([pl.col(x).is_not_null() for x in score_cols]).alias('n_passes'))
            res.write_parquet(os.path.join(out_dir, f'part_{part:04d}.parquet'))
            print(f'[{ctry}] chunk {s // CHUNK} pairs={res.height} ({res.height / min(CHUNK, qm.height - s):.1f}/S1) {time.time() - t:.0f}s', flush=True)
            part += 1


def load(split):
    return pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(CACHE, f'cand_{split}', 'part_*.parquet')))])


if __name__ == '__main__':
    run(sys.argv[1])
