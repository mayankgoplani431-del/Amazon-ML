"""Experiment (a2): char-3gram name+address blocking with common-3gram caps (max_df) -> recall of the missed true pairs
vs compute cost per S1 (the uncapped pass costs ~80 ms/S1 = ~50 h at full scale). Same sample design as exp_lsh.py.
Usage (ER_WORK=work_e07): python exp_lsh2.py <out_json> [n_per_country]
"""
import os, sys, json, glob, time
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn
from common import CACHE, load_gt

t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
CAPS = (5000, 20000, 100000)
KS = (12, 25)
COLS = ['entity_id', 'country_n', 'name_core', 'addr_norm']
txt = lambda df: (df['name_core'].str.replace_all(' ', '') + ' ' + df['addr_norm']).to_list()


def main(out_json, n_per=3000):
    rd = lambda s: pl.read_parquet(os.path.join(CACHE, f'train_s{s}.parquet'), columns=COLS)
    q, p = rd(1).with_row_index('qi'), pl.concat([rd(2), rd(3)]).with_row_index('pi')
    cast = lambda d: d.with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
    truth = cast(load_gt().join(q.select(pl.col('entity_id').alias('s1'), 'qi'), on='s1')
                          .join(p.select(pl.col('entity_id').alias('cand'), 'pi'), on='cand').select('qi', 'pi'))
    parts = sorted(glob.glob(os.path.join(CACHE, 'cand_train', 'part_*.parquet')))
    in_union = pl.concat([truth.join(cast(pl.read_parquet(f, columns=['qi', 'pi'])), on=['qi', 'pi'], how='semi') for f in parts]).unique()
    miss = truth.join(in_union, on=['qi', 'pi'], how='anti').join(q.select('qi', 'country_n'), on='qi')
    res = {}
    for ctry in miss['country_n'].unique().sort().to_list():
        s1s = miss.filter(pl.col('country_n') == ctry)['qi'].unique().sort().sample(n_per, seed=0)   # same sample as exp_lsh.py
        tgt = miss.filter(pl.col('qi').is_in(s1s.implode())).select('qi', 'pi')
        qm, pm = q.filter(pl.col('qi').is_in(s1s.implode())), p.filter(pl.col('country_n') == ctry)
        un = pl.concat([cast(pl.read_parquet(f, columns=['qi', 'pi'])).filter(pl.col('qi').is_in(s1s.implode())) for f in parts])
        v = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), binary=True, use_idf=False, norm=None, min_df=2, dtype=np.float32)
        B = v.fit_transform(txt(pm)).tocsc()
        A = v.transform(txt(qm)).tocsc()
        df = np.diff(B.indptr)
        res[ctry] = {'sampled_s1': qm.height, 'missed_pairs_in_sample': tgt.height}
        for cap in CAPS:
            keep = np.flatnonzero(df <= cap)
            Bc, Ac = normalize(B[:, keep].tocsr()), normalize(A[:, keep].tocsr())
            t = time.time()
            C = sp_matmul_topn(Ac, Bc.T.tocsr(), top_n=max(KS), n_threads=14, sort=True).tocsr()
            ms = 1000 * (time.time() - t) / qm.height
            rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
            rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], np.diff(C.indptr))
            hits = cast(pl.DataFrame({'qi': qm['qi'].to_numpy()[rows], 'pi': pm['pi'].to_numpy()[C.indices], 'rank': rank.astype(np.int32)}))
            found, new = tgt.join(hits, on=['qi', 'pi']), hits.join(un, on=['qi', 'pi'], how='anti')
            r = {f'recall@{k}': round(found.filter(pl.col('rank') < k).height / max(tgt.height, 1), 4) for k in KS}
            r.update({f'new_per_s1@{k}': round(new.filter(pl.col('rank') < k).height / qm.height, 1) for k in KS})
            r.update(ms_per_s1=round(ms, 2), kept_3grams=int(len(keep)), dropped_3grams=int(len(df) - len(keep)))
            res[ctry][f'max_df={cap}'] = r
            say(ctry, cap, r)
    json.dump(res, open(out_json, 'w'), indent=1)
    say('done')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
