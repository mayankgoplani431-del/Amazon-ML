"""Experiment (a): would an extra fuzzy (MinHash/LSH-style) blocking pass recover the true pairs blocking misses?

Measured on TRAIN (labels) with the E07 blocking union (ER_WORK=work_e07):
  1. missed = true pairs not in the blocking union (never reach any model) + union pairs pruned by stage 0.
  2. Profile of the missed pairs (country, char-3gram Jaccard of core names, empty address, non-ASCII name).
  3. For a random sample of S1 with missed pairs: exact top-K search over the SAME-country pool by char-3gram set
     similarity (binary cosine = |A∩B|/sqrt(|A||B|), a monotone proxy of the Jaccard that MinHash/LSH estimates)
     on (i) core name, (ii) core name + address. Exact search = an UPPER BOUND on what LSH (approximate) retrieves.
     Reports recall of the missed pairs at K and the number of NEW candidates per S1 (volume cost).
Usage: python exp_lsh.py <out_json> [n_sample_per_country]
"""
import os, sys, json, glob, time
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import CACHE, load_gt

t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
KS = (12, 25, 50, 100, 200)
COLS = ['entity_id', 'country_n', 'business_name', 'name_core', 'addr_norm']


def grams(s):
    s = s.replace(' ', '')
    return {s[i:i + 3] for i in range(max(len(s) - 2, 1))} if s else set()


def main(out_json, n_per=3000):
    rd = lambda s: pl.read_parquet(os.path.join(CACHE, f'train_s{s}.parquet'), columns=COLS)
    q, p = rd(1).with_row_index('qi'), pl.concat([rd(2), rd(3)]).with_row_index('pi')
    truth = (load_gt().join(q.select(pl.col('entity_id').alias('s1'), 'qi'), on='s1')
                      .join(p.select(pl.col('entity_id').alias('cand'), 'pi'), on='cand').select('qi', 'pi'))
    cast = lambda d: d.with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
    truth = cast(truth)
    # stream the parts: one 202M-row join crashed polars on 16 GB; keep only true pairs found per part
    found = lambda name: pl.concat([truth.join(cast(pl.read_parquet(f, columns=['qi', 'pi'])), on=['qi', 'pi'], how='semi')
                                    for f in sorted(glob.glob(os.path.join(CACHE, name, 'part_*.parquet')))]).unique()
    in_union, in_final = found('cand_train'), found('pr_train')
    miss_u = truth.join(in_union, on=['qi', 'pi'], how='anti')
    pruned = in_union.join(in_final, on=['qi', 'pi'], how='anti')
    res = dict(true_pairs=truth.height, missed_blocking=miss_u.height, missed_pruning=pruned.height)
    say(res)
    # profile of blocking misses
    d = (miss_u.join(q.select('qi', 'country_n', pl.col('name_core').alias('nq')), on='qi')
               .join(p.select('pi', pl.col('name_core').alias('nc'), pl.col('addr_norm').alias('ac'), pl.col('business_name').alias('bc')), on='pi'))
    jac = [len(a & b) / max(len(a | b), 1) for a, b in ((grams(x), grams(y)) for x, y in zip(d['nq'].to_list(), d['nc'].to_list()))]
    d = d.with_columns(pl.Series('jac3', jac, dtype=pl.Float32))
    prof = d.group_by('country_n').agg(pl.len().alias('pairs'), pl.col('jac3').median().round(3).alias('jac3_median'),
                                       (pl.col('jac3') >= 0.5).mean().round(3).alias('jac3_ge_0.5'), (pl.col('jac3') >= 0.3).mean().round(3).alias('jac3_ge_0.3'),
                                       (pl.col('ac') == '').mean().round(3).alias('cand_addr_empty'),
                                       pl.col('bc').str.contains(r'[^\x00-\x7f]').mean().round(3).alias('cand_name_non_ascii')).sort('country_n')
    say(f'profile of blocking misses:\n{prof}')
    res['miss_profile'] = prof.to_dicts()
    rng_seed = 0
    res['by_country'] = {}
    for ctry in d['country_n'].unique().sort().to_list():
        s1s = d.filter(pl.col('country_n') == ctry)['qi'].unique().sort().sample(min(n_per, d.filter(pl.col('country_n') == ctry)['qi'].n_unique()), seed=rng_seed)
        tgt = miss_u.filter(pl.col('qi').is_in(s1s.implode()))
        qm = q.filter(pl.col('qi').is_in(s1s.implode()))
        pm = p.filter(pl.col('country_n') == ctry)
        un = pl.concat([cast(pl.read_parquet(f, columns=['qi', 'pi'])).filter(pl.col('qi').is_in(s1s.implode()))
                        for f in sorted(glob.glob(os.path.join(CACHE, 'cand_train', 'part_*.parquet')))])
        out = {'sampled_s1': qm.height, 'missed_pairs_in_sample': tgt.height}
        for tag, txt in (('name', lambda df: df['name_core'].str.replace_all(' ', '').to_list()),
                         ('name_addr', lambda df: (df['name_core'].str.replace_all(' ', '') + ' ' + df['addr_norm']).to_list())):
            v = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), binary=True, use_idf=False, norm='l2', min_df=2, dtype=np.float32)
            B = v.fit_transform(txt(pm))
            A = v.transform(txt(qm))
            C = sp_matmul_topn(A, B.T.tocsr(), top_n=max(KS), n_threads=14, sort=True).tocsr()
            del B
            rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
            rank = np.arange(C.nnz) - np.repeat(C.indptr[:-1], np.diff(C.indptr))
            hits = pl.DataFrame({'qi': qm['qi'].to_numpy()[rows], 'pi': pm['pi'].to_numpy()[C.indices], 'rank': rank.astype(np.int32)})
            hits = hits.with_columns(pl.col('qi').cast(pl.UInt32), pl.col('pi').cast(pl.UInt32))
            found = tgt.join(hits, on=['qi', 'pi'], how='inner')
            new = hits.join(un, on=['qi', 'pi'], how='anti')
            out[tag] = {f'recall@{k}': round(found.filter(pl.col('rank') < k).height / max(tgt.height, 1), 4) for k in KS}
            out[tag].update({f'new_cands_per_s1@{k}': round(new.filter(pl.col('rank') < k).height / max(qm.height, 1), 1) for k in KS})
            say(ctry, tag, out[tag])
        res['by_country'][ctry] = out
    json.dump(res, open(out_json, 'w'), indent=1, default=str)
    say('done')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
