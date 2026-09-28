"""Low-memory go/no-go for a candidate submission vs the confirmed best, calibrated on a leaderboard result.

1. Streams <best> and <new> side by side (same S1 order) -> the ADDED and REMOVED pair lists (O(1) memory).
2. Profiles each list per country, loading only the records involved (parquet column/row pushdown).
3. Compares with the S5 benchmark: the 66,330 pairs S5 removed from S4 had these profiles and turned out to be
   ~34-43% false merges (leaderboard 0.969 -> 0.971). A new REMOVAL set helps if it looks at least as
   distractor-like as that benchmark; an ADDED set should look like already-agreed pairs (true-match-like).
Usage: python setprof.py <best_matching.tsv> <new_matching.tsv> <out_prefix>
"""
import os, sys
import polars as pl
from diag import COLS, pair_frame, FLAGS
from common import CACHE

# profile of the S5 removal set (S4 - S5), measured at decision time (docs/audit_s5.md §2 + diffprof output)
BENCH_REMOVED = {'france': dict(num1_diff=0.637, filler_only_diff=0.614, cand_addr_empty=0.142),
                 'india': dict(num1_diff=0.515, filler_only_diff=0.372, cand_addr_empty=0.068),
                 'us': dict(num1_diff=0.443, filler_only_diff=0.304, cand_addr_empty=0.273)}
TRUE_REF = {'india': dict(num1_diff=0.206, cand_addr_empty=0.040), 'us': dict(num1_diff=0.112, cand_addr_empty=0.048)}


def stream_diff(best, new, prefix):
    a, b = open(best, encoding='utf-8'), open(new, encoding='utf-8')
    next(a); next(b)
    wa, wr = open(prefix + '_added.tsv', 'w', encoding='utf-8', newline='\n'), open(prefix + '_removed.tsv', 'w', encoding='utf-8', newline='\n')
    na = nr = 0
    for la, lb in zip(a, b):
        s1, _, x = la.rstrip('\n').partition('\t'); s1b, _, y = lb.rstrip('\n').partition('\t')
        assert s1 == s1b, 'files not in the same S1 order'
        X, Y = {t for t in x.split(',') if t}, {t for t in y.split(',') if t}
        for c in Y - X: wa.write(f'{s1}\t{c}\n'); na += 1
        for c in X - Y: wr.write(f'{s1}\t{c}\n'); nr += 1
    wa.close(); wr.close()
    return na, nr


def profile(path):
    if os.path.getsize(path) == 0:
        return None
    s = pl.read_csv(path, separator='\t', has_header=False, new_columns=['s1', 'c'], quote_char=None, infer_schema=False)
    if s.height == 0:
        return None
    ids1, ids2 = s['s1'].unique().implode(), s['c'].unique().implode()
    q = pl.scan_parquet(os.path.join(CACHE, 'test_s1.parquet')).select(COLS).filter(pl.col('entity_id').is_in(ids1)).collect().with_row_index('qi')
    p = pl.concat([pl.scan_parquet(os.path.join(CACHE, f'test_s{i}.parquet')).select(COLS).filter(pl.col('entity_id').is_in(ids2)).collect()
                   for i in (2, 3)]).with_row_index('pi')
    pr = s.join(q.select('qi', pl.col('entity_id').alias('s1')), on='s1').join(p.select('pi', pl.col('entity_id').alias('c')), on='c').select('qi', 'pi')
    d = pair_frame(q.drop('entity_id'), p.drop('entity_id'), pr)
    return d.group_by('country_n').agg(pl.len().alias('pairs'), *[pl.col(f).mean().round(3) for f in FLAGS]).sort('country_n')


def main(best, new, prefix):
    na, nr = stream_diff(best, new, prefix)
    print(f'{best} -> {new}: +{na:,} added, -{nr:,} removed')
    for tag in ('removed', 'added'):
        r = profile(f'{prefix}_{tag}.tsv')
        if r is None:
            print(f'-- {tag.upper()}: none'); continue
        print(f'-- {tag.upper()}\n{r}')
        for row in r.iter_rows(named=True):
            c, b = row['country_n'], BENCH_REMOVED.get(row['country_n'], {})
            if tag == 'removed':   # at least as distractor-like as the benchmark set on the two main signals
                score = (row['num1_diff'] >= b['num1_diff']) + (row['filler_only_diff'] + row['cand_addr_empty'] >= b['filler_only_diff'] + b['cand_addr_empty'])
                print(f"   {c:7s} n={row['pairs']:>7,}  signals as bad as the S5 benchmark: {score}/2 -> {'REMOVE (helps)' if score == 2 else 'uncertain' if score == 1 else 'KEEP (removal likely hurts)'}")
            else:
                t = TRUE_REF.get(c, dict(num1_diff=0.159, cand_addr_empty=0.044))
                ok = row['num1_diff'] <= 1.5 * t['num1_diff'] and row['cand_addr_empty'] <= 2 * t['cand_addr_empty']
                print(f"   {c:7s} n={row['pairs']:>7,}  true-match-like: {'yes -> ADD' if ok else 'no -> do not add'}" + ('  (France: pooled reference)' if c not in TRUE_REF else ''))


if __name__ == '__main__':
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    main(*sys.argv[1:])
