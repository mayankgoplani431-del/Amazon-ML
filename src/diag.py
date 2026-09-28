"""Label-free per-country diagnostics of a submission (F0): is France behaving like the training countries?

  1. prediction profile per country: empty rate, matches per S1, count histogram
  2. pair-type profile of predicted pairs (distractor signatures) per country, and of TRUE train pairs as reference
  3. data-driven substitution mining: when a predicted pair differs by exactly one name/address token, which
     token replaced which ("r"->"rue", "sarl"->"sas", city words ...) = abbreviations/noise the normalizer misses
Usage: python diag.py <matching_results.tsv> [tag]   -> work/reports/diag_<tag>.txt
"""
import os, sys
import polars as pl
import candidates
from common import WORK, CACHE, read_tsv, load_gt

COLS = ['entity_id', 'country_n', 'name_norm', 'name_core', 'addr_norm', 'addr_nums']


def lean(split):
    """(q, p) with only COLS, row order identical to candidates.load_split (qi/pi = row index). Reads just these
    columns from parquet: load_split reads whole tables and segfaulted on the 16 GB laptop."""
    rd = lambda s: pl.read_parquet(os.path.join(CACHE, f'{split}_s{s}.parquet'), columns=COLS)
    return rd(1).with_row_index('qi'), pl.concat([rd(2), rd(3)]).with_row_index('pi')


def pair_frame(q, p, pairs):
    """pairs (qi, pi) -> one row per pair with both sides' fields and signature flags."""
    d = pairs.join(q, on='qi').join(p, on='pi', suffix='_c')
    tok = lambda c: pl.col(c).str.split(' ').list.eval(pl.element().filter(pl.element() != ''))
    d = d.with_columns(tok('name_core').alias('A'), tok('name_core_c').alias('B'), tok('addr_norm').alias('AA'),
                       tok('addr_norm_c').alias('AB'),
                       pl.col('addr_nums').str.split(' ').list.first().fill_null('').alias('n1'),
                       pl.col('addr_nums_c').str.split(' ').list.first().fill_null('').alias('n1c'))
    d = d.with_columns(pl.col('A').list.set_difference('B').alias('A_B'), pl.col('B').list.set_difference('A').alias('B_A'),
                       pl.col('A').list.set_intersection('B').alias('AiB'),
                       pl.col('AA').list.set_difference('AB').alias('AA_B'), pl.col('AB').list.set_difference('AA').alias('AB_A'))
    shared_in_addr = pl.col('AiB').list.set_difference('AA').list.len() == 0
    return d.with_columns(
        (pl.col('name_core') == pl.col('name_core_c')).alias('name_eq'),
        ((pl.col('name_core') == pl.col('name_core_c')) & (pl.col('addr_norm') == pl.col('addr_norm_c'))).alias('exact'),
        ((pl.col('name_core') == pl.col('name_core_c')) & (pl.col('name_norm') != pl.col('name_norm_c'))).alias('filler_only_diff'),
        ((pl.col('n1') != '') & (pl.col('n1c') != '') & (pl.col('n1') != pl.col('n1c'))).alias('num1_diff'),
        (pl.col('addr_norm_c') == '').alias('cand_addr_empty'),
        ((pl.col('A_B').list.len() == 1) & (pl.col('B_A').list.len() == 1)).alias('one_word_swapped'),
        ((pl.col('AiB').list.len() > 0) & shared_in_addr & (pl.col('A_B').list.len() > 0)).alias('only_address_words_shared'),
    )


FLAGS = ['exact', 'name_eq', 'filler_only_diff', 'num1_diff', 'cand_addr_empty', 'one_word_swapped', 'only_address_words_shared']


def profile(d, tag):
    t = d.group_by('country_n').agg(pl.len().alias('pairs'), *[pl.col(f).mean().round(4) for f in FLAGS]).sort('country_n')
    return f'-- pair-type profile: {tag}\n{t}'


def substitutions(d, a, b, tag, k=40):
    s = (d.filter((pl.col(a).list.len() == 1) & (pl.col(b).list.len() == 1))
          .select('country_n', pl.col(a).list.first().alias('s1_word'), pl.col(b).list.first().alias('pool_word'))
          .group_by('country_n', 's1_word', 'pool_word').len().sort('len', descending=True))
    out = [f'-- most frequent single-word substitutions ({tag})']
    for c in s['country_n'].unique().sort().to_list():
        out.append(f'  [{c}] ' + ', '.join(f'{x}->{y} {n}' for _, x, y, n in s.filter(pl.col('country_n') == c).head(k).rows()))
    return '\n'.join(out)


def main(path, tag='sub'):
    q, p = lean('test')
    m = read_tsv(path)
    per = m.select(pl.col('source1_entity_id').alias('entity_id'),
                   pl.col('matched_entity_ids').str.split(',').list.eval(pl.element().filter(pl.element() != '')).alias('ids'))
    per = per.join(q.select('qi', 'entity_id', 'country_n'), on='entity_id', how='right')
    per = per.with_columns(pl.col('ids').fill_null([]).list.len().alias('k'))
    out = [f'DIAG {tag}: {path}']
    prof = per.group_by('country_n').agg(pl.len().alias('S1'), (pl.col('k') == 0).mean().round(4).alias('empty_rate'),
                                         pl.col('k').mean().round(3).alias('matches_per_S1'),
                                         *[(pl.col('k') == i).mean().round(4).alias(f'k{i}') for i in range(1, 8)],
                                         (pl.col('k') >= 8).mean().round(4).alias('k8+')).sort('country_n')
    out.append(f'-- prediction profile per country\n{prof}')
    pairs = (per.select('qi', 'ids').explode('ids').drop_nulls('ids')
                .join(p.select('pi', pl.col('entity_id').alias('ids')), on='ids').select('qi', 'pi'))
    d = pair_frame(q.drop('entity_id'), p.drop('entity_id'), pairs)
    out += [profile(d, f'predicted pairs ({tag})'), substitutions(d, 'A_B', 'B_A', 'name core'),
            substitutions(d, 'AA_B', 'AB_A', 'address')]
    # reference: TRUE train pairs (what real matches look like in the training countries)
    try:
        if os.environ.get('DIAG_TRAIN_REF', '0') != '1':
            raise RuntimeError('set DIAG_TRAIN_REF=1 (diffprof.py prints the train references)')
        tq, tp = lean('train')
        gt = (load_gt().join(tq.select(pl.col('entity_id').alias('s1'), 'qi'), on='s1')
                       .join(tp.select(pl.col('entity_id').alias('cand'), 'pi'), on='cand').select('qi', 'pi'))
        out.append(profile(pair_frame(tq.drop('entity_id'), tp.drop('entity_id'), gt.sample(min(gt.height, 2_000_000), seed=0)),
                           'TRUE train pairs (reference)'))
    except Exception as e:   # train caches not present: profile of the submission alone is still useful
        out.append(f'(train reference skipped: {e!r})')
    txt = '\n\n'.join(out)
    os.makedirs(os.path.join(WORK, 'reports'), exist_ok=True)
    open(os.path.join(WORK, 'reports', f'diag_{tag}.txt'), 'w', encoding='utf-8').write(txt)
    print(txt)


if __name__ == '__main__':
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'sub')
