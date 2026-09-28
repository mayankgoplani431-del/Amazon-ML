"""implement_this.md, Stages 1-3, 5, 6 as specified, each rule validated on the 0.977 model's held-out OOF (half B) first.
Stage 1  pre-cleaning (DOMAIN/URL/BRACKET/SUITE regexes, French street abbreviations)
Stage 2  identity_core (legal forms by boundary-frequency mining >85% + known lists), legal_form_concept,
         rule R2: reject if identity_core similarity < 0.65 while the address is identical
Stage 3  rule R3: numbers differ -> keep if |d|<=2 & core>=0.95 & street>=0.90; reject if core similarity is low (<0.80)
Stage 5  rule R5: singleton guard, lone predicted match needs score >= 0.72; multi-match keeps score >= 0.58
Stage 6  integrity checks (rows, duplicates, S1 ids, singleton rate 5.7-6.0%)
Stage 4 (dense embeddings) is not run: needs torch + model download + ~14M CPU encodings + retrain, beyond the deadline.
Validated rules (half-B gain > 0 on both plain and density-stressed F0.5) are applied to S14 (US/India; France only if
the rule is not name-dissimilarity based, since S10 showed French name-dissimilar pairs are ~93% true).
Usage: python plan_md.py [--apply]
"""
import os, re, sys, json, time, collections, shutil
import numpy as np
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
sys.path.insert(0, 'D:/Amazon ML/new')
from tune_0977 import decide, macro

R, N = 'D:/Amazon ML/', 'D:/Amazon ML/new/'
DS = R + 'student_resource/dataset/'
t0 = time.time()
say = lambda *a: print(*a, f'[{time.time() - t0:.0f}s]', flush=True)
W = {'India': 0.4675, 'US': 0.3827, 'France': 0.1498}

# ---------------- Stage 1: deterministic pre-cleaning (patterns exactly as in implement_this.md)
DOMAIN_PATTERN = re.compile(r'(?i)\.(com|org|net|in|fr|co\.in|gov|io|biz|info)\b')
URL_PREFIX_PATTERN = re.compile(r'(?i)\b(https?://|www\.)')
BRACKET_PATTERN = re.compile(r'\[\[.*?\]\]|<<.*?>>|[\[\]<>]')
SUITE_PATTERN = re.compile(r'(?i)\b(pmb|suite|ste|apt|fl|floor|#)\s*\d+')
FR_STREET = {'bd': 'boulevard', 'bvd': 'boulevard', 'rue': 'rue', 'r': 'rue', 'av': 'avenue', 'ave': 'avenue', 'allee': 'allee',
             'cedex': '', 'bis': 'bis', 'ter': 'ter'}
KNOWN_LEGAL = {'llc': 'LIMITED_LIABILITY', 'ltd': 'LIMITED_LIABILITY', 'limited': 'LIMITED_LIABILITY', 'pvt': 'LIMITED_LIABILITY',
               'private': 'LIMITED_LIABILITY', 'sarl': 'LIMITED_LIABILITY', 'eurl': 'LIMITED_LIABILITY', 'sas': 'SIMPLIFIED_STOCK',
               'sasu': 'SIMPLIFIED_STOCK', 'sa': 'STOCK', 'inc': 'CORPORATION', 'incorporated': 'CORPORATION', 'corp': 'CORPORATION',
               'corporation': 'CORPORATION', 'llp': 'PARTNERSHIP', 'lp': 'PARTNERSHIP', 'plc': 'PUBLIC', 'co': 'COMPANY', 'company': 'COMPANY',
               'pc': 'PROFESSIONAL', 'pllc': 'PROFESSIONAL', 'sci': 'CIVIL', 'snc': 'PARTNERSHIP', 'gmbh': 'LIMITED_LIABILITY', 'ag': 'STOCK'}


def clean_name(s):
    s = anyascii(s or '').lower()
    s = URL_PREFIX_PATTERN.sub(' ', s); s = DOMAIN_PATTERN.sub(' ', s); s = BRACKET_PATTERN.sub(' ', s)
    s = re.sub(r'\b[a-z](?:\.[a-z])+\.?(?![a-z])', lambda m: m.group().replace('.', ''), s)   # s.a.r.l. -> sarl
    return re.findall(r'[a-z0-9]+', s)


def clean_addr(s):
    s = anyascii(s or '').lower()
    s = SUITE_PATTERN.sub(' ', s); s = BRACKET_PATTERN.sub(' ', s)
    toks = [FR_STREET.get(t, t) for t in re.findall(r'[a-z0-9]+', s)]
    return [t for t in toks if t]


# ---------------- Stage 2: boundary-frequency legal-form mining (>85% of occurrences at name start or end)
def mine_legal(names, min_count=2000, share=0.85):
    tot, bnd = collections.Counter(), collections.Counter()
    for toks in names:
        if len(toks) < 2: continue
        for i, t in enumerate(toks):
            tot[t] += 1
            if i == 0 or i == len(toks) - 1: bnd[t] += 1
    return {t for t, n in tot.items() if n >= min_count and bnd[t] / n > share and not t.isdigit()}


def analyse(names, addrs, legal):
    core, concept, street, num = [], [], [], []
    for n, a in zip(names, addrs):
        tn = clean_name(n)
        core.append(' '.join(t for t in tn if t not in legal and t not in KNOWN_LEGAL))
        concept.append('|'.join(sorted({KNOWN_LEGAL[t] for t in tn if t in KNOWN_LEGAL})))
        ta = clean_addr(a)
        nums = [t for t in ta if t.isdigit()]
        num.append(nums[0].lstrip('0') if nums else '')
        street.append(' '.join(t for t in ta if not t.isdigit()))
    return core, concept, street, num


def pair_signals(d):
    """d has core_q/core_c/street_q/street_c/num_q/num_c/addr_q/addr_c columns -> similarity signals (0-1)."""
    core = cpdist(d['core_q'].to_list(), d['core_c'].to_list(), scorer=fuzz.token_sort_ratio, workers=-1) / 100
    street = cpdist(d['street_q'].to_list(), d['street_c'].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100
    addr = cpdist(d['addr_q'].to_list(), d['addr_c'].to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100
    nq, nc = d['num_q'].cast(pl.Int64, strict=False), d['num_c'].cast(pl.Int64, strict=False)
    return d.with_columns(pl.Series('core_sim', core), pl.Series('street_sim', street), pl.Series('addr_sim2', addr),
                          ((d['num_q'] != '') & (d['num_c'] != '') & (d['num_q'] != d['num_c'])).alias('num_diff'),
                          (nq - nc).abs().alias('num_absdiff'))


RULES = {
    'R2_core<0.65_same_address': lambda: (pl.col('core_sim') < 0.65) & (pl.col('addr_sim2') >= 0.95),
    'R3_numbers_differ_low_core': lambda: pl.col('num_diff') & (pl.col('core_sim') < 0.80)
                                          & ~((pl.col('num_absdiff') <= 2) & (pl.col('core_sim') >= 0.95) & (pl.col('street_sim') >= 0.90)),
    'R5_singleton_guard': lambda: ((pl.col('n_pred') == 1) & (pl.col('score') < 0.72)) | ((pl.col('n_pred') > 1) & (pl.col('score') < 0.58)),
}


def load(split, ids_s1, ids_pool):
    rd = lambda f: pl.read_csv(DS + f'{split}/{split}_{f}.tsv', separator='\t', quote_char=None, infer_schema=False).fill_null('')
    q = rd('source1').filter(pl.col('entity_id').is_in(ids_s1.implode()))
    p = pl.concat([rd('source2'), rd('source3')]).filter(pl.col('entity_id').is_in(ids_pool.implode()))
    return q, p


def build(pairs, q, p, legal):
    cq = analyse(q['business_name'].to_list(), q['business_address'].to_list(), legal)
    cp = analyse(p['business_name'].to_list(), p['business_address'].to_list(), legal)
    Q = pl.DataFrame({'s1': q['entity_id'], 'core_q': cq[0], 'concept_q': cq[1], 'street_q': cq[2], 'num_q': cq[3],
                      'addr_q': [' '.join(clean_addr(a)) for a in q['business_address'].to_list()]})
    P = pl.DataFrame({'cand': p['entity_id'], 'core_c': cp[0], 'concept_c': cp[1], 'street_c': cp[2], 'num_c': cp[3],
                      'addr_c': [' '.join(clean_addr(a)) for a in p['business_address'].to_list()]})
    d = pairs.join(Q, on='s1').join(P, on='cand')
    d = pair_signals(d).with_columns(pl.len().over('s1').alias('n_pred'))
    return d


def main(apply=False):
    oof = pl.read_parquet(N + 'train_oof_scores.parquet').rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'})
    gt = (pl.read_csv(DS + 'train/train_ground_truth.tsv', separator='\t', quote_char=None, infer_schema=False).fill_null('')
            .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    owned, nt_all = gt['cand'].unique(), gt.group_by('s1').len().rename({'len': 'ntrue'})
    halves = sorted(oof.filter(pl.col('held_out'))['heldout_half'].unique().to_list())
    B = oof.filter(pl.col('held_out') & (pl.col('heldout_half') == halves[-1]))
    base = decide(B.select('s1', 'cand', 'score', 'country'), {}, {}).join(B.select('s1', 'cand', 'score', 'country'), on=['s1', 'cand'])
    q, p = load('train', B['s1'].unique(), base['cand'].unique())
    # Stage 2 mining on the full train S1 names (label-free)
    legal = mine_legal(clean_name(n) for n in pl.read_csv(DS + 'train/train_source1.tsv', separator='\t', quote_char=None,
                                                         infer_schema=False, columns=['business_name']).fill_null('')['business_name'].to_list())
    say('Stage 2 mined boundary legal forms:', sorted(legal)[:60], f'({len(legal)} total)')
    d = build(base.select('s1', 'cand', 'score', 'country'), q, p, legal)
    lab = oof.select('s1', 'cand', 'label')
    ntB = B.select('s1', 'country').unique().join(nt_all, on='s1', how='left').with_columns(pl.col('ntrue').fill_null(0))
    report = {}
    for name, rule in RULES.items():
        drop = d.filter(rule())
        pr = base.select('s1', 'cand').join(drop.select('s1', 'cand'), on=['s1', 'cand'], how='anti')
        r = {}
        for c in ('India', 'US'):
            ntc, bc, prc = ntB.filter(pl.col('country') == c).drop('country'), base.filter(pl.col('country') == c).select('s1', 'cand'), pr.join(ntB.filter(pl.col('country') == c).select('s1'), on='s1', how='semi')
            r[c] = dict(dropped=int(drop.filter(pl.col('country') == c).height),
                        dropped_true_share=round(float(drop.filter(pl.col('country') == c).join(lab, on=['s1', 'cand'])['label'].mean() or 0), 4),
                        gain_plain=round(macro(prc, lab, ntc, owned, 1.0) - macro(bc, lab, ntc, owned, 1.0), 5),
                        gain_stressed=round(macro(prc, lab, ntc, owned, 1.9) - macro(bc, lab, ntc, owned, 1.9), 5))
        report[name] = r
        say(name, json.dumps(r))
    json.dump(report, open(N + 'plan_md_halfB.json', 'w'), indent=1)
    good = [k for k, v in report.items() if all(v[c]['gain_plain'] > 0 and v[c]['gain_stressed'] > 0 for c in ('India', 'US'))]
    say('rules validated on held-out half B (applied):', good, '| rejected:', [k for k in report if k not in good])
    if not apply:
        return
    ex = lambda path: (pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False).fill_null('')
                         .select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('cand')).explode('cand').filter(pl.col('cand') != ''))
    s14 = ex(N + 'S14/matching_results.tsv')
    out = s14
    if good:
        ts = pl.read_parquet(N + 'test_scores.parquet', columns=['source1_entity_id', 'candidate_entity_id', 'score', 'country']).rename({'source1_entity_id': 's1', 'candidate_entity_id': 'cand'})
        cur = s14.join(ts, on=['s1', 'cand'], how='left').with_columns(pl.col('score').fill_null(1.0))
        qt, pt = load('test', cur['s1'].unique(), cur['cand'].unique())
        dt = build(cur, qt, pt, legal)
        drop = pl.concat([dt.filter(RULES[k]()).select('s1', 'cand', 'country') for k in good]).unique()
        name_based = any(k.startswith('R2') for k in good)
        if name_based:   # S10 lesson: never apply name-dissimilarity removals to France
            drop = drop.filter(pl.col('country') != 'France')
        out = s14.join(drop.select('s1', 'cand'), on=['s1', 'cand'], how='anti')
        say('removed from S14 on test:', drop.group_by('country').len().rows())
    # ---------------- Stage 6: integrity + write S16
    os.makedirs(N + 'S16', exist_ok=True)
    order = pl.read_csv(DS + 'test/test_source1.tsv', separator='\t', quote_char=None, infer_schema=False, columns=['entity_id']).rename({'entity_id': 's1'})
    g = out.sort(['s1', 'cand']).group_by('s1', maintain_order=True).agg(pl.col('cand').str.join(','))
    fin = order.join(g, on='s1', how='left').with_columns(pl.col('cand').fill_null(''))
    assert fin.height == 1_732_544 and out.group_by('cand').len().filter(pl.col('len') > 1).height == 0
    assert not out['cand'].str.starts_with('S1-').any()
    rate = float((fin['cand'] == '').mean())
    with open(N + 'S16/matching_results.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for a, b in fin.iter_rows():
            f.write(f'{a}\t{b}\n')
    shutil.copyfile(N + 'candidate_pairs.tsv', N + 'S16/candidate_pairs.tsv')
    say(f'S16 written: {out.height:,} pairs, singleton rate {100 * rate:.2f}% (target 5.7-6.0%: {"OK" if 0.057 <= rate <= 0.060 else "OUTSIDE"})')


if __name__ == '__main__':
    main('--apply' in sys.argv)
