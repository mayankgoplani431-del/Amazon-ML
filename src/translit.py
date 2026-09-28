"""Learn native-script -> English transliteration dictionaries from TRAINING ground truth only.

Names: Source 2/3 records whose name is written in an Indic script are word-aligned with the
matched Source 1 English name (token counts agree in >99.99% of pairs), so position-wise
co-occurrence gives a word dictionary (e.g. प्राइवेट -> private).
Addresses: a native-script comma component is mapped to the matched S1 record's state.
"""
import collections, json, os, re
import polars as pl
from common import DATA, CACHE, read_tsv, load_gt

NATIVE = re.compile(r'[ऀ-෿]')   # Devanagari .. Sinhala blocks (all Indic scripts)
PATH = os.path.join(CACHE, 'translit.json')


def tok(s):
    return [t for t in re.split(r'[\s,]+', s.replace('‌', '').replace('‍', '')) if t]


def build():
    from normalize import norm_addr
    s1 = read_tsv(os.path.join(DATA, 'train', 'train_source1.tsv'))
    p = pl.concat([read_tsv(os.path.join(DATA, 'train', f'train_source{i}.tsv')) for i in (2, 3)])
    gt = load_gt()
    d = (gt.join(p.select(pl.col('entity_id').alias('cand'), pl.col('business_name').alias('nn'),
                          pl.col('business_address').alias('na')), on='cand')
           .join(s1.select(pl.col('entity_id').alias('s1'), pl.col('business_name').alias('en'),
                           pl.col('business_address').alias('ea'), pl.col('country').alias('c')), on='s1'))
    dn = d.filter(pl.col('nn').str.contains(NATIVE.pattern))
    co = collections.defaultdict(collections.Counter)
    for en, nn in zip(dn['en'].to_list(), dn['nn'].to_list()):
        a, b = tok(en.lower()), tok(nn)
        if len(a) == len(b):
            for x, y in zip(a, b):
                if NATIVE.search(y):
                    co[y][re.sub(r'[^a-z0-9&]', '', x)] += 1
    names = {k: c.most_common(1)[0][0] for k, c in co.items() if c.most_common(1)[0][1] >= 2}
    da = d.filter(pl.col('na').str.contains(NATIVE.pattern))
    ca = collections.defaultdict(collections.Counter)
    for na, ea, c in zip(da['na'].to_list(), da['ea'].to_list(), da['c'].to_list()):
        st = norm_addr(ea.lower(), c.strip().lower())[2]
        if st:
            for comp in na.split(','):
                comp = comp.strip()
                if NATIVE.search(comp):
                    ca[comp][st] += 1
    addr = {k: c.most_common(1)[0][0] for k, c in ca.items()
            if c.most_common(1)[0][1] >= 5 and c.most_common(1)[0][1] / sum(c.values()) > 0.8}
    json.dump({'name': names, 'addr': addr}, open(PATH, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'translit: {len(names)} name words, {len(addr)} address components')


_D = None


def _load():
    global _D
    if _D is None:
        _D = json.load(open(PATH, encoding='utf-8'))
    return _D


def name(s: str) -> str:
    if not NATIVE.search(s):
        return s
    m = _load()['name']
    return ' '.join(m.get(t, t) for t in tok(s))


def addr(s: str) -> str:
    if not NATIVE.search(s):
        return s
    m = _load()['addr']
    return ','.join(m.get(c.strip(), c) for c in s.split(','))


if __name__ == '__main__':
    build()
    print(name('विजन कंसल्टिंग प्राइवेट लिमिटेड'), '|', addr('18, THANE, महाराष्ट्र'))
