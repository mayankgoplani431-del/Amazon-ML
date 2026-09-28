"""Synthetic mini dataset in the competition format (smoke test only, not for modelling)."""
import random, os, sys
random.seed(0)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
W = 'alpha nova cook keystone navios vision harbor summit prime metro pixel orbit lotus crest river maple cedar'.split()
LEG = {'us': ['LLC', 'Inc', 'Corp', 'L.L.C.', ''], 'india': ['Private Limited', 'Pvt Ltd', 'LLP', ''], 'france': ['SARL', 'S.A.R.L.', 'SAS', '']}
CITY = {'us': ['Algood, Tennessee', 'Baltimore, MD', 'Austin, Texas'], 'india': ['Thane, Maharashtra 400601', 'Noida, Uttar Pradesh 201301'],
        'france': ['44000 Nantes, Pays de la Loire', '33000 Bordeaux, Gironde', '59000 Lille, Nord']}
ST = {'us': ['Main St', 'Dry Valley Rd', 'Oak Ave'], 'india': ['MG Road', 'Sector 5'], 'france': ['Chemin Moliere', 'R. de la Convention', 'Av Victor Hugo']}
def rec(c):
    return f"{random.choice(W).title()} {random.choice(W).title()}", random.choice(LEG[c]), random.randint(1, 999), random.choice(ST[c]), random.choice(CITY[c])
def fmt(name, leg, num, st, city, noise):
    n = f'{name} {leg}'.strip()
    if noise and random.random() < .3: n = n.replace('o', '0', 1)
    if noise and random.random() < .2: n = n.upper()
    a = f'{num} {st}, {city}' if not (noise and random.random() < .15) else f'{st}, {city}'
    return n, a
def split(name, n1, countries, gt):
    ids = iter(random.sample(range(10**8, 10**9), n1 * 20))
    s1, s2, s3, g = [], [], [], []
    for _ in range(n1):
        c = random.choice(countries); base = rec(c); i1 = f'S1-{next(ids)}'
        s1.append((i1, *fmt(*base, False), c.upper() if c == 'us' else c.title()))
        m = []
        for _ in range(random.choice([0, 1, 2, 3, 4, 5])):
            src = random.choice(['S2', 'S3']); i = f'{src}-{next(ids)}'; m.append(i)
            (s2 if src == 'S2' else s3).append((i, *fmt(*base, True), c.title()))
        for _ in range(random.choice([0, 1, 2])):   # distractor: shifted number / suffix added
            src = random.choice(['S2', 'S3']); i = f'{src}-{next(ids)}'
            nm, lg, num, st, city = base
            d = (nm + ' Holding', lg, num, st, city) if random.random() < .5 else (nm, lg, num + random.randint(1, 9), st, city)
            (s2 if src == 'S2' else s3).append((i, *fmt(*d, True), c.title()))
        g.append((i1, ','.join(m)))
    hdr = 'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
    for k, rows in ((1, s1), (2, s2), (3, s3)):
        with open(os.path.join(R, name, f'{name}_source{k}.tsv'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(hdr + ''.join('\t'.join(r) + '\n' for r in rows))
    if gt:
        with open(os.path.join(R, name, f'{name}_ground_truth.tsv'), 'w', encoding='utf-8', newline='\n') as f:
            f.write('source1_entity_id\tmatched_entity_ids\n' + ''.join(f'{a}\t{b}\n' for a, b in g))
split('train', 4000, ['us', 'india'], True)
split('test', 3000, ['us', 'india', 'france'], False)
print('smoke data written')
