"""Field normalization for business names and addresses.

All rules are generic, hand-written string transforms (no external lookup).
Raw fields are kept; normalized representations are added as new columns.
"""
import re
import polars as pl
from anyascii import anyascii
import translit

# --- name vocab -----------------------------------------------------------
NAME_CANON = {
    'pvt': 'private', 'pvt.': 'private', 'prv': 'private', 'priv': 'private',
    'ltd': 'limited', 'ltd.': 'limited', 'lt': 'limited', 'ld': 'limited',
    'corp': 'corporation', 'corpn': 'corporation', 'inc': 'incorporated', 'lnc': 'incorporated',
    'co': 'company', 'cos': 'company', 'compny': 'company', 'intl': 'international', 'int': 'international',
    'bros': 'brothers', 'bro': 'brothers', 'svcs': 'services', 'svc': 'services', 'srvcs': 'services',
    'mfg': 'manufacturing', 'mgmt': 'management', 'tech': 'technologies', 'techs': 'technologies',
    'technology': 'technologies', 'assoc': 'associates', 'assocs': 'associates', 'grp': 'group',
    'ent': 'enterprises', 'entp': 'enterprises', 'inds': 'industries', 'ind': 'industries',
    'sys': 'systems', 'syst': 'systems', 'natl': 'national', 'univ': 'university', 'hosp': 'hospital',
    'ctr': 'center', 'centre': 'center', 'cntr': 'center', 'dept': 'department', 'soc': 'society',
    'lndia': 'india', 'ndia': 'india', 'service': 'services', 'phys': 'physicians', 'sch': 'school',
    'ets': 'etablissements', 'cie': 'compagnie', 'ste': 'societe', 'assn': 'association',
    'and': 'and', 'et': 'and', 'n': 'and', 'y': 'and',
}
# legal forms / honorifics / filler that do not identify the business
NAME_STOP = set('''private limited llc incorporated corporation company llp lp pc pllc plc public
the of and a an m s mr mrs ms smt shri sri shree dr prop proprietor firm
sarl sas sasu sa eurl sci snc scop scs compagnie etablissements groupe holding
gmbh ag bv nv pty com www net org in id dba aka fka formerly known as doing business'''.split())
ALIAS_SPLIT = re.compile(r'\b(?:d/?b/?a|a/?k/?a|f/?k/?a|formerly known as|formerly|doing business as|trading as|t/a)\b')
LEET = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '8': 'b', '@': 'a', '$': 's'})

# --- address vocab --------------------------------------------------------
ADDR_CANON = {
    'st': 'street', 'str': 'street', 'strt': 'street', 'rd': 'road', 'rad': 'road', 'roda': 'road',
    'dr': 'drive', 'drv': 'drive', 'ave': 'avenue', 'av': 'avenue', 'avn': 'avenue', 'avnue': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'bvd': 'boulevard', 'boul': 'boulevard',
    'ln': 'lane', 'ct': 'court', 'crt': 'court', 'cir': 'circle', 'pl': 'place', 'pkwy': 'parkway',
    'hwy': 'highway', 'trl': 'trail', 'ter': 'terrace', 'terr': 'terrace', 'sq': 'square', 'mt': 'mount',
    'ste': 'suite', 'apt': 'apartment', 'fl': 'floor', 'flr': 'floor', 'bldg': 'building', 'blk': 'block',
    'hno': 'house', 'hn': 'house', 'h': 'house', 'nr': 'near', 'opp': 'opposite', 'nagr': 'nagar',
    'sec': 'sector', 'sect': 'sector', 'dist': 'district', 'distt': 'district', 'po': 'post', 'ps': 'police',
    'r': 'rue', 'all': 'allee', 'imp': 'impasse', 'che': 'chemin', 'chem': 'chemin', 'rte': 'route',
    'pl': 'place', 'fbg': 'faubourg', 'qu': 'quai', 'crs': 'cours', 'sq': 'square',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west', 'ne': 'northeast', 'nw': 'northwest',
    'se': 'southeast', 'sw': 'southwest', 'ciy': 'city', 'citty': 'city', 'ciity': 'city', 'ciyt': 'city',
    'icty': 'city', 'ccity': 'city', 'ctiy': 'city', 'ity': 'city', 'cty': 'city',
    # common city-name variants
    'bombay': 'mumbai', 'madras': 'chennai', 'bengaluru': 'bangalore', 'gurugram': 'gurgaon',
    'calcutta': 'kolkata', 'ahmadabad': 'ahmedabad', 'poona': 'pune', 'mysuru': 'mysore',
    'trivandrum': 'thiruvananthapuram', 'shivamogga': 'shimoga', 'hubballi': 'hubli', 'belagavi': 'belgaum',
    'kalyan': 'kalyan', 'vizag': 'visakhapatnam', 'baroda': 'vadodara', 'cochin': 'kochi',
}
ORDINAL_WORDS = {w: str(i + 1) for i, w in enumerate(
    'first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth '
    'fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split())}
US_STATES = dict(al='alabama', ak='alaska', az='arizona', ar='arkansas', ca='california', co='colorado',
    ct='connecticut', de='delaware', fl='florida', ga='georgia', hi='hawaii', id='idaho', il='illinois',
    ia='iowa', ks='kansas', ky='kentucky', la='louisiana', me='maine', md='maryland', ma='massachusetts',
    mi='michigan', mn='minnesota', ms='mississippi', mo='missouri', mt='montana', ne='nebraska', nv='nevada',
    nh='new hampshire', nj='new jersey', nm='new mexico', ny='new york', nc='north carolina',
    nd='north dakota', oh='ohio', ok='oklahoma', pa='pennsylvania', ri='rhode island', sc='south carolina',
    sd='south dakota', tn='tennessee', tx='texas', ut='utah', vt='vermont', va='virginia', wa='washington',
    wv='west virginia', wi='wisconsin', wy='wyoming', dc='district of columbia')
IN_STATES = dict(mh='maharashtra', dl='delhi', up='uttar pradesh', wb='west bengal', gj='gujarat',
    tg='telangana', ts='telangana', hr='haryana', rj='rajasthan', kl='kerala', keralam='kerala', pb='punjab',
    od='odisha', orissa='odisha', ka='karnataka', tn='tamil nadu', ap='andhra pradesh', mp='madhya pradesh',
    br='bihar', jh='jharkhand', cg='chhattisgarh', ct='chhattisgarh', uk='uttarakhand', ua='uttarakhand',
    hp='himachal pradesh', jk='jammu kashmir', ga='goa', as_='assam', ch='chandigarh', py='puducherry')
STATE_NAMES = set(US_STATES.values()) | set(IN_STATES.values())


def _ascii(s: pl.Expr) -> pl.Expr:
    # anyascii only for strings that actually contain non-ASCII chars (fast path otherwise)
    return pl.when(s.str.contains(r'[^\x00-\x7f]')).then(
        s.map_elements(anyascii, return_dtype=pl.String)).otherwise(s)


def name_tokens(n: str):
    """Canonical token list for an already ascii+lowercased name."""
    out = []
    for t in re.findall(r'[a-z0-9]+', n):
        if re.search(r'[a-z]', t) and re.search(r'\d', t):  # leetspeak like c0ok, 5ervices
            t = t.translate(LEET)
        out.append(NAME_CANON.get(t, t))
    return out


def norm_name(raw: str):
    s = raw.lower().replace('&', ' and ').replace('+', ' plus ')
    s = re.sub(r'\(\s*id[:\s]*\d+\s*\)|#\s*\d+|\bid[:\s]+\d+', ' ', s)   # appended record ids
    s = re.sub(r'\.(com|net|org|in|co)\b', r' \1', s)
    parts = [p for p in ALIAS_SPLIT.split(s) if p.strip()] or ['']
    toks = name_tokens(ALIAS_SPLIT.sub(' ', s))
    core = [t for t in toks if t not in NAME_STOP and not t.isdigit()]
    alias_cores = [' '.join(t for t in name_tokens(p) if t not in NAME_STOP and not t.isdigit()) for p in parts]
    return ' '.join(toks), ' '.join(core), '|'.join(alias_cores)


# France: regions and departments are interchangeable "state" components (S1 writes the region,
# S2/S3 often the department), so both map to the region.
FR_REGIONS = ['hauts de france', 'nouvelle aquitaine', 'pays de la loire', 'ile de france', 'auvergne rhone alpes',
              'occitanie', 'bretagne', 'normandie', 'grand est', 'provence alpes cote d azur',
              'bourgogne franche comte', 'centre val de loire', 'corse']
FR_DEPTS = {'nord': 'hauts de france', 'pas de calais': 'hauts de france', 'somme': 'hauts de france',
            'oise': 'hauts de france', 'aisne': 'hauts de france', 'gironde': 'nouvelle aquitaine',
            'landes': 'nouvelle aquitaine', 'pyrenees atlantiques': 'nouvelle aquitaine',
            'charente maritime': 'nouvelle aquitaine', 'loire atlantique': 'pays de la loire',
            'maine et loire': 'pays de la loire', 'vendee': 'pays de la loire', 'sarthe': 'pays de la loire',
            'mayenne': 'pays de la loire'}
FR_ADDR = {'st': 'saint', 'ste': 'sainte', 'ch': 'chemin', 'che': 'chemin', 'res': 'residence', 'qu': 'quai',
           'bld': 'boulevard', 'fg': 'faubourg', 'pass': 'passage', 'sq': 'square', 'prom': 'promenade'}
STATE_LOOKUP = {'us': {**US_STATES, **{v: v for v in US_STATES.values()}},
                'india': {**{k.rstrip('_'): v for k, v in IN_STATES.items()}, **{v: v for v in IN_STATES.values()}},
                'france': {**{r: r for r in FR_REGIONS}, **FR_DEPTS}}


def norm_addr(raw: str, country: str = ''):
    """Returns (normalized address without state component, house/street numbers, state)."""
    s = raw.lower()
    s = re.sub(r'\bnull\b|\bn/a\b|\bna\b', ' ', s)
    s = re.sub(r'\bn(?:[°o]|deg)\.?\s*(?=\d)', ' ', s)
    lk = STATE_LOOKUP.get(country, {})  # unknown countries: no state stripping
    fr = country == 'france'
    comps, state = [], ''
    for c in s.split(','):
        key = ' '.join(re.findall(r'[a-z]+', c))
        if key in lk and not re.search(r'\d', c):   # '817 38nd' is an address part, not North Dakota
            state = lk[key]
        else:
            comps.append(c)
    s = ' , '.join(comps)
    s = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1 ', s)              # 10th -> 10
    s = re.sub(r'(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)', ' ', s)     # 13815c -> 13815 c, flr10 -> flr 10
    toks, nums = [], []
    for t in re.findall(r'[a-z0-9]+', s):
        if t.isdigit():
            t = t.lstrip('0') or '0'
            nums.append(t)
        else:
            m = re.fullmatch(r'(\d+)(st|nd|rd|th|st)', t)
            if m:
                t = m.group(1)
                nums.append(t)
            elif t in ORDINAL_WORDS:
                t = ORDINAL_WORDS[t]
                nums.append(t)
            else:
                t = FR_ADDR.get(t) or ADDR_CANON.get(t, t) if fr else ADDR_CANON.get(t, t)
        toks.append(t)
    return ' '.join(toks), ' '.join(dict.fromkeys(nums)), state


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Adds name_norm, name_core, name_alias, addr_norm, addr_nums, country_n columns."""
    df = df.with_columns(
        _ascii(pl.col('business_name').map_elements(translit.name, return_dtype=pl.String)).alias('_n'),
        _ascii(pl.col('business_address').map_elements(translit.addr, return_dtype=pl.String)).alias('_a'),
        pl.col('country').str.strip_chars().str.to_lowercase().alias('country_n'),
    )
    n = df['_n'].to_list()
    a = df['_a'].to_list()
    nn = [norm_name(x) for x in n]
    aa = [norm_addr(x, c) for x, c in zip(a, df['country_n'].to_list())]
    return df.drop('_n', '_a').with_columns(
        pl.Series('name_norm', [x[0] for x in nn]),
        pl.Series('name_core', [x[1] for x in nn]),
        pl.Series('name_alias', [x[2] for x in nn]),
        pl.Series('addr_norm', [x[0] for x in aa]),
        pl.Series('addr_nums', [x[1] for x in aa]),
        pl.Series('addr_state', [x[2] for x in aa]),
    )


if __name__ == '__main__':
    for r in ['Cook K�ystone Navios  LLC', 'SMT SARDA & BROTHERS CLINIC', 'Nylaflux Co d/b/a Wellness Developers Private Limited',
              'M/s Divyansh  Connect Private Ltd (ID: 70983)', 'Strategic Dibglsotic Digital LLC #71076', 'allpropertysolutions.com',
              'C0ok 5ervices lnc', 'राम मार्केटिंग प्राइवेट लिमिटेड']:
        print(r, '->', norm_name(anyascii(r).lower()))
    for r in ['#500 Dry Valley Rd, # C101, Algood, Tennessee', '1541 1/2 ENSOR ST, N/A, BALTIMORE, MD', '3744 TENTH STREET',
              '0028 RUE DE LA CONVENTION', 'N° 16 AVENUE DE LA LIBÉRATION', '3744 10st St', '(6) RUE FERNAND HABASQUE']:
        print(r, '->', norm_addr(anyascii(r), 'us'))
    assert norm_addr('3744 TENTH STREET')[1] == '3744 10'
    assert norm_name('c0ok 5ervices lnc')[1] == 'cook services'
