"""Pairwise features for (S1, candidate) pairs. Everything is vectorized (rapidfuzz cpdist,
numpy, polars). No country one-hot: features are country-agnostic so they transfer to unseen
country labels (e.g. France in test)."""
import re
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

LEGAL = {'llc': 'llc', 'incorporated': 'inc', 'corporation': 'corp', 'limited': 'ltd', 'private': 'pvt',
         'llp': 'llp', 'lp': 'lp', 'pc': 'pc', 'pllc': 'pllc', 'plc': 'plc', 'company': 'co', 'public': 'public',
         'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'sa': 'sa', 'eurl': 'eurl', 'sci': 'sci', 'snc': 'snc',
         'compagnie': 'cie', 'gmbh': 'gmbh', 'trust': 'trust', 'foundation': 'foundation'}


def legal_set(name_norm: pl.Series) -> pl.Series:
    return name_norm.map_elements(lambda s: ' '.join(sorted({LEGAL[t] for t in s.split() if t in LEGAL})),
                                  return_dtype=pl.String)


def _rf(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _rowdot(A, B, qi, pi, chunk=2_000_000):
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        e = s + chunk
        out[s:e] = np.asarray(A[qi[s:e]].multiply(B[pi[s:e]]).sum(1)).ravel()
    return out


def _set_feats(a: pl.Series, b: pl.Series, prefix):
    """Token-set overlap stats between space-separated strings."""
    df = pl.DataFrame({'a': a.str.split(' ').list.eval(pl.element().filter(pl.element() != '')),
                       'b': b.str.split(' ').list.eval(pl.element().filter(pl.element() != ''))})
    df = df.with_columns(pl.col('a').list.unique().alias('a'), pl.col('b').list.unique().alias('b'))
    df = df.with_columns(pl.col('a').list.set_intersection('b').list.len().alias('i'),
                         pl.col('a').list.len().alias('la'), pl.col('b').list.len().alias('lb'))
    i, la, lb = df['i'].to_numpy().astype(np.float32), df['la'].to_numpy().astype(np.float32), df['lb'].to_numpy().astype(np.float32)
    u = la + lb - i
    with np.errstate(divide='ignore', invalid='ignore'):
        return {f'{prefix}_inter': i, f'{prefix}_jacc': np.where(u > 0, i / u, np.nan),
                f'{prefix}_cont_q': np.where(la > 0, i / la, np.nan), f'{prefix}_cont_c': np.where(lb > 0, i / lb, np.nan),
                f'{prefix}_len_q': la, f'{prefix}_len_c': lb}


def _per_country_tfidf(q, p, col):
    """Block-diagonal TF-IDF: one vectorizer per country, columns offset per country."""
    import scipy.sparse as sp
    qr, qc, qv, pr, pc, pv, off = [], [], [], [], [], [], 0
    qcty, pcty = q['country_n'].to_numpy(), p['country_n'].to_numpy()
    for c in np.unique(np.concatenate([qcty, pcty])):
        iq, ip = np.flatnonzero(qcty == c), np.flatnonzero(pcty == c)
        v = TfidfVectorizer(token_pattern=r'\S+', sublinear_tf=True, dtype=np.float32, min_df=1)
        v.fit(p[col].gather(ip).to_list() + q[col].gather(iq).to_list())
        for idx, df, R, C, V in ((iq, q, qr, qc, qv), (ip, p, pr, pc, pv)):
            if len(idx):
                X = v.transform(df[col].gather(idx).to_list()).tocoo()
                R.append(idx[X.row]); C.append(X.col + off); V.append(X.data)
        off += len(v.vocabulary_)
    mk = lambda R, C, V, n: sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, off), dtype=np.float32)
    return mk(qr, qc, qv, q.height), mk(pr, pc, pv, p.height)


class Featurizer:
    """Holds IDF vectorizers fit once on the pool so train/val/test use identical transforms."""

    def __init__(self, q: pl.DataFrame, p: pl.DataFrame):
        # TF-IDF and name frequency are fit PER COUNTRY (open set of labels): word rarity means the same thing
        # in a new country (e.g. France) as in the training countries.
        self.Qn, self.Pn = _per_country_tfidf(q, p, 'name_core')
        self.Qa, self.Pa = _per_country_tfidf(q, p, 'addr_norm')
        allc = pl.concat([q.select('country_n', 'name_core'), p.select('country_n', 'name_core')])
        freq = allc.group_by('country_n', 'name_core').len()
        size = p.group_by('country_n').agg(pl.len().alias('_sz'))
        fq = q.select('country_n', 'name_core').join(freq, on=['country_n', 'name_core'], how='left').join(size, on='country_n', how='left')
        fp = p.select('country_n', 'name_core').join(freq, on=['country_n', 'name_core'], how='left').join(size, on='country_n', how='left')
        self.q_nfreq, self.p_nfreq = fq['len'].to_numpy(), fp['len'].to_numpy()
        self.q_nfreq_rel = (fq['len'] / fq['_sz'] * 1e6).to_numpy()
        self.p_nfreq_rel = (fp['len'] / fp['_sz'] * 1e6).to_numpy()
        self.q, self.p = q, p
        self.q_legal, self.p_legal = legal_set(q['name_norm']), legal_set(p['name_norm'])

    def group_features(self, qi: np.ndarray, pi: np.ndarray, s0: np.ndarray) -> pl.DataFrame:
        """Candidate-vs-candidate context inside each S1 group: is this candidate a near-copy of the
        group's best-scoring OTHER candidate, and how many candidates share its address / name."""
        p = self.p
        g = pl.DataFrame({'r': np.arange(len(qi)), 'qi': qi, 'pi': pi, 's0': s0,
                          'n': p['name_core'].gather(pi), 'a': p['addr_norm'].gather(pi)})
        g = g.with_columns(pl.col('s0').rank('ordinal', descending=True).over('qi').alias('rk'))
        best = g.filter(pl.col('rk') == 1).select('qi', pl.col('n').alias('n1'), pl.col('a').alias('a1'))
        second = g.filter(pl.col('rk') == 2).select('qi', pl.col('n').alias('n2'), pl.col('a').alias('a2'))
        g = g.join(best, on='qi', how='left').join(second, on='qi', how='left').sort('r')
        on = pl.when(pl.col('rk') == 1).then(pl.col('n2')).otherwise(pl.col('n1')).fill_null('')
        oa = pl.when(pl.col('rk') == 1).then(pl.col('a2')).otherwise(pl.col('a1')).fill_null('')
        g = g.with_columns(on.alias('on'), oa.alias('oa'),
                           pl.len().over(['qi', 'a']).alias('g_same_addr'), pl.len().over(['qi', 'n']).alias('g_same_name'))
        alone = (pl.len().over('qi') == 1)
        has_other = (~g.select(alone.alias('x'))['x']).to_numpy()
        return pl.DataFrame({
            'cvc_name': np.where(has_other, _rf(g['n'].to_list(), g['on'].to_list(), fuzz.token_sort_ratio), np.nan).astype(np.float32),
            'cvc_addr': np.where(has_other, _rf(g['a'].to_list(), g['oa'].to_list(), fuzz.token_sort_ratio), np.nan).astype(np.float32),
            'g_same_addr': g['g_same_addr'].cast(pl.Float32).to_numpy(),
            'g_same_name': g['g_same_name'].cast(pl.Float32).to_numpy(),
        })

    def pair_features(self, qi: np.ndarray, pi: np.ndarray) -> pl.DataFrame:
        q, p = self.q, self.p
        g = lambda df, col, idx: df[col].gather(idx)
        nq, nc = g(q, 'name_core', qi), g(p, 'name_core', pi)
        aq, ac = g(q, 'addr_norm', qi), g(p, 'addr_norm', pi)
        nql, ncl = nq.to_list(), nc.to_list()
        aql, acl = aq.to_list(), ac.to_list()
        F = {}
        # ---- name
        F['n_ratio'] = _rf(nql, ncl, fuzz.ratio)
        F['n_pratio'] = _rf(nql, ncl, fuzz.partial_ratio)
        F['n_tsort'] = _rf(nql, ncl, fuzz.token_sort_ratio)
        F['n_tset'] = _rf(nql, ncl, fuzz.token_set_ratio)
        F['n_jw'] = _rf(nql, ncl, JaroWinkler.normalized_similarity)
        F['n_lev'] = _rf(nql, ncl, Levenshtein.distance)
        nsq = nq.str.replace_all(' ', '').to_list(); nsc = nc.str.replace_all(' ', '').to_list()
        F['n_nospace_ratio'] = _rf(nsq, nsc, fuzz.ratio)
        F['n_nospace_pratio'] = _rf(nsq, nsc, fuzz.partial_ratio)
        F['nfull_tsort'] = _rf(g(q, 'name_norm', qi).to_list(), g(p, 'name_norm', pi).to_list(), fuzz.token_sort_ratio)
        # best alias part of the candidate (d/b/a, formerly ...)
        al = g(p, 'name_alias', pi).str.split('|')
        ex = pl.DataFrame({'r': np.arange(len(pi)), 'a': al}).explode('a')
        sc = _rf(nq.gather(ex['r'].to_numpy()).to_list(), ex['a'].fill_null('').to_list(), fuzz.token_sort_ratio)
        F['n_alias_best'] = pl.DataFrame({'r': ex['r'], 's': sc}).group_by('r').agg(pl.col('s').max()).sort('r')['s'].to_numpy()
        F.update(_set_feats(nq, nc, 'ntok'))
        F['n_tfidf'] = _rowdot(self.Qn, self.Pn, qi, pi)
        F['n_empty_c'] = (nc.str.len_chars() == 0).to_numpy().astype(np.float32)
        F['n_freq_q'] = np.log1p(self.q_nfreq[qi]).astype(np.float32)
        F['n_freq_c'] = np.log1p(self.p_nfreq[pi]).astype(np.float32)
        F['n_freq_rel_q'] = np.log1p(self.q_nfreq_rel[qi]).astype(np.float32)
        F['n_freq_rel_c'] = np.log1p(self.p_nfreq_rel[pi]).astype(np.float32)
        lq, lc = self.q_legal.gather(qi), self.p_legal.gather(pi)
        has_q, has_c = (lq != '').to_numpy(), (lc != '').to_numpy()
        F['legal_both'] = (has_q & has_c).astype(np.float32)
        F['legal_eq'] = np.where(has_q & has_c, (lq == lc).to_numpy(), np.nan).astype(np.float32)
        F['legal_any'] = (has_q | has_c).astype(np.float32)
        lsf = _set_feats(lq, lc, 'lg')
        F['legal_jacc'] = lsf['lg_jacc']
        # ---- address
        F['a_empty_q'] = (aq.str.len_chars() == 0).to_numpy().astype(np.float32)
        F['a_empty_c'] = (ac.str.len_chars() == 0).to_numpy().astype(np.float32)
        F['a_ratio'] = _rf(aql, acl, fuzz.ratio)
        F['a_pratio'] = _rf(aql, acl, fuzz.partial_ratio)
        F['a_tsort'] = _rf(aql, acl, fuzz.token_sort_ratio)
        F['a_tset'] = _rf(aql, acl, fuzz.token_set_ratio)
        F.update(_set_feats(aq, ac, 'atok'))
        F['a_tfidf'] = _rowdot(self.Qa, self.Pa, qi, pi)
        mq, mc = g(q, 'addr_nums', qi), g(p, 'addr_nums', pi)
        F.update(_set_feats(mq, mc, 'num'))
        f1q, f1c = mq.str.split(' ').list.first(), mc.str.split(' ').list.first()
        both = ((f1q != '') & (f1c != '')).to_numpy()
        F['num_first_eq'] = np.where(both, (f1q == f1c).to_numpy(), np.nan).astype(np.float32)
        F['num_first_lev'] = np.where(both, _rf(f1q.to_list(), f1c.to_list(), Levenshtein.distance), np.nan).astype(np.float32)
        F['num_first_pfx'] = np.where(both, _rf(f1q.to_list(), f1c.to_list(), fuzz.partial_ratio), np.nan).astype(np.float32)
        # distractors often shift the house number slightly (80 vs 89, 815 vs 817)
        nfq = pl.Series(f1q).cast(pl.Float64, strict=False).to_numpy()
        nfc = pl.Series(f1c).cast(pl.Float64, strict=False).to_numpy()
        F['num_first_absdiff'] = np.where(both, np.abs(nfq - nfc), np.nan).astype(np.float32)
        # ---- name edits typical of distractors: changed last token (VI -> VIJ), added/removed legal form
        lq_t, lc_t = nq.str.split(' ').list.last(), nc.str.split(' ').list.last()
        F['n_last_eq'] = (lq_t == lc_t).fill_null(False).to_numpy().astype(np.float32)
        F['n_first_eq'] = (nq.str.split(' ').list.first() == nc.str.split(' ').list.first()).fill_null(False).to_numpy().astype(np.float32)
        F['legal_only_c'] = (~has_q & has_c).astype(np.float32)
        F['legal_only_q'] = (has_q & ~has_c).astype(np.float32)
        F['ntok_extra_c'] = F['ntok_len_c'] - F['ntok_inter']
        F['ntok_missing_q'] = F['ntok_len_q'] - F['ntok_inter']
        sq, scs = g(q, 'addr_state', qi), g(p, 'addr_state', pi)
        F['state_eq'] = np.where(((sq != '') & (scs != '')).to_numpy(), (sq == scs).to_numpy(), np.nan).astype(np.float32)
        # address words without numbers (street + city)
        wq = aq.str.replace_all(r'\b\d+\b', '').str.replace_all(r'\s+', ' ').str.strip_chars()
        wc = ac.str.replace_all(r'\b\d+\b', '').str.replace_all(r'\s+', ' ').str.strip_chars()
        F['aw_tset'] = _rf(wq.to_list(), wc.to_list(), fuzz.token_set_ratio)
        # ---- cross: name vs whole candidate string (fields sometimes shifted/merged)
        F['is_s3'] = g(p, 'entity_id', pi).str.starts_with('S3').to_numpy().astype(np.float32)
        return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})
