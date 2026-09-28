"""E05: decorrelated model E04 + union candidate set + 3-model agreement ensemble (E02, S3, E04).

Design (see docs/audit_consensus.md, C1-C5):
  * Candidate set = union of E04's stage-0 top-N and the LEGACY pool (E02/S3 stage-0 model top-20, same model
    both used) and, on test, S3's actual candidate_pairs.tsv (forced in). candidate_pairs.tsv is this union.
  * E04 scores the whole union. E02/S3 re-score only the legacy pool with legacy group statistics (exactly
    their training conditions); their vote on E04-only pairs is "no" (documented design choice).
  * E04 drops density/blocking-source-dependent features and up-weights negatives x2 (train/test distractor
    density gap: pool per S1 4.67 train vs 5.5-5.8 test -> ~1.2 vs ~2.3 distractors per S1).
"""
import os, json, pickle, time
import numpy as np
import polars as pl
import lightgbm as lgb
import pipeline as P, stage0, decide as D
from candidates import CFG
from features import Featurizer
from run_train import sweep, apply_rule
from common import WORK

ARCH = os.path.join(WORK, 'models_archive')
LEG_PASSES = ['comb', 'addr', 'name_ch', 'rev', 'nameaddr', 'rev_na', 'k_name_num', 'k_name']   # E02/S3 blocking
TOP_N = 20
NEG_WEIGHT = 2.0
E04_DROP = ['cvc_name', 'cvc_addr', 'g_same_addr', 'g_same_name', 'n_freq_q', 'n_freq_c', 'g_size',
            's0_q_n', 's0_p_n', 's0_q_sum', 'in_e04', 'in_leg', 'forced']


def leg_feat_cols():
    return ([f'{n}_score' for n in LEG_PASSES] + [f'{n}_rank' for n in LEG_PASSES] +
            ['n_passes', 'c_n_tset', 'c_n_ratio', 'c_a_tset', 'c_a_ratio', 'c_num1_eq', 'c_a_empty',
             'g_comb_rank', 'g_size', 'g_sim_rank'])


def _leg_view(c: pl.DataFrame) -> pl.DataFrame:
    """Columns as the legacy models saw them: n_passes over the 8 legacy passes."""
    return c.with_columns(pl.sum_horizontal([(pl.col(f'{n}_score') > 0) for n in LEG_PASSES]).cast(pl.Int64).alias('n_passes'))


def _oof_or_avg(models, X, fold, split):
    if split == 'train':
        s = np.zeros(len(X), np.float32)
        for k in range(P.NF):
            m = fold == k
            if m.any():
                s[m] = models[k].predict(X[m], num_threads=14)
        return s
    return np.mean([m.predict(X, num_threads=14) for m in models], axis=0).astype(np.float32)


# ------------------------------------------------------------------ stage 0: union candidate set
def run_stage0_union(split, q, p, extra_pairs: pl.DataFrame = None, sample_frac=0.15):
    """extra_pairs (qi, pi): candidates forced into the legacy pool (test: S3's candidate file)."""
    t = time.time()
    lab = P.labels(q, p) if split == 'train' else None
    for f in P.parts(split, 'cand'):
        out = f.replace(f'cand_{split}', f'c0_{split}')
        if os.path.exists(out):
            continue
        c = pl.read_parquet(f).with_columns(pl.lit(0, pl.Int8).alias('forced'))
        if extra_pairs is not None:
            qs = c['qi'].unique()
            ex = extra_pairs.filter(pl.col('qi').is_in(qs.implode())).join(c.select('qi', 'pi'), on=['qi', 'pi'], how='anti')
            if ex.height:
                c = pl.concat([c, ex.with_columns(pl.lit(1, pl.Int8).alias('forced'))], how='diagonal_relaxed')
        c = c.with_columns(pl.col('n_passes').fill_null(0))
        c = stage0.cheap_features(c, q, p).with_columns(P.fold_of(pl.col('qi')).alias('fold'))
        if lab is not None:
            c = c.join(lab, on=['qi', 'pi'], how='left').with_columns(pl.col('label').fill_null(0))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        c.write_parquet(out)
    print(f'[stage0-union] cheap features {time.time() - t:.0f}s', flush=True)
    c0 = P.parts(split, 'c0')
    mpath = os.path.join(P.MODELS, 'stage0.pkl')
    if split == 'train' and not os.path.exists(mpath):
        smp = pl.concat([pl.read_parquet(f).filter((pl.col('qi').hash(seed=3) % 1000) < sample_frac * 1000) for f in c0])
        pickle.dump([stage0.train(smp.filter(pl.col('fold') != k)) for k in range(P.NF)], open(mpath, 'wb'))
    m_e04 = pickle.load(open(mpath, 'rb'))
    m_leg = pickle.load(open(os.path.join(ARCH, 'E02', 'stage0.pkl'), 'rb'))
    P.d(split, 'pr')
    hits = {}
    for f in c0:
        c = pl.read_parquet(f)
        fo = c['fold'].to_numpy()
        c = c.with_columns(pl.Series('s0', _oof_or_avg(m_e04, c.select(stage0.feat_cols()).to_numpy(), fo, split)),
                           pl.Series('s0L', _oof_or_avg(m_leg, c.select(leg_feat_cols()).to_numpy(), fo, split)))
        c = c.with_columns(pl.col('s0').rank('ordinal', descending=True).over('qi').alias('_re'),
                           pl.col('s0L').rank('ordinal', descending=True).over('qi').alias('_rl'))
        c = c.with_columns((pl.col('_re') <= TOP_N).cast(pl.Int8).alias('in_e04'),
                           ((pl.col('_rl') <= TOP_N) | (pl.col('forced') == 1)).cast(pl.Int8).alias('in_leg'))
        if 'label' in c.columns:
            for k, cond in (('union_blocking', pl.lit(True)), ('e04_top20', pl.col('in_e04') == 1),
                            ('legacy_top20', pl.col('in_leg') == 1), ('final_union', (pl.col('in_e04') == 1) | (pl.col('in_leg') == 1))):
                hits[k] = hits.get(k, 0) + int(c.filter(cond & (pl.col('label') == 1)).height)
                hits[k + '_pairs'] = hits.get(k + '_pairs', 0) + int(c.filter(cond).height)
        c = c.filter((pl.col('in_e04') == 1) | (pl.col('in_leg') == 1)).drop('_re', '_rl')
        c.write_parquet(f.replace(f'c0_{split}', f'pr_{split}'))
    if hits:
        print('[stage0-union] true pairs / pairs by candidate pool:', hits, flush=True)
    print(f'[stage0-union] done {time.time() - t:.0f}s', flush=True)


# ------------------------------------------------------------------ features (computed once)
def run_features_union(split, q, p):
    t = time.time()
    keys = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 's0', 's0L', 'in_leg']) for f in P.parts(split, 'pr')])
    ctx = P.group_stats(keys.select('qi', 'pi', 's0'), 's0', 's0').drop('s0')
    leg = keys.filter(pl.col('in_leg') == 1).select('qi', 'pi', pl.col('s0L').alias('s0'))
    ctx_leg = P.group_stats(leg, 's0', 's0').drop('s0')
    ctx_leg = ctx_leg.rename({c: c.replace('s0_', 'L_s0_') for c in ctx_leg.columns if c.startswith('s0_')})
    need = ['entity_id', 'country_n', 'name_norm', 'name_core', 'name_alias', 'addr_norm', 'addr_nums', 'addr_state']
    fz = Featurizer(q.select(need), p.select(need))
    print(f'[features-union] featurizer ready {time.time() - t:.0f}s pairs={keys.height} legacy={leg.height}', flush=True)
    del keys, leg
    for f in P.parts(split, 'pr'):
        out = f.replace(f'pr_{split}', f'feat_{split}')
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        c = pl.read_parquet(f).join(ctx, on=['qi', 'pi'], how='left').join(ctx_leg, on=['qi', 'pi'], how='left')
        F = fz.pair_features(c['qi'].to_numpy(), c['pi'].to_numpy())
        c = pl.concat([c, F], how='horizontal').with_row_index('_r')
        lg = c.filter(pl.col('in_leg') == 1)
        G = fz.group_features(lg['qi'].to_numpy(), lg['pi'].to_numpy(), lg['s0L'].to_numpy())   # legacy groups only
        c = c.join(pl.concat([lg.select('_r'), G], how='horizontal'), on='_r', how='left').drop('_r')
        c.write_parquet(out)
        print(f'  features {os.path.basename(f)} rows={c.height} {time.time() - t:.0f}s', flush=True)


# ------------------------------------------------------------------ E04 training / scoring
def e04_cols(df):
    return [c for c in P.fcols(df, E04_DROP) if not c.startswith('L_s0') and c != 's0L']


def train_e04(stage, extra=None, rounds=1000, sample_frac=0.3):
    t = time.time()
    smp = pl.concat([c.filter((pl.col('qi').hash(seed=11) % 1000) < sample_frac * 1000) for c in P.iter_parts('train', extra)])
    cols = e04_cols(smp)
    if stage == 'stage2_e04':
        cols = [c for c in cols if c not in ('p1_q_n', 'p1_p_n', 'p1_q_sum')]
    models = []
    for k in range(P.NF):
        tr = smp.filter(pl.col('fold') != k)
        y = tr['label'].to_numpy()
        w = np.where(y == 1, 1.0, NEG_WEIGHT).astype(np.float32)
        models.append(lgb.train(P.P1, lgb.Dataset(tr.select(cols).to_numpy(), y, weight=w, feature_name=cols), rounds))
        print(f'  {stage} fold {k}: {tr.height} rows, {time.time() - t:.0f}s', flush=True)
    pickle.dump((models, cols), open(os.path.join(P.MODELS, f'{stage}.pkl'), 'wb'))


def predict_e04(split, stage, extra=None):
    models, cols = pickle.load(open(os.path.join(P.MODELS, f'{stage}.pkl'), 'rb'))
    out = []
    for c in P.iter_parts(split, extra):
        s = _oof_or_avg(models, c.select(cols).to_numpy(), c['fold'].to_numpy() if 'fold' in c.columns else None, split)
        keep = ['qi', 'pi', 'in_e04', 'in_leg', 'forced'] + (['label'] if 'label' in c.columns else [])
        out.append(c.select(keep).with_columns(pl.Series(stage, s)))
    return pl.concat(out)


# ------------------------------------------------------------------ legacy models (E02, S3) on the legacy pool
def score_legacy(split, name):
    """Returns (qi, pi, label?, score) for the legacy pool, scored exactly as at the model's training time."""
    d = os.path.join(ARCH, name)
    m1, c1 = pickle.load(open(os.path.join(d, 'stage1.pkl'), 'rb'))
    ren = lambda c: c.drop([x for x in c.columns if x.startswith('s0_')] + ['s0']).rename(
        {'s0L': 's0', **{x: x[2:] for x in c.columns if x.startswith('L_s0_')}})
    out = []
    for c in P.iter_parts(split):
        c = _leg_view(ren(c.filter(pl.col('in_leg') == 1)))
        s = _oof_or_avg(m1, c.select(c1).to_numpy(), c['fold'].to_numpy() if 'fold' in c.columns else None, split)
        out.append((c, s))
    s1 = pl.concat([c.select(['qi', 'pi'] + (['label'] if 'label' in c.columns else [])).with_columns(pl.Series('p', s)) for c, s in out])
    if not os.path.exists(os.path.join(d, 'stage2.pkl')):
        return s1
    m2, c2 = pickle.load(open(os.path.join(d, 'stage2.pkl'), 'rb'))
    ctx = P.group_stats(s1.select('qi', 'pi', pl.col('p').alias('stage1')), 'stage1', 'p1')
    res = []
    for c, _ in out:
        c = c.join(ctx, on=['qi', 'pi'], how='left')
        s = _oof_or_avg(m2, c.select(c2).to_numpy(), c['fold'].to_numpy() if 'fold' in c.columns else None, split)
        res.append(c.select(['qi', 'pi'] + (['label'] if 'label' in c.columns else [])).with_columns(pl.Series('p', s)))
    return pl.concat(res)


def legacy_votes(scores: pl.DataFrame, name):
    rule = json.load(open(os.path.join(ARCH, name, 'decision_stage1.json' if name == 'E02' else 'decision.json')))
    return apply_rule(scores, 'p', rule).select('qi', 'pi')


# ------------------------------------------------------------------ ensemble rules
def ensemble(v04, v02, v3, rule):
    k = ['qi', 'pi']
    if rule == 'e04':
        return v04.select(k)
    if rule == 'majority':
        allv = pl.concat([v04.select(k), v02.select(k), v3.select(k)])
        return allv.group_by(k).len().filter(pl.col('len') >= 2).select(k)
    if rule == 'e04_and_any':
        return v04.select(k).join(pl.concat([v02.select(k), v3.select(k)]).unique(), on=k, how='semi')
    if rule == 'e02_and_s3':
        return v02.select(k).join(v3.select(k), on=k, how='semi')
    raise ValueError(rule)
