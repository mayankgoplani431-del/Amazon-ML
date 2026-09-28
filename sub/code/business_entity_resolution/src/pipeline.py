"""End-to-end driver after blocking (candidates.py):

  stage0  : cheap features + pruning ranker  -> final candidate set (top-N per S1)
  context : S1-side and candidate-side competition statistics of the stage-0 score
  features: full pairwise features on the final candidate set
  stage1  : LightGBM matcher (2-fold out-of-fold on train, fold-average on test)
  stage2  : LightGBM re-scorer with competition features of stage-1 probabilities
Decisions (thresholds / exclusivity) live in decide.py.
"""
import os, sys, glob, time, json, pickle
import numpy as np
import polars as pl
import lightgbm as lgb
from common import CACHE, WORK, load_gt
import candidates, stage0, cleanup
from features import Featurizer

NF = 2
MODELS = os.path.join(WORK, 'models')
os.makedirs(MODELS, exist_ok=True)


def d(split, name):
    p = os.path.join(CACHE, f'{name}_{split}')
    os.makedirs(p, exist_ok=True)
    return p


def parts(split, name):
    return sorted(glob.glob(os.path.join(d(split, name), 'part_*.parquet')))


def labels(q, p):
    gt = load_gt()
    return (gt.join(q.select(pl.col('entity_id').alias('s1'), pl.col('idx_q').alias('qi')), on='s1')
              .join(p.select(pl.col('entity_id').alias('cand'), pl.col('idx_p').alias('pi')), on='cand')
              .select('qi', 'pi').with_columns(pl.lit(1, pl.Int8).alias('label')))


def fold_of(qi: pl.Expr):
    return (qi.hash(seed=7) % NF).cast(pl.Int8)


# ---------------------------------------------------------------- stage 0
def run_stage0(split, q, p, top_n=20, sample_frac=0.15):
    if parts(split, 'pr') and not parts(split, 'c0'):
        print(f'stage0: pruned candidates for {split} already exist, skipping', flush=True)
        return
    lab = labels(q, p) if split == 'train' else None
    t = time.time()
    for f in parts(split, 'cand'):
        out = f.replace(f'cand_{split}', f'c0_{split}')
        if os.path.exists(out):
            continue
        c = stage0.cheap_features(pl.read_parquet(f), q, p).with_columns(fold_of(pl.col('qi')).alias('fold'))
        if lab is not None:
            c = c.join(lab, on=['qi', 'pi'], how='left').with_columns(pl.col('label').fill_null(0))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        c.write_parquet(out)
    print(f'stage0 cheap features {time.time() - t:.0f}s', flush=True)
    c0 = parts(split, 'c0')
    mpath = os.path.join(MODELS, 'stage0.pkl')
    d(split, 'pr')
    if split == 'train' and not os.path.exists(mpath):
        smp = pl.concat([pl.read_parquet(f).filter((pl.col('qi').hash(seed=3) % 1000) < sample_frac * 1000) for f in c0])
        models = [stage0.train(smp.filter(pl.col('fold') != k)) for k in range(NF)]
        pickle.dump(models, open(mpath, 'wb'))
    models = pickle.load(open(mpath, 'rb'))
    hits = {}
    for f in c0:
        c = pl.read_parquet(f)
        if split == 'train':   # out-of-fold scores
            s = np.zeros(c.height, np.float32)
            for k in range(NF):
                m = (c['fold'] == k).to_numpy()
                s[m] = stage0.predict(models[k], c.filter(pl.col('fold') == k))
        else:
            s = np.mean([stage0.predict(m, c) for m in models], axis=0)
        c = c.with_columns(pl.Series('s0', s))
        if 'label' in c.columns:   # recall of the pruned set as a function of N
            r = c.with_columns(pl.col('s0').rank('ordinal', descending=True).over('qi').alias('_r')).filter(pl.col('label') == 1)['_r']
            for n in (5, 10, 15, 20, 30, 50):
                hits[n] = hits.get(n, 0) + int((r <= n).sum())
            hits['union'] = hits.get('union', 0) + len(r)
            hits['union_pairs'] = hits.get('union_pairs', 0) + c.height
        c = stage0.prune(c, 's0', top_n)
        c.write_parquet(f.replace(f'c0_{split}', f'pr_{split}'))
    if hits:
        print('stage0 pruned-set true pairs kept by N:', hits, flush=True)
    print(f'stage0 done {time.time() - t:.0f}s', flush=True)
    cleanup.after_prune(split)


# ---------------------------------------------------------------- context stats
def group_stats(df: pl.DataFrame, col: str, pfx: str) -> pl.DataFrame:
    """Competition features of score `col` within the S1 group (qi) and the candidate group (pi)."""
    return df.with_columns(
        pl.col(col).rank('ordinal', descending=True).over('qi').cast(pl.Float32).alias(f'{pfx}_q_rank'),
        (pl.col(col) - pl.col(col).max().over('qi')).alias(f'{pfx}_q_gap_max'),
        pl.col(col).sum().over('qi').alias(f'{pfx}_q_sum'),
        pl.len().over('qi').cast(pl.Float32).alias(f'{pfx}_q_n'),
        pl.col(col).rank('ordinal', descending=True).over('pi').cast(pl.Float32).alias(f'{pfx}_p_rank'),
        (pl.col(col) - pl.col(col).max().over('pi')).alias(f'{pfx}_p_gap_max'),
        pl.len().over('pi').cast(pl.Float32).alias(f'{pfx}_p_n'),
        # second-best score in the S1 group (margin for the top candidate)
        (pl.col(col) - pl.col(col).sort(descending=True).slice(1, 1).first().over('qi')).alias(f'{pfx}_q_gap_2nd'),
        (pl.col(col) - pl.col(col).sort(descending=True).slice(1, 1).first().over('pi')).alias(f'{pfx}_p_gap_2nd'),
    )


# ---------------------------------------------------------------- features
def run_features(split, q, p):
    """One feature part per pruned part; competition stats of s0 are computed globally first."""
    t = time.time()
    keys = pl.concat([pl.read_parquet(f, columns=['qi', 'pi', 's0']) for f in parts(split, 'pr')])
    ctx = group_stats(keys, 's0', 's0').drop('s0')
    need = ['entity_id', 'country_n', 'name_norm', 'name_core', 'name_alias', 'addr_norm', 'addr_nums', 'addr_state']
    fz = Featurizer(q.select(need), p.select(need))   # raw text columns not kept in memory
    print(f'featurizer ready {time.time() - t:.0f}s pairs={keys.height}', flush=True)
    del keys
    for f in parts(split, 'pr'):
        out = f.replace(f'pr_{split}', f'feat_{split}')
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        c = pl.read_parquet(f).join(ctx, on=['qi', 'pi'], how='left')
        F = fz.pair_features(c['qi'].to_numpy(), c['pi'].to_numpy())
        G = fz.group_features(c['qi'].to_numpy(), c['pi'].to_numpy(), c['s0'].to_numpy())
        pl.concat([c, F, G], how='horizontal').write_parquet(out)
        print(f'  features {os.path.basename(f)} rows={c.height} {time.time() - t:.0f}s', flush=True)


NON_FEAT = {'qi', 'pi', 'label', 'fold', 'p1', 'p2'}
P1 = dict(objective='binary', learning_rate=0.05, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
          bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14, seed=0)


def fcols(df, drop=()):
    return [c for c in df.columns if c not in NON_FEAT and c not in drop]


def iter_parts(split, extra=None):
    """Yields feature parts, optionally joined with an extra (qi, pi, ...) frame."""
    for f in parts(split, 'feat'):
        c = pl.read_parquet(f)
        if extra is not None:
            c = c.join(extra, on=['qi', 'pi'], how='left')
        yield c


def train_stage(stage, extra=None, drop=(), rounds=1000, sample_frac=0.3, params=P1):
    """Trains NF fold models on a sample of S1 groups (all their candidates = natural hard negatives)."""
    t = time.time()
    smp = pl.concat([c.filter((pl.col('qi').hash(seed=11) % 1000) < sample_frac * 1000) for c in iter_parts('train', extra)])
    cols = fcols(smp, drop)
    models = []
    for k in range(NF):
        tr = smp.filter(pl.col('fold') != k)
        m = lgb.train(params, lgb.Dataset(tr.select(cols).to_numpy(), tr['label'].to_numpy(), feature_name=cols), rounds)
        models.append(m)
        print(f'  {stage} fold {k}: {tr.height} rows, {time.time() - t:.0f}s', flush=True)
    pickle.dump((models, cols), open(os.path.join(MODELS, f'{stage}.pkl'), 'wb'))
    return models, cols


def predict_stage(split, stage, extra=None):
    """Train: out-of-fold scores. Test: average of fold models. Returns (qi, pi, label?, score)."""
    models, cols = pickle.load(open(os.path.join(MODELS, f'{stage}.pkl'), 'rb'))
    out = []
    for c in iter_parts(split, extra):
        X = c.select(cols).to_numpy()
        if split == 'train':
            s = np.zeros(c.height, np.float32)
            fo = c['fold'].to_numpy()
            for k in range(NF):
                m = fo == k
                if m.any():
                    s[m] = models[k].predict(X[m], num_threads=14)
        else:
            s = np.mean([m.predict(X, num_threads=14) for m in models], axis=0).astype(np.float32)
        keep = ['qi', 'pi'] + (['label'] if 'label' in c.columns else [])
        out.append(c.select(keep).with_columns(pl.Series(stage, s)))
    return pl.concat(out)


# ---- domain-robustness helpers (unseen countries) ---------------------------------------------
MONO_POS = ('n_ratio n_pratio n_tsort n_tset n_jw n_nospace_ratio n_nospace_pratio nfull_tsort n_alias_best ntok_jacc '
            'n_tfidf a_ratio a_pratio a_tsort a_tset atok_jacc a_tfidf num_jacc num_first_eq aw_tset c_n_tset c_n_ratio '
            'c_a_tset c_a_ratio c_num1_eq s0 s0_q_gap_max s0_p_gap_max n_last_eq n_first_eq').split()
MONO_NEG = 'n_lev num_first_lev num_first_absdiff s0_q_rank s0_p_rank'.split()


def monotone(cols):
    return [1 if c in MONO_POS else -1 if c in MONO_NEG else 0 for c in cols]


def country_rank_norm(df: pl.DataFrame, cols, group='country_n') -> pl.DataFrame:
    """Percentile of every feature within its own country (label-free, transductive)."""
    return df.with_columns([(pl.col(c).rank('average').over(group) / pl.len().over(group)).cast(pl.Float32).alias(c)
                            for c in cols])

