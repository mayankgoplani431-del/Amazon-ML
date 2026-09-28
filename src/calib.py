"""F2: strictness for an unseen country, chosen label-free by matching the in-domain prediction profile.

The generator uses the same match-count prior in every training country (US and India: 5.6% singletons,
3.46 matches/S1), so an unseen country whose model predicts MORE matches / FEWER empties than the in-domain
countries is over-matching. We shift its probabilities by one logit offset delta until its predicted profile
(empty rate, matches per S1) moves a fraction `frac` of the way to the in-domain profile.

  python calib.py loco [train_country] [eval_country]
      honest simulation of France: train on 30% of one country, tune the decision rule on held-out S1 of that
      SAME country, apply to the other country as if it were unseen. Prints F0.5 (true labels) for every delta
      and for the label-free profile-matched delta -> tells whether the method helps and which frac to use.
  python calib.py test <out_dir> <country> <frac>
      applies it on test to <country> (e.g. france) using work/cache/test_stage2.parquet + models/decision.json.
"""
import os, sys, json
import numpy as np
import polars as pl
import lightgbm as lgb
import pipeline as P, decide as D
from common import load_norm
from run_train import sweep, apply_rule

GRID = [round(x, 2) for x in np.arange(-0.5, 3.01, 0.25)]


def shift(s, col, delta):
    """Logit offset: p' = sigmoid(logit(p) - delta). delta > 0 = stricter."""
    lg = (pl.col(col).clip(1e-6, 1 - 1e-6) / (1 - pl.col(col).clip(1e-6, 1 - 1e-6))).log()
    return s.with_columns((1 / (1 + (-(lg - delta)).exp())).cast(pl.Float32).alias(col))


def profile(pred, qis):
    k = qis.join(pred.group_by('qi').len(), on='qi', how='left').with_columns(pl.col('len').fill_null(0))['len']
    return float((k == 0).mean()), float(k.mean())


def pick(curve, target, frac):
    """curve: [(delta, empty, mean_k)] of the unseen country; target: in-domain (empty, mean_k).
    Goal = own delta=0 profile moved `frac` of the way to the target; each dimension measured in units of its own
    gap to the target (relative-to-value units let the small empty rate dominate)."""
    e0, k0 = next((e, k) for d, e, k in curve if d == 0)
    ge, gk = e0 + frac * (target[0] - e0), k0 + frac * (target[1] - k0)
    se, sk = max(abs(target[0] - e0), 1e-4), max(abs(target[1] - k0), 1e-3)
    return min(curve, key=lambda r: abs(r[1] - ge) / se + abs(r[2] - gk) / sk)[0]


def loco(tr_c='us', ev_c='india'):
    TR, EV = int(os.environ.get('LOCO_TR', '100')), int(os.environ.get('LOCO_EV', '250'))   # per-mille samples: fits 16 GB
    ctry = load_norm('train', 1).select(pl.int_range(pl.len()).cast(pl.UInt32).alias('qi'), 'country_n')
    h = lambda: pl.col('qi').hash(seed=11) % 1000
    tr = []
    for c in P.iter_parts('train'):
        c = c.with_columns(pl.col('qi').cast(pl.UInt32)).join(ctry, on='qi')
        tr.append(c.filter((pl.col('country_n') == tr_c) & (h() < TR)).drop('country_n'))
    tr = pl.concat(tr)
    cols = P.fcols(tr, [c for c in os.environ.get('DROP', '').split(',') if c])
    m = lgb.train(P.P1, lgb.Dataset(tr.select(cols).cast(pl.Float32).to_numpy(), tr['label'].to_numpy(), feature_name=cols), 1000)
    del tr
    hold, ev = [], []   # predict part by part: keeps memory flat
    for c in P.iter_parts('train'):
        c = c.with_columns(pl.col('qi').cast(pl.UInt32)).join(ctry, on='qi')
        for keep, lst in (((pl.col('country_n') == tr_c) & (h() >= 300) & (h() < 300 + EV), hold), ((pl.col('country_n') == ev_c) & (h() < EV), ev)):
            x = c.filter(keep)
            lst.append(x.select('qi', 'pi', 'label').with_columns(pl.Series('p', m.predict(x.select(cols).cast(pl.Float32).to_numpy(), num_threads=14).astype(np.float32))))
    hold, ev = pl.concat(hold), pl.concat(ev)
    tc = os.path.join(P.CACHE, 'truth_counts.parquet')
    if not os.path.exists(tc):   # same file loco.py builds: true match count per train S1 (incl. blocking misses)
        import candidates
        q, p = candidates.load_split('train', ['entity_id'])
        P.labels(q, p).group_by('qi').agg(pl.len().alias('ntrue')).with_columns(pl.col('qi').cast(pl.UInt32)).write_parquet(tc)
    gt = pl.read_parquet(tc)
    nt = lambda cond: ctry.filter(cond).select('qi').join(gt, on='qi', how='left').with_columns(pl.col('ntrue').fill_null(0))
    nt_hold = nt((pl.col('country_n') == tr_c) & (h() >= 300) & (h() < 300 + EV))
    nt_ev = nt((pl.col('country_n') == ev_c) & (h() < EV))
    rule = sweep(hold, 'p', nt_hold, None, None)[0][1]            # tuned in-domain only, like the real models
    target = profile(apply_rule(hold, 'p', rule), nt_hold.select('qi'))
    print(f'rule tuned on held-out {tr_c}: {rule}; in-domain profile empty={target[0]:.4f} matches/S1={target[1]:.3f}')
    oracle = sweep(ev, 'p', nt_ev, None, None)[0]
    print(f'(oracle: rule tuned on {ev_c} labels = {oracle[0]:.5f})')
    curve = []
    for d in GRID:
        pred = apply_rule(shift(ev, 'p', d), 'p', rule)
        e, k = profile(pred, nt_ev.select('qi'))
        curve.append((d, e, k))
        print(f'  delta={d:5.2f}  empty={e:.4f}  matches/S1={k:.3f}  F0.5={D.macro_f05(pred, nt_ev)["macro_f05"]:.5f}', flush=True)
    for frac in (0.25, 0.5, 0.75, 1.0):
        d = pick(curve, target, frac)
        print(f'label-free pick frac={frac}: delta={d} -> F0.5 {D.macro_f05(apply_rule(shift(ev, "p", d), "p", rule), nt_ev)["macro_f05"]:.5f}')


def test(out_dir, country, frac, delta=None):
    from run_test import write_lists
    import candidates
    q, p = candidates.load_split('test', ['entity_id', 'country_n'])
    ids_q = q.select(pl.col('idx_q').alias('qi'), pl.col('entity_id').alias('source1_entity_id'))
    p_ids = p.select(pl.col('idx_p').alias('pi'), pl.col('entity_id').alias('cand_id'))
    ctry = q.select(pl.col('idx_q').alias('qi'), 'country_n')
    col = 'stage1' if os.environ.get('STAGES', '2') == '1' else 'stage2'   # same switch as run_test.py
    s = pl.read_parquet(os.path.join(P.CACHE, f'test_{col}.parquet')).with_columns(pl.col('qi').cast(pl.UInt32))
    rule = json.load(open(os.path.join(P.MODELS, 'decision_stage1.json' if col == 'stage1' else 'decision.json')))
    ind = ctry.filter(pl.col('country_n') != country).select('qi')
    target = profile(apply_rule(s, col, rule).join(ind, on='qi'), ind)
    tq = ctry.filter(pl.col('country_n') == country).select('qi')
    st = s.join(tq, on='qi')
    curve = [(d, *profile(apply_rule(shift(st, col, d), col, rule), tq)) for d in GRID]
    d = pick(curve, target, frac) if delta is None else delta   # explicit delta: chosen on the loco curve
    print(f'in-domain profile {target}; {country} curve {curve}; chosen delta={d}')
    s2 = pl.concat([s.join(ind, on='qi'), shift(st, col, d)])
    pred = apply_rule(s2, col, rule)
    cand = pl.concat([pl.read_parquet(f, columns=['qi', 'pi']) for f in P.parts('test', 'pr')])
    os.makedirs(out_dir, exist_ok=True)
    write_lists(ids_q, cand, p_ids, 'candidate_entity_ids', os.path.join(out_dir, 'candidate_pairs.tsv'))
    write_lists(ids_q, pred, p_ids, 'matched_entity_ids', os.path.join(out_dir, 'matching_results.tsv'))
    print(f'{country} delta={d}: predicted pairs {pred.height}')


def _demo():
    s = pl.DataFrame({'qi': [1, 1, 2], 'pi': [1, 2, 3], 'p': [0.5, 0.9, 0.2]})
    assert abs(shift(s, 'p', 0.0)['p'][0] - 0.5) < 1e-6 and shift(s, 'p', 1.0)['p'][0] < 0.5
    curve = [(0.0, 0.04, 3.5), (0.5, 0.05, 3.4), (1.0, 0.06, 3.3)]
    assert pick(curve, (0.06, 3.3), 1.0) == 1.0 and pick(curve, (0.06, 3.3), 0.5) == 0.5 and pick(curve, (0.06, 3.3), 0.0) == 0.0
    # smoke-test regression: tiny empty-rate steps must not override the halfway point on matches/S1
    c2 = [(0.0, .1878, 2.210), (0.5, .1888, 2.194), (1.25, .1898, 2.172), (2.75, .1937, 2.144)]
    assert pick(c2, (.1997, 2.172), 0.5) == 0.5
    print('calib demo ok')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a:
        _demo()
    elif a[0] == 'loco':
        loco(*a[1:3])
    else:
        test(a[1], a[2], float(a[3]), float(a[4]) if len(a) > 4 else None)
