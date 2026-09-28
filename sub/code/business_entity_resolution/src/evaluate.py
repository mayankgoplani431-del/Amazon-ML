"""Competition metric (macro F0.5 per Source-1 entity) + experiment scorecard/log."""
import json, os, time
import numpy as np
from common import WORK

EXP_DIR = os.path.join(WORK, 'experiments')
os.makedirs(EXP_DIR, exist_ok=True)


def f05(tp, npred, ntrue):
    """Per-entity F0.5 with the competition's singleton convention."""
    if ntrue == 0:
        return 1.0 if npred == 0 else 0.0
    if npred == 0 or tp == 0:
        return 0.0
    p, r = tp / npred, tp / ntrue
    return 1.25 * p * r / (0.25 * p + r)


def score(pred: dict, truth: dict, s1_ids, cands: dict = None):
    """pred/truth/cands: s1 -> set of S2/S3 ids. Returns the scorecard dict."""
    fs, tp_all, np_all, nt_all, sing_ok, sing_n, fp, fn = [], 0, 0, 0, 0, 0, 0, 0
    blk_hit = 0
    for s in s1_ids:
        p, t = pred.get(s, set()), truth.get(s, set())
        tp = len(p & t)
        fs.append(f05(tp, len(p), len(t)))
        tp_all += tp; np_all += len(p); nt_all += len(t)
        fp += len(p) - tp; fn += len(t) - tp
        if not t:
            sing_n += 1; sing_ok += (not p)
        if cands is not None:
            blk_hit += len(cands.get(s, set()) & t)
    sc = dict(n_s1=len(fs), macro_f05=float(np.mean(fs)),
              pair_precision=tp_all / max(np_all, 1), pair_recall=tp_all / max(nt_all, 1),
              singleton_acc=sing_ok / max(sing_n, 1), false_pos=fp, false_neg=fn)
    if cands is not None:
        sizes = np.array([len(cands.get(s, ())) for s in s1_ids])
        sc.update(blocking_recall=blk_hit / max(nt_all, 1), avg_cands=float(sizes.mean()),
                  p95_cands=float(np.percentile(sizes, 95)), max_cands=int(sizes.max()))
    return sc


def scorecard_text(name, sc):
    rows = [('Blocking recall', f"{100 * sc.get('blocking_recall', float('nan')):.2f} %"),
            ('Avg candidates / S1', f"{sc.get('avg_cands', float('nan')):.1f}"),
            ('P95 candidates / S1', f"{sc.get('p95_cands', float('nan')):.0f}"),
            ('Pair precision', f"{100 * sc['pair_precision']:.2f} %"),
            ('Pair recall', f"{100 * sc['pair_recall']:.2f} %"),
            ('Macro F0.5', f"{sc['macro_f05']:.4f}"),
            ('Singleton accuracy', f"{100 * sc['singleton_acc']:.2f} %"),
            ('False positives', f"{sc['false_pos']}"),
            ('False negatives', f"{sc['false_neg']}")]
    w = 40
    lines = ['┌' + '─' * w + '┐', '│' + f' EXPERIMENT SCORECARD: {name}'[:w].ljust(w) + '│', '├' + '─' * w + '┤']
    lines += ['│ ' + k.ljust(24) + v.rjust(w - 26) + ' │' for k, v in rows]
    lines.append('└' + '─' * w + '┘')
    return '\n'.join(lines)


def log_experiment(name, sc, config: dict, status='INCONCLUSIVE', notes=''):
    """Appends to experiments/log.jsonl and scorecards.md, returns scorecard text."""
    rec = dict(experiment_id=name, date=time.strftime('%Y-%m-%d %H:%M'), status=status, notes=notes,
               config=config, **sc)
    with open(os.path.join(EXP_DIR, 'log.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    txt = scorecard_text(name, sc)
    with open(os.path.join(EXP_DIR, 'scorecards.md'), 'a', encoding='utf-8') as f:
        f.write(f"\n### {name}  ({rec['date']})  [{status}]\n{notes}\n\nconfig: `{json.dumps(config)}`\n\n```\n{txt}\n```\n")
    return txt


if __name__ == '__main__':
    assert abs(f05(2, 3, 2) - 0.7142857) < 1e-6          # README example
    assert f05(0, 0, 0) == 1.0 and f05(0, 1, 0) == 0.0     # singleton convention
    sc = score({'a': {'x', 'y', 'z'}, 'b': set()}, {'a': {'x', 'z'}, 'b': set()}, ['a', 'b'])
    assert abs(sc['macro_f05'] - (0.7142857 + 1) / 2) < 1e-6
    print(scorecard_text('selftest', {**sc, 'blocking_recall': 1, 'avg_cands': 3, 'p95_cands': 3}))
