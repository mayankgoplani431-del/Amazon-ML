"""Streams the model outputs side by side (same S1 order, O(1) memory) and writes the candidate submissions
for the 6th file. Every combination is a subset of E06 or E07 predictions, so the one-owner-per-record rule holds.
  A = E02 ∩ S3 ∩ E07         (S5 recipe with E07 in place of E06)
  B = E02 ∩ S3 ∩ E06 ∩ E07   (S5 ∩ E07: 4-model agreement)
  C = S5 ∪ (E06 ∩ E07)       (S5 + pairs both new models agree on)
candidate_pairs.tsv for each = union of the E06, E07 and S3 candidate sets (what the models scored).
Usage: python combos.py <out_root> <E02.tsv> <S3.tsv> <E06.tsv> <E07.tsv> <cand_E06> <cand_E07> <cand_S3>
"""
import os, sys


def rows(path):
    with open(path, encoding='utf-8') as f:
        next(f)
        for ln in f:
            a, _, b = ln.rstrip('\n').partition('\t')
            yield a, {t for t in b.split(',') if t}


def main(root, e02, s3, e06, e07, c06, c07, cs3):
    outs = {k: os.path.join(root, k) for k in ('A', 'B', 'C')}
    for d in outs.values():
        os.makedirs(d, exist_ok=True)
    wm = {k: open(os.path.join(d, 'matching_results.tsv'), 'w', encoding='utf-8', newline='\n') for k, d in outs.items()}
    wc = [open(os.path.join(d, 'candidate_pairs.tsv'), 'w', encoding='utf-8', newline='\n') for d in outs.values()]
    for w in wm.values():
        w.write('source1_entity_id\tmatched_entity_ids\n')
    for w in wc:
        w.write('source1_entity_id\tcandidate_entity_ids\n')
    stats = {k: [0, 0] for k in wm}
    bad = 0
    for (s1, P02), (x3, P3), (x6, P6), (x7, P7), (y6, C6), (y7, C7), (y3, C3) in zip(*(rows(p) for p in (e02, s3, e06, e07, c06, c07, cs3))):
        assert s1 == x3 == x6 == x7 == y6 == y7 == y3, f'S1 order differs at {s1}'
        s5 = P02 & P3 & P6
        res = {'A': P02 & P3 & P7, 'B': s5 & P7, 'C': s5 | (P6 & P7)}
        cand = C6 | C7 | C3
        for k, v in res.items():
            bad += len(v - cand)
            wm[k].write(f'{s1}\t{",".join(sorted(v))}\n')
            stats[k][0] += len(v); stats[k][1] += not v
        line = f'{s1}\t{",".join(sorted(cand))}\n'
        for w in wc:
            w.write(line)
    for w in list(wm.values()) + wc:
        w.close()
    assert bad == 0, f'{bad} matches not in candidates'
    for k, (n, e) in stats.items():
        print(f'{k}: {n:,} pairs, {e:,} empty')


if __name__ == '__main__':
    main(*sys.argv[1:])
