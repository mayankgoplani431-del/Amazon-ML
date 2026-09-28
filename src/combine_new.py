"""Final file for E08: S5 + E08's matches on NEWLY FOUND candidates only (streamed, same S1 order everywhere).
A pair (s1, c) from E08 is added only if c was NOT a candidate of s1 for the models in S5 (E06 set or legacy S3 set:
nobody in the vote ever judged it, so they abstain instead of voting no) and c is not already matched to another
S1 in S5 (one-owner rule). Nothing in S5 is removed. candidate_pairs = union of E06, S3 and E08 candidate sets.
Usage: python combine_new.py <out_dir> <S5_matching> <E08_matching> <cand_E06> <cand_S3> <cand_E08>
"""
import os, sys


def rows(path):
    with open(path, encoding='utf-8') as f:
        next(f)
        for ln in f:
            a, _, b = ln.rstrip('\n').partition('\t')
            yield a, [t for t in b.split(',') if t]


def main(out, s5, e08, c06, cs3, c08):
    os.makedirs(out, exist_ok=True)
    used = {c for _, v in rows(s5) for c in v}                  # pool ids already owned in S5
    wm = open(os.path.join(out, 'matching_results.tsv'), 'w', encoding='utf-8', newline='\n')
    wc = open(os.path.join(out, 'candidate_pairs.tsv'), 'w', encoding='utf-8', newline='\n')
    wm.write('source1_entity_id\tmatched_entity_ids\n'); wc.write('source1_entity_id\tcandidate_entity_ids\n')
    added = blocked = pairs = empty = 0
    for (s1, P5), (a, P8), (b, C6), (c, C3), (d, C8) in zip(*(rows(p) for p in (s5, e08, c06, cs3, c08))):
        assert s1 == a == b == c == d, f'S1 order differs at {s1}'
        old = set(C6) | set(C3)
        new = [x for x in P8 if x not in old and x not in set(P5)]
        keep = [x for x in new if x not in used]
        blocked += len(new) - len(keep)
        used.update(keep)
        m = sorted(set(P5) | set(keep))
        added += len(keep); pairs += len(m); empty += not m
        wm.write(f'{s1}\t{",".join(m)}\n')
        wc.write(f'{s1}\t{",".join(sorted(old | set(C8)))}\n')
    wm.close(); wc.close()
    print(f'{out}: {pairs:,} pairs ({added:,} added on new candidates; {blocked:,} skipped: record already owned in S5), empty {empty:,}')


if __name__ == '__main__':
    main(*sys.argv[1:])
