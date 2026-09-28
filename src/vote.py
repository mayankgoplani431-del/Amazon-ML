"""k-of-n vote over submission files (stdlib only, no dataset needed). S4 = E02 ∩ S3 was `vote.py out 2 cand E02 S3`.
Keeps a pair when >= k of the given matching_results.tsv files predict it; enforces matches ⊆ candidates and
each S2/S3 id -> at most one S1; prints the diff vs the first matching file.
Usage: python vote.py <out_dir>/matching_results.tsv <k> <cand1.tsv>[,<cand2.tsv>...] <matching1.tsv> <matching2.tsv> [...]
       (writes <out_dir>/candidate_pairs.tsv = union of the candidate files)
"""
import sys
from collections import Counter


def load(path):
    with open(path, encoding='utf-8') as f:
        next(f)
        return {a: [x for x in b.split(',') if x] for a, _, b in (ln.rstrip('\n').partition('\t') for ln in f)}


def vote(files, k):
    votes = Counter((s1, x) for m in files for s1, v in m.items() for x in v)
    return {p for p, n in votes.items() if n >= k}


def main(out, k, cand_paths, *paths):
    """cand_paths: comma-separated candidate files (models with different blocking); their union is written as
    candidate_pairs.tsv next to <out>, so matches ⊆ candidates holds for the voted file."""
    ms = [load(p) for p in paths]
    assert all(set(m) == set(ms[0]) for m in ms), 'S1 id sets differ between files'
    keep = vote(ms, int(k))
    kq = {}
    for s1, x in keep:
        kq.setdefault(s1, set()).add(x)
    # stream the candidate files line by line (each ~470 MB; loading several as sets would need ~8 GB)
    import os
    out_cand = os.path.join(os.path.dirname(os.path.abspath(out)), 'candidate_pairs.tsv')
    assert all(os.path.abspath(cp) != out_cand for cp in cand_paths.split(',')), 'output folder would overwrite an input candidate file'
    fs = [open(cp, encoding='utf-8') for cp in cand_paths.split(',')]
    bad, seen = 0, 0
    with open(out_cand, 'w', encoding='utf-8', newline='\n') as w:
        for f in fs:
            next(f)
        w.write('source1_entity_id\tcandidate_entity_ids\n')
        for lines in zip(*fs):
            rows = [ln.rstrip('\n').partition('\t') for ln in lines]
            s1 = rows[0][0]
            assert all(r[0] == s1 for r in rows), f'candidate files not in the same S1 order at {s1}'
            u = {x for r in rows for x in r[2].split(',') if x}
            bad += len(kq.get(s1, set()) - u)
            seen += 1
            w.write(f'{s1}\t{",".join(sorted(u))}\n')
    for f in fs:
        f.close()
    assert seen == len(ms[0]), f'candidate files have {seen} rows, matching files {len(ms[0])}'
    assert bad == 0, f'{bad} voted pairs are not in {cand_paths}'
    multi = sum(1 for n in Counter(x for _, x in keep).values() if n > 1)
    assert multi == 0, f'{multi} S2/S3 ids assigned to >1 S1 (use odd n or k > n/2)'
    by = {}
    for s1, x in keep:
        by.setdefault(s1, []).append(x)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1 in ms[0]:   # same row order as the input files
            f.write(f'{s1}\t{",".join(sorted(by.get(s1, [])))}\n')
    ref = {(s1, x) for s1, v in ms[0].items() for x in v}
    n = len(ms[0])
    print(f'{out}: {len(keep)} pairs ({len(keep) / n:.3f}/S1), empty {n - len(by)} ({100 * (n - len(by)) / n:.2f}%) | '
          f'vs {paths[0]}: +{len(keep - ref)} added, -{len(ref - keep)} removed')


def _demo():
    a = {'q1': ['S2-1', 'S3-1'], 'q2': ['S2-2'], 'q3': []}
    b = {'q1': ['S2-1'], 'q2': ['S2-2', 'S3-9'], 'q3': ['S3-5']}
    c = {'q1': ['S2-1', 'S3-1'], 'q2': [], 'q3': ['S3-5']}
    assert vote([a, b], 2) == {('q1', 'S2-1'), ('q2', 'S2-2')}                       # intersection
    assert vote([a, b, c], 2) == {('q1', 'S2-1'), ('q1', 'S3-1'), ('q2', 'S2-2'), ('q3', 'S3-5')}   # majority
    assert vote([a, b, c], 3) == {('q1', 'S2-1')}
    print('vote demo ok')


if __name__ == '__main__':
    _demo() if len(sys.argv) == 1 else main(*sys.argv[1:])
