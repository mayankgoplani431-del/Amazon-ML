"""Independent second validation of the two submission files (in addition to utils/validate_submission.py).
Usage: python check_output.py <out_dir> <test_dataset_dir>
"""
import sys, os, collections


def read(path, col):
    raw = open(path, 'rb').read()
    assert b'\r' not in raw, 'CRLF line endings found'
    lines = raw.decode('utf-8').split('\n')
    assert lines[-1] == '', 'file must end with a newline'
    assert lines[0] == f'source1_entity_id\t{col}', f'bad header: {lines[0]!r}'
    rows = {}
    for i, ln in enumerate(lines[1:-1], 2):
        parts = ln.split('\t')
        assert len(parts) == 2, f'line {i}: expected 2 tab-separated columns, got {len(parts)}'
        s1, ids = parts
        assert s1 not in rows, f'duplicate S1 row {s1}'
        lst = ids.split(',') if ids else []
        assert len(lst) == len(set(lst)), f'duplicate ids in list for {s1}'
        for x in lst:
            assert x and x.strip() == x and x.lower() not in ('nan', 'none', 'null'), f'bad id {x!r} for {s1}'
        rows[s1] = lst
    return rows


def ids_of(path):
    with open(path, encoding='utf-8') as f:
        next(f)
        return [ln.split('\t', 1)[0] for ln in f if ln.strip()]


def main(out, test):
    s1 = ids_of(os.path.join(test, 'test_source1.tsv'))
    pool = set(ids_of(os.path.join(test, 'test_source2.tsv'))) | set(ids_of(os.path.join(test, 'test_source3.tsv')))
    m = read(os.path.join(out, 'matching_results.tsv'), 'matched_entity_ids')
    c = read(os.path.join(out, 'candidate_pairs.tsv'), 'candidate_entity_ids')
    for name, rows in (('matching', m), ('candidate', c)):
        assert len(rows) == len(s1) == len(set(s1)), f'{name}: {len(rows)} rows vs {len(s1)} test S1'
        assert set(rows) == set(s1), f'{name}: S1 id set differs from test_source1'
        bad = [x for v in rows.values() for x in v if x not in pool or not (x.startswith('S2-') or x.startswith('S3-'))]
        assert not bad, f'{name}: {len(bad)} ids not in test S2/S3, e.g. {bad[:3]}'
    not_sub = sum(1 for k, v in m.items() for x in v if x not in set(c[k]))
    assert not_sub == 0, f'{not_sub} matched ids are not in candidate_pairs'
    # each S2/S3 id should be matched to at most one S1 (exclusivity rule of the pipeline)
    owner = collections.Counter(x for v in m.values() for x in v)
    multi = sum(1 for n in owner.values() if n > 1)
    n_pairs = sum(len(v) for v in m.values())
    n_empty = sum(1 for v in m.values() if not v)
    print(f'CHECK PASS: {len(m)} S1 rows, {n_pairs} matched pairs, {n_empty} empty (singletons) '
          f'({100 * n_empty / len(m):.2f}%), avg candidates {sum(len(v) for v in c.values()) / len(c):.1f}, '
          f'ids matched to >1 S1: {multi}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
