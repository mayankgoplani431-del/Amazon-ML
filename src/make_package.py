"""Builds <team>_submission.zip in the structure required by the official README:
  output/{matching_results.tsv, candidate_pairs.tsv}, code/business_entity_resolution/{src/, README.md,
  requirements.txt}, Documentation_template.md. Refuses to package outputs that fail the matches ⊆ candidates check.
Usage: python make_package.py <team> <output_dir> <documentation.md> [zip_dir]
"""
import os, sys, zipfile
from vote import load

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(team, out_dir, doc, zip_dir=None):
    m, c = (os.path.join(out_dir, f) for f in ('matching_results.tsv', 'candidate_pairs.tsv'))
    mm, cc = load(m), load(c)
    assert list(mm) == list(cc), 'matching and candidate files differ in S1 rows/order'
    bad = sum(1 for s1, v in mm.items() for x in v if x not in set(cc[s1]))
    assert bad == 0, f'{bad} matched ids are not candidates'
    del mm, cc
    path = os.path.join(zip_dir or os.path.dirname(REPO), f'{team}_submission.zip')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(m, 'output/matching_results.tsv')
        z.write(c, 'output/candidate_pairs.tsv')
        base = 'code/business_entity_resolution'
        for f in sorted(os.listdir(os.path.join(REPO, 'src'))):
            if f.endswith('.py'):
                z.write(os.path.join(REPO, 'src', f), f'{base}/src/{f}')
        z.write(os.path.join(REPO, 'scripts', 'run_e06.sh'), f'{base}/scripts/run_e06.sh')
        z.write(os.path.join(REPO, 'README.md'), f'{base}/README.md')
        z.write(os.path.join(REPO, 'requirements.txt'), f'{base}/requirements.txt')
        z.write(doc, 'Documentation_template.md')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    print(f'{path}: {os.path.getsize(path) / 1e6:.0f} MB, {len(names)} files')
    for n in names:
        if not n.startswith(f'code/business_entity_resolution/src/'):
            print('  ', n)


if __name__ == '__main__':
    main(*sys.argv[1:])
