"""Watch the shared Drive folder; download the two score files when they appear (only those two).
Usage: python drive_watch.py [--once]"""
import re, sys, time, os, urllib.request

FOLDER = '1ju3J4ceH8wZ2Hq9S4maSJrRXcePI0FUx'
WANT = ('train_oof_scores.parquet', 'test_scores.parquet')
HERE = os.path.dirname(os.path.abspath(__file__))


def listing():
    html = urllib.request.urlopen(f'https://drive.google.com/embeddedfolderview?id={FOLDER}', timeout=30).read().decode('utf-8', 'replace')
    out = {}
    for block in html.split('flip-entry"')[1:]:            # one block per entry
        m_id, m_name = re.search(r'/d/([A-Za-z0-9_-]+)', block), re.search(r'flip-entry-title">([^<]+)', block)
        if m_id and m_name:
            out[m_name.group(1).strip()] = m_id.group(1)
    return out


def fetch(name, fid):
    dst, part = os.path.join(HERE, name), os.path.join(HERE, name + '.part')
    url = f'https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t'
    urllib.request.urlretrieve(url, part)
    with open(part, 'rb') as f:
        head = f.read(4)
    if head != b'PAR1':                                     # parquet magic; anything else = an HTML error page
        raise RuntimeError(f'{name}: download is not a parquet file (got {head!r})')
    os.replace(part, dst)
    print(time.strftime('%H:%M'), 'downloaded', name, f'{os.path.getsize(dst) / 1e6:.1f} MB', flush=True)


def main(once=False):
    while True:
        try:
            files = listing()
            print(time.strftime('%H:%M'), 'folder has:', sorted(files), flush=True)
            for name in WANT:
                if name in files and not os.path.exists(os.path.join(HERE, name)):
                    fetch(name, files[name])
        except Exception as e:
            print(time.strftime('%H:%M'), 'retry later:', e, flush=True)
        if all(os.path.exists(os.path.join(HERE, n)) for n in WANT):
            print('BOTH FILES READY', flush=True)
            return
        if once:
            return
        time.sleep(120)


if __name__ == '__main__':
    main('--once' in sys.argv)
