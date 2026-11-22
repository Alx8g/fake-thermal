"""Fetch a sampled, aligned subset of SpeakingFaces into the pairs/ training layout.

SpeakingFaces (ISSAI; Abdrakhmanova et al., Sensors 21(10):3465, 2021; CC BY 4.0,
https://huggingface.co/datasets/issai/Speaking_Faces) has 142 people filmed close up by a
FLIR T540 (464x348) and a webcam, still (session 1) and reading commands aloud (session 2),
from nine angles. The full set is 2.1 TB of zipped PNG frames; this reads each subject zip's
central directory over HTTP and fetches only the sampled members by byte range (about 13 s
and 40 MB per zip).

Thermal frames are FLIR iron palette PNGs (224 colours, lossless): inverted to grey by palette
index. The dataset's aligned RGB is used; the warp's black fill is trimmed. Both are resized to
a 512 short side, like the other sources.

Stems: sf_<sub>_<trial>_<session>_<pos>_<clip>_<frame>. Official split: subjects 1-100 go to
train; 101-142 (valid + test) are held out in val.

    python scripts/fetch_speakingfaces.py --out data512/pairs
"""
import argparse, csv, io, json, random, struct, threading, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

try:  # only the download needs it; the palette and trim helpers do not
    import requests
except ImportError:  # pragma: no cover
    requests = None

BASE = 'https://huggingface.co/datasets/issai/Speaking_Faces/resolve/main/'
# The 224-entry FLIR iron palette, coldest first: every frame uses all of it. Ordered as the
# shortest path through the colours seen in 36 frames from six subjects (largest step 10).
IRON_HEX = (
    '00111800101d000e24000d29000c2f000b3600093b00084000074500064b0004500104550503580a025d0e0163110166'
    '15006b1a006e1c007121007424007a28007d2c00802f008333008636008839008b3d008e40008f430091460094490095'
    '4d009750009a53009b55009c5a009d5c009e5f00a06200a16500a26900a36c00a46f00a57100a77400a67700a77a00a8'
    '7c00a77f00a88200a88500a78700a88a00a78b01a68e01a79102a69403a69604a59904a49a06a39d07a2a007a1a109a0'
    'a40aa0a70a9fa80c9eaa0d9dad0e9aae0f99b11098b21298b51395b61494b91593ba1790bd188fbe1a8cc11b8cc21c89'
    'c51d88c61f85c72184ca2281cb2380cb247cce2579cf2779d02976d32a73d42b72d52d6fd62f6cd8316bda3269db3466'
    'dd3663de3862df395fe03b5ce13d5be23f59e44156e54353e64550e7464fe8484ce94a49eb4c46ec4e46ed5043ee5240'
    'ef533ff0553cf25739f35936f45b35f35d33f55f30f6612ff7632cf86529f86728f96925fa6b23fb6d22fb6f1ffc711e'
    'fd731bfd751afe7717ff7917ff7b14ff7d13ff7f10ff810fff830eff860bff870aff890aff8b09ff8c05ff8e04ff9003'
    'ff9202ff9401ff9600ff9800ff9a00ff9c00ff9e00ffa000ffa200ffa400ffa500ffa800ffaa00ffab00ffae00ffb000'
    'ffb100ffb300ffb500ffb700ffb900ffbb00ffbc00ffbe00ffc000ffc201ffc302ffc503ffc705ffc906ffcb07ffcc0a'
    'ffce0bffd00cffd110ffd311ffd514ffd617ffd818ffd91bffdb1fffdd22ffde25ffe028ffe12bffe32effe432ffe535'
    'ffe63affe83dffe942ffeb46ffec4bffec4fffed54ffee59fff05efef164fef269fdf36efdf473fdf57afcf680fcf787'
    'fbf88cfbf993fafa9bfafba2fafca7f8fdb0f7feb7f7ffbff6ffc6f6ffcff5ffd6f5ffe0f5ffe7f4fff0f4fff9f3ffff'
)
PALETTE = np.array([[int(IRON_HEX[i + k:i + k + 2], 16) for k in (0, 2, 4)]
                    for i in range(0, len(IRON_HEX), 6)], np.int64)
LOOKUP = {int(r) << 16 | int(g) << 8 | int(b): i for i, (r, g, b) in enumerate(PALETTE)}


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests."""

    def __init__(self, url, session=None):
        self.s = session or requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.size, self.pos = r.url, int(r.headers['content-length']), 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        r = self.s.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'}, timeout=120)
        r.raise_for_status()
        b[:len(r.content)] = r.content
        self.pos += len(r.content)
        return len(r.content)


def open_zip(path):
    f = HttpFile(BASE + path)
    return zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16)), f


local = threading.local()


def session():
    if not hasattr(local, 's'):
        local.s = requests.Session()
    return local.s


def fetch_member(url, info):
    start = info.header_offset
    end = start + 30 + len(info.orig_filename.encode()) + 256 + info.compress_size
    for attempt in range(6):
        try:
            r = session().get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raw = r.content
    name_len, extra_len = struct.unpack('<HH', raw[26:30])
    data = raw[30 + name_len + extra_len:30 + name_len + extra_len + info.compress_size]
    if info.compress_type == 8:
        data = zlib.decompress(data, -15)
    assert len(data) == info.file_size, info.filename
    return data


def thermal_to_grey(png):
    a = np.asarray(Image.open(io.BytesIO(png)).convert('RGB')).astype(np.int64)
    packed = a[..., 0] << 16 | a[..., 1] << 8 | a[..., 2]
    u, inv = np.unique(packed, return_inverse=True)
    idx = np.array([LOOKUP.get(int(v), -1) for v in u])
    miss = (idx < 0).sum()
    if miss:  # off-palette colours (none expected): nearest palette entry
        for k in np.where(idx < 0)[0]:
            c = np.array([u[k] >> 16, u[k] >> 8 & 255, u[k] & 255])
            idx[k] = int(np.argmin(((PALETTE - c) ** 2).sum(1)))
    grey = (idx[inv].reshape(packed.shape) * 255 / (len(PALETTE) - 1)).round().astype(np.uint8)
    return grey, int(miss)


def valid_box(rgb, frac=.01):
    """Trim edge rows/cols where the alignment warp left exact-black fill, worst edge first."""
    zero = (rgb == 0).all(2)
    t, b, l, r = 0, zero.shape[0], 0, zero.shape[1]
    while b - t > 64 and r - l > 64:
        share = {'t': zero[t, l:r].mean(), 'b': zero[b - 1, l:r].mean(),
                 'l': zero[t:b, l].mean(), 'r': zero[t:b, r - 1].mean()}
        side = max(share, key=share.get)
        if share[side] <= frac:
            break
        if side == 't': t += 1
        elif side == 'b': b -= 1
        elif side == 'l': l += 1
        else: r -= 1
    return t, b, l, r


def plan_subject(z, kind, split, rng, args):
    """Pick (thermal_info, rgb_info, stem) triples for one subject zip."""
    infos = {i.filename: i for i in z.infolist() if not i.is_dir() and '@eaDir' not in i.filename}
    thr_dir, rgb_dir = ('thr_image', 'rgb_image_aligned') if kind == 'io' else ('thr_image_cmd', 'rgb_image_cmd_aligned')
    clips = {}
    for name in infos:
        parts = name.split('/')
        if len(parts) != 4 or parts[2] != thr_dir:
            continue
        rgb_name = '/'.join(parts[:2] + [rgb_dir, parts[3][:-6] + '_3.png'])
        if rgb_name not in infos:
            continue
        f = parts[3][:-6].split('_')  # sub trial session pos [cmd] frame
        key = tuple(f[1:-1])  # trial, session, pos (+ cmd): one clip
        clips.setdefault(key, []).append((int(f[-1]), name, rgb_name))
    picks = []
    for key in sorted(clips):
        frames = sorted(clips[key])
        if kind == 'io':
            n = args.io_train if split == 'train' else args.io_val
        else:
            n = 1 if split == 'train' or rng.random() < args.ia_val_share else 0
        middle = frames[len(frames) // 10: len(frames) - len(frames) // 10] or frames
        for frame, thr, rgb in rng.sample(middle, min(n, len(middle))):
            f = thr.split('/')[-1][:-6].split('_')
            trial, sess, pos = f[1], f[2], f[3]
            clip = f[4] if len(f) == 6 else '0'
            stem = f'sf_{int(f[0]):03d}_{trial}_{sess}_{pos}_{clip}_{frame:04d}'
            picks.append((infos[thr], infos[rgb], stem))
    return picks


def process(url, thr_info, rgb_info, stem, out, size):
    if (out / 'thermal' / f'{stem}.png').exists():
        return stem, 0, 'skip'
    grey, miss = thermal_to_grey(fetch_member(url, thr_info))
    rgb = np.asarray(Image.open(io.BytesIO(fetch_member(url, rgb_info))).convert('RGB'))
    if rgb.shape[:2] != grey.shape:
        return stem, miss, f'shape {rgb.shape} vs {grey.shape}'
    t, b, l, r = valid_box(rgb)
    rgb, grey = rgb[t:b, l:r], grey[t:b, l:r]
    h, w = grey.shape
    scale = size / min(h, w)
    wh = (round(w * scale), round(h * scale))
    Image.fromarray(rgb).resize(wh, Image.LANCZOS).save(out / 'rgb' / f'{stem}.jpg', quality=95)
    Image.fromarray(grey).resize(wh, Image.BICUBIC).save(out / 'thermal' / f'{stem}.png')
    return stem, miss, f'trim {t},{b},{l},{r}'


def process_safe(*a):
    try:
        return process(*a)
    except Exception as e:  # one bad member must not stop the subject
        return a[3], 0, f'error {e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True, help='pairs folder (gets train/ and val/)')
    p.add_argument('--subjects', default='', help='comma list; default all')
    p.add_argument('--io-train', type=int, default=2, help='frames per still clip (trial x angle), train')
    p.add_argument('--io-val', type=int, default=1)
    p.add_argument('--ia-val-share', type=float, default=.34, help='share of speaking clips sampled for val')
    p.add_argument('--no-speaking', action='store_true')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    meta = list(csv.DictReader(io.StringIO(requests.get(BASE + 'metadata/subjects.csv', timeout=60).text)))
    split_of = {int(m['Sub_ID']): ('train' if m['Split'] == 'Train' else 'val') for m in meta}
    subs = [int(s) for s in args.subjects.split(',')] if args.subjects else sorted(split_of)
    for split in ('train', 'val'):
        for d in ('rgb', 'thermal'):
            (args.out / split / d).mkdir(parents=True, exist_ok=True)
    log = open(args.out / 'pull_log.jsonl', 'a')
    total = 0
    for sub in subs:
        split = split_of[sub]
        for kind, folder in (('io', 'image_only'), ('ia', 'image_audio')):
            if kind == 'ia' and args.no_speaking:
                continue
            path = f'{folder}/sub_{sub}_{kind}.zip'
            t0 = time.time()
            try:
                z, f = open_zip(path)
            except Exception as e:  # empty or missing zip (sub_2_io is 0 bytes; 16, 28 have no audio zip)
                print(f'SKIP {path}: {e}', flush=True)
                log.write(json.dumps({'zip': path, 'error': str(e)}) + '\n')
                continue
            rng = random.Random(f'{args.seed}-{sub}-{kind}')
            picks = plan_subject(z, kind, split, rng, args)
            out = args.out / split
            misses, bad = 0, []
            with ThreadPoolExecutor(args.workers) as pool:
                futs = [pool.submit(process_safe, f.url, ti, ri, stem, out, args.size) for ti, ri, stem in picks]
                for fut in as_completed(futs):
                    stem, miss, note = fut.result()
                    misses += miss
                    if note.startswith(('shape', 'error')):
                        bad.append(f'{stem}: {note}')
            total += len(picks)
            rec = {'zip': path, 'split': split, 'pairs': len(picks), 'bad': bad, 'off_palette': misses,
                   'seconds': round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(rec, f'total {total}', flush=True)
    print('PULL_DONE', total, flush=True)


if __name__ == '__main__':
    main()
