"""Shared helpers for the tools; import this first, it makes the project root importable. Run tools from the project root."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/dataset_expansion'
sys.path.insert(0, str(ROOT))

from pet_dataset import CLASS_NAMES  # noqa: E402

# Klassennamen in Schreibweise für Ordner und Dateien
CLASS_SLUGS = {name: name.lower().replace(' ', '_') for name in CLASS_NAMES}


# Prüfsumme einer Datei berechnen
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


# CSV-Datei lesen
def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


# CSV-Datei schreiben, erst als .tmp, damit keine halbe Datei entsteht
def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


# Bild prüfen und Größe, Prüfsumme und pHash bestimmen
def inspect(path):
    result = {'sha256': sha(path), 'phash': '', 'width': 0, 'height': 0, 'error': ''}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            if im.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('unsupported format')
            im = ImageOps.exif_transpose(im)
            result.update(width=im.width, height=im.height,
                          phash=str(imagehash.phash(im.convert('RGB'))))
    except Exception as exc:
        result['error'] = str(exc)
    return result


# Suchbaum, um ähnliche pHash-Werte schnell zu finden
class BKTree:
    def __init__(self):
        self.root = None

    def add(self, key, item):
        if self.root is None:
            self.root = [key, [item], {}]
            return
        node = self.root
        while True:
            d = (key ^ node[0]).bit_count()
            if not d:
                node[1].append(item)
                return
            if d not in node[2]:
                node[2][d] = [key, [item], {}]
                return
            node = node[2][d]

    def query(self, key, radius):
        stack = [self.root] if self.root else []
        result = []
        while stack:
            node = stack.pop()
            d = (key ^ node[0]).bit_count()
            if d <= radius:
                result.extend((d, item) for item in node[1])
            stack.extend(child for edge, child in node[2].items() if d - radius <= edge <= d + radius)
        return result


# pHash für gedrehte und gespiegelte Fassungen eines Bildes
def variants(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        ops = {
            'rotate90': Image.Transpose.ROTATE_90, 'rotate180': Image.Transpose.ROTATE_180,
            'rotate270': Image.Transpose.ROTATE_270, 'mirror': Image.Transpose.FLIP_LEFT_RIGHT,
            'flip_vertical': Image.Transpose.FLIP_TOP_BOTTOM, 'transpose': Image.Transpose.TRANSPOSE,
            'transverse': Image.Transpose.TRANSVERSE}
        return {name: int(str(imagehash.phash(rgb.transpose(op))), 16) for name, op in ops.items()}


# Prüfen, ob alle geschützten Dateien unverändert sind
def protected_unchanged(protected):
    """Passes when each protected file has its digest, allowing recorded script updates and recorded moves."""
    updates = json.loads((BASE / 'protected_script_updates.json').read_text(encoding='utf-8'))
    moves = json.loads((BASE / 'protected_moves.json').read_text(encoding='utf-8'))
    for path, digest in protected.items():
        rel = Path(path).relative_to(ROOT).as_posix()
        for old, new in moves.items():
            if rel == old or rel.startswith(old + '/'):
                rel = new + rel[len(old):]
                break
        current = sha(ROOT / rel)
        if current == digest:
            continue
        # Nur freigegebene Änderungen sind erlaubt
        update = updates.get(Path(path).name)
        if not update or update['old'] != digest or update['new'] != current:
            return False
    return True
