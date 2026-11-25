"""Deletes unused files of foreign breeds and the ImageNet parquet from kandidaten_loeschliste.csv.

Without --ausfuehren it only checks and reports. Run from the project root. It stops before deleting anything if a
candidate is used by a training list or if its bytes changed.
"""
import argparse,collections,csv,hashlib,os,shutil
from pathlib import Path

ROOT=Path.cwd(); LIST=ROOT/'tools/kandidaten_loeschliste.csv'; BASE=ROOT/'data/dataset_expansion'

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as h: return list(csv.DictReader(h))
def key(path): return os.path.normpath(str(path)).lower()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Alle Bilder sammeln, die noch gebraucht werden
def used_paths():
    used=set()
    lists=[ROOT/f'data/final_splits/{s}.csv' for s in ('train','val','test')]
    for version in ('combined_versions/20261106T110038Z','combined_versions/20261104T194414Z','augmented_versions/20261106T110038Z_15000'):
        lists+=[BASE/version/f'{s}.csv' for s in ('train','val','test')]
    for path in lists:
        for r in rows(path):
            p=Path(r['path']); used.add(key(p if p.is_absolute() else ROOT/p))
    for r in rows(BASE/'augmented_versions/20261106T110038Z_15000/augmentation_manifest.csv'): used.add(key(ROOT/r['source_path']))
    return used

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ausfuehren',action='store_true'); a=ap.parse_args()
    assert (ROOT/'pet_dataset.py').exists(),'bitte aus dem Projekt-Hauptordner starten'
    used=used_paths(); present=[]; missing=0; sizes=collections.Counter(); counts=collections.Counter()
    # Erst prüfen: nichts löschen, was genutzt oder verändert wurde
    for r in rows(LIST):
        path=ROOT/r['pfad']
        if not path.exists(): missing+=1; continue
        assert key(path) not in used,f'wird von einer Trainingsliste genutzt: {path}'
        assert digest(path)==r['sha256'],f'Inhalt geaendert: {path}'
        present.append(path); sizes[r['grund'][:60]]+=int(r['bytes']); counts[r['grund'][:60]]+=1
    for name,size in sizes.most_common(): print(f'{size/1e9:7.3f} GB {counts[name]:6d} {name}')
    print('Geprueft',len(present),'Dateien,',round(sum(sizes.values())/1e9,3),'GB; bereits fehlend',missing,flush=True)
    if not a.ausfuehren: print('Nichts geloescht. Zum Loeschen: --ausfuehren'); return
    # Dateien löschen und leere Ordner entfernen
    free=shutil.disk_usage(ROOT).free; folders=set()
    for path in present: path.unlink(); folders.add(path.parent)
    for folder in sorted(folders,key=lambda p:-len(p.parts)):
        while folder!=ROOT and folder.exists() and not any(folder.iterdir()): folder.rmdir(); folder=folder.parent
    print('Geloescht',len(present),'Dateien; freigegeben',round((shutil.disk_usage(ROOT).free-free)/1e9,2),'GB',flush=True)

if __name__=='__main__': main()
