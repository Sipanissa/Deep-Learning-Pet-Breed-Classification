"""Read-only project check: versions load, ten classes, protected files intact, licence evidence per active image, tools parse."""
import ast,collections,hashlib,importlib,json,os,zipfile
from pathlib import Path
from common import ROOT,BASE,read,sha,protected_unchanged,CLASS_NAMES
from pet_dataset import PetDataset

# Erwartete Bildzahlen je Datensatzversion
VERSIONS={'combined_versions/20261106T110038Z':(4898,1233,999),'combined_versions/20261104T194414Z':(5817,1491,999),'augmented_versions/20261106T110038Z_15000':(12768,1233,999)}
ACTIVE=BASE/'combined_versions/20261106T110038Z'; VARIANT=BASE/'augmented_versions/20261106T110038Z_15000'; EVIDENCE=BASE/'nachweise'

# Pfade einheitlich schreiben, damit man sie vergleichen kann
def key(path):
    path=Path(path); return os.path.normpath(str(path if path.is_absolute() else ROOT/path)).lower()

# Für jedes Bild muss ein Lizenznachweis vorhanden sein
def licences():
    rows=[r for s in ('train','val','test') for r in read(ACTIVE/f'{s}.csv')]; assert len(rows)==7130
    attribution={key(r['path']) for r in read(ACTIVE/'attribution.csv')}
    felibreed={key(r['path']) for r in read(EVIDENCE/'lizenz/felibreed_attribution.csv')}
    imagenet={key(r['path']) for r in read(ACTIVE/'imagenet_260_auswahl.csv')}
    covered=collections.Counter()
    for r in rows:
        source=r['source']; path=key(r['path'])
        if source.startswith('Catza') or source.startswith('Openverse') or source.startswith('Open Images'): assert path in attribution,r['path']
        elif source.startswith('FeliBreed'): assert path in felibreed,r['path']
        elif source.startswith('ImageNet'): assert path in imagenet,r['path']
        else: assert source.startswith(('Oxford-IIIT Pet','Stanford Dogs')),source
        covered[source.split(' / ')[0][:24]]+=1
    assert (ROOT/'data/oxford-iiit-pet/annotations/README').exists()
    for name in ('catza_home.html','license_audit.csv','cc_by_4_legalcode.html','cat_breeds_LICENSE.txt'): assert (EVIDENCE/'lizenz'/name).exists(),name
    print('PASS Lizenznachweis je aktivem Bild',dict(covered),flush=True)

def main():
    # Aktive Version und alle Datensätze prüfen
    pointer=Path((BASE/'LATEST_COMBINED_VERSION.txt').read_text(encoding='utf-8').strip()); assert pointer.name=='20261106T110038Z' and pointer.exists(); print('PASS aktiver Zeiger',pointer.name,flush=True)
    for name,sizes in VERSIONS.items():
        data={s:PetDataset(BASE/name/f'{s}.csv') for s in ('train','val','test')}; got=tuple(len(data[s]) for s in ('train','val','test')); assert got==sizes,(name,got)
        for s,d in data.items(): assert {x['breed'] for x in d.samples}==set(CLASS_NAMES),(name,s)
        assert sha(BASE/name/'test.csv')==sha(ROOT/'data/final_splits/test.csv'); print('PASS Pfade und alle zehn Klassen',name,got,flush=True)
    assert len(CLASS_NAMES)==10
    # Geschützte Dateien und Original-Splits prüfen
    protected=json.loads((BASE/'protected_hashes.json').read_text(encoding='utf-8'))
    assert protected_unchanged(protected); print('PASS geschuetzte Dateien inkl. Oxford-Test',len(protected),flush=True)
    for f,digest in {'data/final_splits/train.csv':'39064ab53f87','data/final_splits/val.csv':'a0619d34f355','data/final_splits/test.csv':'158e85fc7cf3','pet_dataset.py':'39447a48b183'}.items(): assert sha(ROOT/f).startswith(digest),f
    print('PASS Original-Splits und PetDataset unveraendert',flush=True)
    # Jedes der 999 Testbilder einzeln prüfen
    test=read(ROOT/'data/final_splits/test.csv'); assert len(test)==999 and all('oxford-iiit-pet' in r['path'].replace('\\','/') for r in test)
    for r in test: assert sha(Path(r['path']))==protected[str(ROOT/Path(r['path']))],r['path']
    print('PASS 999 Oxford-Testbilder einzeln gegen geschuetzte Pruefsummen',flush=True)
    # Zustimmung zur Nutzung der ImageNet-Bilder prüfen
    d=json.loads((BASE/'USAGE_DECISIONS.json').read_text(encoding='utf-8'))['imagenet']; assert d['decision']=='ADOPT'
    assert sha(BASE/d['scope_file'])==d['scope_file_sha256']; assert (BASE/'academic_pool/imagenet/ACCEPTANCE.json').exists(); print('PASS ImageNet-Zustimmung und Auswahlliste',flush=True)
    for name in ('attribution.csv','ausschluss_manifest.csv','added_from_pools.csv','verification.json'): assert (ACTIVE/name).exists(),name
    licences()
    assert len(read(VARIANT/'augmentation_manifest.csv'))==7870; print('PASS Manifest der Variante',flush=True)
    # Gesicherte Nachweise prüfen
    for r in read(EVIDENCE/'PRUEFSUMMEN.csv'):
        assert hashlib.sha256((ROOT/r['datei']).read_bytes()).hexdigest()==r['sha256'],r['datei']
        if r['datei'].endswith('.zip'):
            with zipfile.ZipFile(ROOT/r['datei']) as z: assert z.testzip() is None,r['datei']
    assert (BASE/'activation_20261106.log').exists(); print('PASS gesicherte Nachweise und ZIP-Dateien',flush=True)
    # Alle Python-Dateien müssen lesbar sein
    bad=[];n=0
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('tools/*.py')):
        n+=1
        try: ast.parse(p.read_text(encoding='utf-8'))
        except SyntaxError as e: bad.append((p.name,str(e)))
    assert not bad,bad; print('PASS Python-Dateien lesbar',n,flush=True)
    for m in ('dataset_paths','prepare_augmentation'): importlib.import_module(m)
    print('PASS Kernmodule importierbar',flush=True)
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
