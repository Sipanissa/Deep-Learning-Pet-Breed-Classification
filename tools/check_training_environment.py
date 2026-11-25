"""Read-only check of the active dataset for training: every image through PetDataset, labels against paths.

--augmentation-sample N also runs the real generate step on N planned images in a temporary folder, loads the result
and deletes it again. No model is trained and nothing in the project changes.
"""
import argparse,collections,json,re,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from common import read,write,CLASS_SLUGS
from pet_dataset import PetDataset,CLASS_NAMES,CLASS_TO_IDX
from dataset_paths import resolve_dataset,sha256,EXPECTED_COUNTS
import prepare_augmentation as A

# ImageNet-Kennungen der Rassen
SYNSETS={'n02112350':'Keeshond','n02111500':'Great Pyrenees','n02111277':'Newfoundland','n02123394':'Persian','n02111889':'Samoyed','n02112018':'Pomeranian'}

ALIASES={'Persian':['persian'],'Pomeranian':['pomeranian'],'Samoyed':['samoyed'],'Keeshond':['keeshond'],'Newfoundland':['newfoundland'],'Great Pyrenees':['great_pyrenees','great-pyrenees'],
         'British Shorthair':['british_shorthair'],'Russian Blue':['russian_blue'],'Ragdoll':['ragdoll'],'Bengal':['bengal']}
# Rasse aus dem Dateipfad ableiten
def label_from_path(path):
    text=path.replace('\\','/').lower(); synset=re.search(r'n0\d{7}',text)
    if synset and synset.group(0) in SYNSETS: return {SYNSETS[synset.group(0)]}
    return {b for b,names in ALIASES.items() if any(n in text for n in names)}

# Alle Bilder einmal laden
def load_all(dataset):
    cnn=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])
    resnet=ResNet18_Weights.DEFAULT.transforms()
    for split in ('train','val','test'):
        start=time.time(); seen=0; per=collections.Counter()
        for name,transform in (('cnn',cnn),('resnet18',resnet)):
            data=PetDataset(dataset[split],transform=transform); n=0
            for images,labels in DataLoader(data,batch_size=64,shuffle=False,num_workers=0):
                assert images.shape[1:]==(3,224,224); n+=len(labels)
                if name=='cnn': per.update(int(x) for x in labels)
            assert n==len(data); seen=n
        print(f'LOAD {split}: {seen} Bilder mit beiden Vorverarbeitungen, {time.time()-start:.0f}s |',{CLASS_NAMES[i]:per[i] for i in range(10)},flush=True)
        yield split,seen,per

# Label und Dateipfad dürfen sich nicht widersprechen
def labels_against_paths(dataset):
    conflict=[];unknown=collections.Counter()
    for split in ('train','val','test'):
        for r in read(Path(dataset[split])):
            found=label_from_path(r['path'])
            if not found: unknown[r['source'].split(' / ')[0][:24]]+=1
            elif r['breed'] not in found: conflict.append((split,r['breed'],r['path']))
    print('LABEL gegen Pfad: Widersprueche',len(conflict),conflict[:5],'| ohne Rasse im Pfad',dict(unknown),flush=True)
    assert not conflict

# Augmentation an wenigen Bildern in einem temporären Ordner testen
def augmentation_sample(dataset,n):
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp); source=A.plan_dir(dataset); plan=read(source/'augmentation_plan.csv')
        rows=[r for b in CLASS_NAMES for r in [x for x in plan if x['breed']==b][:max(1,n//10)]]
        fake=tmp/'plans'/source.name; fake.mkdir(parents=True); write(fake/'augmentation_plan.csv',rows,A.FIELDS)
        for f in ('augmentation_config.json','allocation.csv'): shutil.copy2(source/f,fake/f)
        keep=A.PLAN_ROOT,A.OUT_ROOT; A.PLAN_ROOT,A.OUT_ROOT=tmp/'plans',tmp/'out'
        try: A.generate('active',True)
        finally: A.PLAN_ROOT,A.OUT_ROOT=keep
        out=tmp/'out'/source.name; train=read(out/'train.csv')
        assert len(train)==len(read(Path(dataset['train'])))+len(rows) and sha256(out/'val.csv')==sha256(dataset['val']) and sha256(out/'test.csv')==sha256(dataset['test'])
        manifest=read(out/'augmentation_manifest.csv'); assert all(min(int(m['width']),int(m['height']))>=224 for m in manifest)
        for script in ('train_resnet18.py',):
            run=subprocess.run([sys.executable,'-X','utf8',script,'--dataset',str(out),'--online-flip','off','--check-only'],capture_output=True,text=True,encoding='utf-8')
            assert run.returncode==0,run.stderr[-500:]
            print('E2E',script,[l for l in run.stdout.splitlines() if l.startswith('CHECK train') or l.startswith('CHECK OK')][-1][:60],flush=True)
        print('E2E Probe:',len(rows),'Bilder erzeugt, geprueft, geladen; temporaerer Ordner wird geloescht',flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='active'); ap.add_argument('--augmentation-sample',type=int,default=0); a=ap.parse_args()
    assert list(CLASS_NAMES)==list(CLASS_TO_IDX) and len(CLASS_NAMES)==10 and set(CLASS_NAMES)==set(CLASS_SLUGS); print('KLASSEN',CLASS_TO_IDX,flush=True)
    dataset=resolve_dataset(a.dataset); print('DATENSATZ',dataset['name'],flush=True)
    sizes={split:seen for split,seen,_ in load_all(dataset)}
    if dataset['name'] in EXPECTED_COUNTS: assert (sizes['train'],sizes['val'],sizes['test'])==EXPECTED_COUNTS[dataset['name']]
    labels_against_paths(dataset)
    if a.augmentation_sample: augmentation_sample(dataset,a.augmentation_sample)
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
