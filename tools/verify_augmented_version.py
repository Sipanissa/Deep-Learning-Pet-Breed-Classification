"""Independent check of a generated augmented version against its active base version; writes only its two report CSVs."""
import argparse,collections,json,random,re,sys
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from common import ROOT,read,write,inspect,sha,BKTree,variants
from pet_dataset import PetDataset,CLASS_NAMES,CLASS_TO_IDX
import prepare_augmentation as A
from dataset_paths import resolve_dataset

# Augmentierten Datensatz unabhängig prüfen
def main(folder):
    out=Path(folder); base=resolve_dataset('active'); config=json.loads((out/'augmentation_config.json').read_text(encoding='utf-8'))
    report={}
    assert base['name']==config['dataset']=='20261106T110038Z'; print('PASS aktive Version',base['name'],flush=True)
    assert sha(out/'val.csv')==sha(Path(base['val'])) and sha(out/'test.csv')==sha(Path(base['test']))==sha(ROOT/'data/final_splits/test.csv'); assert sha(out/'train_original.csv')==sha(Path(base['train']))
    print('PASS val.csv, test.csv, train_original.csv bytegleich zur aktiven Version',flush=True)
    # Anzahl und Reihenfolge der Bilder prüfen
    orig=read(out/'train_original.csv'); aug=read(out/'train_augmented.csv'); train=read(out/'train.csv'); val=read(out/'val.csv'); test=read(out/'test.csv'); man=read(out/'augmentation_manifest.csv')
    assert (len(orig),len(aug),len(train),len(val),len(test),len(man))==(4898,7870,12768,1233,999,7870); assert len(train)+len(val)+len(test)==15000
    assert [r['path'] for r in train]==[r['path'] for r in orig+aug]; print('PASS Zahlen 4898+7870=12768, 1233, 999, gesamt 15000',flush=True)
    alloc={r['klasse']:r for r in read(out/'allocation.csv')}; per=collections.Counter(r['breed'] for r in train); pa=collections.Counter(r['breed'] for r in aug)
    for b in CLASS_NAMES: assert per[b]==int(alloc[b]['train_gesamt']) and pa[b]==int(alloc[b]['zusaetzlich']),b
    print('PASS Verteilung je Klasse wie geplant',{b:(pa[b],per[b]) for b in CLASS_NAMES},flush=True)
    # Jede Variante muss aus einem Trainingsbild stammen
    src={r['path']:r for r in orig}; valtest={r['path'] for r in val+test}; bysha={sha((ROOT/r['path']).resolve()):r for r in orig}
    assert len({m['aug_id'] for m in man})==7870
    for m in man:
        s=src[m['source_path']]; assert m['breed']==s['breed'] and m['split']=='train' and m['source_path'] not in valtest and m['source_sha256'] in bysha,m['aug_id']
        p=json.loads(m['params']); assert abs(p['rotation'])<=10 and 0.85<=p['crop_area']<=1 and 0.25<=p['crop_x']<=0.75 and 0.25<=p['crop_y']<=0.75 and 0.9<=p['crop_ratio']<=1.1 and 0.85<=p['brightness']<=1.15 and 0.85<=p['contrast']<=1.15 and 0.9<=p['saturation']<=1.1 and p['blur_radius']<=1.0
        assert int(m['seed']) and sha(Path(m['local_path']))==m['output_sha256']
    print('PASS jedes Bild: Original aus Train, Klasse wie Original, Parameter im Bereich, Prüfsumme wie Manifest; Versuche',dict(collections.Counter(m['attempt'] for m in man)),flush=True)
    # Alle Bilder einmal laden
    t=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    for name,rows in (('train.csv',train),('val.csv',val),('test.csv',test)):
        ds=PetDataset(out/name,transform=t); n=0; c=collections.Counter()
        for images,labels in DataLoader(ds,batch_size=64,num_workers=0):
            n+=len(labels); c.update(int(x) for x in labels); assert images.shape[1:]==(3,224,224)
        assert n==len(rows) and all(c[CLASS_TO_IDX[b]]==k for b,k in collections.Counter(r['breed'] for r in rows).items()); print('PASS geladen',name,n,flush=True)
    # Größe prüfen und doppelte Bilder suchen
    smallest=[];hashes=collections.Counter();trainrows=[]
    for r in train:
        q=inspect((ROOT/r['path']).resolve()); assert not q['error']; hashes[q['sha256']]+=1; smallest.append(min(q['width'],q['height'])); trainrows.append(dict(r,phash=q['phash']))
    assert max(hashes.values())==1 and min(smallest)>=224; extra=[inspect((ROOT/r['path']).resolve())['sha256'] for r in val+test]; assert not set(extra)&set(hashes)
    smallest.sort(); print('PASS keine exakten Duplikate in 15000 Prüfsummen; Train-Kurzseite min/median',smallest[0],smallest[len(smallest)//2],flush=True)
    # Ähnliche Bilder in Validierung und Test suchen
    ref=BKTree()
    for r in val+test:
        q=inspect((ROOT/r['path']).resolve()); ref.add(int(q['phash'],16),dict(path=r['path'],breed=r['breed'],split='test' if r in test else 'val'))
    testpaths={r['path'] for r in test}; leak=collections.Counter(); bad=[]; other=[]
    for r in trainrows:
        for h in {int(r['phash'],16),*variants((ROOT/r['path']).resolve()).values()}:
            for d,o in ref.query(h,10):
                split='test' if o['path'] in testpaths else 'val'; same=r['breed']==o['breed']
                leak[(split,same)]+=1
                if split=='test' or same: bad.append((r['path'],o['path'],d))
                else: other.append(dict(path=r['path'],breed=r['breed'],other=o['path'],other_breed=o['breed'],distance=d))
    assert not bad,bad[:5]
    write(out/'pruefung_val_andere_klasse.csv',other,['path','breed','other','other_breed','distance']) if other else None
    print('PASS Leakage: kein Train-Bild pHash <=10 zum Test, keines zu Val derselben Klasse; Rest',dict(leak),'Paare',len({(x['path'],x['other']) for x in other}),flush=True)
    # Ähnliche Bilder innerhalb des Trainings suchen
    tree=BKTree(); source_of={m['local_path']:m['source_path'] for m in man}
    for r in trainrows: tree.add(int(r['phash'],16),r)
    same=0; different=[]; seen=set()
    for r in trainrows:
        for d,o in tree.query(int(r['phash'],16),5):
            if o['path']==r['path'] or (o['path'],r['path']) in seen: continue
            seen.add((r['path'],o['path']))
            a=source_of.get(r['path'],r['path']); b=source_of.get(o['path'],o['path'])
            if a==b: same+=1
            else: different.append(dict(a=r['path'],b=o['path'],distance=d,breed_a=r['breed'],breed_b=o['breed']))
    if different: write(out/'pruefung_nahe_paare_verschiedene_originale.csv',different,list(different[0]))
    print('INFO Train-intern pHash <=5: gleiche Herkunft',same,'| verschiedene Originale',len(different),collections.Counter(x['breed_a']==x['breed_b'] for x in different),flush=True)
    # Stichprobe neu berechnen, muss bytegleich sein
    sample=random.Random(20261106).sample(man,200)
    for m in sample:
        with Image.open((ROOT/m['source_path']).resolve()) as image: raw=A.encode(A.augment(image,json.loads(m['params'])))
        assert __import__('hashlib').sha256(raw).hexdigest()==m['output_sha256'],m['aug_id']
    print('PASS 200 zufällige Bilder aus Manifest bytegleich neu berechnet',flush=True)
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('folder'); main(ap.parse_args().folder)
