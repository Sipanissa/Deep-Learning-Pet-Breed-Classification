"""Reproducible offline augmentation of the training split; validation and test are only copied byte for byte.

plan:      allocation per class and one plan row per future image (source, seed, exact parameters); writes no image.
self-test: applies the plan to a sample in memory (repeatability, size, difference from source); writes no file.
generate:  writes the images and a new split folder; refuses to run without --confirm.

Colour stays close to the source because coat colour separates several classes, so no hue change, grayscale,
vertical flip or erasing. Every parameter comes from the source hash, variant number and base seed, not from run order.
"""
import argparse,collections,hashlib,io,json,math,platform,random,shutil
from datetime import datetime,timezone
from pathlib import Path
import PIL
from PIL import Image,ImageEnhance,ImageFilter
from common import BASE,ROOT,read,write,sha,inspect,BKTree,variants,CLASS_NAMES,CLASS_SLUGS
from dataset_paths import resolve_dataset

# Grundwerte der Augmentation
BASE_SEED=20261106; EXTRA=7870; TARGET=15000; JPEG_QUALITY=95; MIN_SIDE=224; MAX_ATTEMPTS=5
tree_info={}
# Erlaubte Bereiche für die Änderungen am Bild
RANGES=dict(flip_probability=0.5,rotation_degrees=[-10,10],crop_area=[0.85,1.0],crop_position=[0.25,0.75],crop_ratio=[0.9,1.1],brightness=[0.85,1.15],contrast=[0.85,1.15],saturation=[0.9,1.1],blur_probability=0.1,blur_radius=[0.3,1.0])
PLAN_ROOT=BASE/'augmentation_plans'; OUT_ROOT=BASE/'augmented_versions'
FIELDS=['aug_id','breed','split','source_path','source_sha256','source_origin','source_license','variant','seed','params','relative_path']

# Wie viele Varianten jede Klasse braucht
def allocate(counts,extra=EXTRA):
    """Fills the classes to one common size; the remainder goes to the classes with the fewest originals."""
    total=sum(counts.values())+extra; level,rest=divmod(total,len(counts))
    assert level>=max(counts.values()),'a class is already above the common level'
    smallest=sorted(counts,key=lambda b:(counts[b],b))[:rest]
    return {b:level+(b in smallest)-counts[b] for b in counts}

# Zufällige Werte für eine Variante, durch den Seed wiederholbar
def parameters(seed):
    rng=random.Random(seed); r=RANGES; uni=lambda k:round(rng.uniform(*r[k]),4)
    blur=rng.random()<r['blur_probability']
    return dict(flip=rng.random()<r['flip_probability'],rotation=uni('rotation_degrees'),crop_area=uni('crop_area'),
        crop_ratio=round(math.exp(rng.uniform(math.log(r['crop_ratio'][0]),math.log(r['crop_ratio'][1]))),4),crop_x=uni('crop_position'),crop_y=uni('crop_position'),
        brightness=uni('brightness'),contrast=uni('contrast'),saturation=uni('saturation'),blur_radius=uni('blur_radius') if blur else 0.0)

# Größter Ausschnitt ohne leere Ecken nach der Drehung
def inscribed(w,h,degrees):
    """Largest upright rectangle inside the rotated image, so that no empty corner remains."""
    a=abs(math.radians(degrees)); s,c=math.sin(a),math.cos(a); long,short=max(w,h),min(w,h)
    if s==0: return w,h
    if short<=2*s*c*long:
        x=0.5*short; return (x/s,x/c) if w>=h else (x/c,x/s)
    d=c*c-s*s; return (w*c-h*s)/d,(h*c-w*s)/d

# Eine Variante aus einem Bild erzeugen
def augment(image,p):
    image=image.convert('RGB')
    if p['flip']: image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # Drehen und so zuschneiden, dass keine leeren Ecken bleiben
    w,h=image.size; rotated=image.rotate(p['rotation'],resample=Image.Resampling.BICUBIC)
    iw,ih=inscribed(w,h,p['rotation']); left,top=(w-iw)/2,(h-ih)/2
    area=iw*ih*p['crop_area']; cw=min(iw,math.sqrt(area*p['crop_ratio'])); ch=min(ih,math.sqrt(area/p['crop_ratio']))
    x=left+(iw-cw)*p['crop_x']; y=top+(ih-ch)*p['crop_y']
    box=(round(x),round(y),round(x+cw),round(y+ch)); result=rotated.crop(box)
    # Zu kleine Ausschnitte vergrößern
    if min(result.size)<MIN_SIDE:
        factor=MIN_SIDE/min(result.size); result=result.resize((math.ceil(result.width*factor),math.ceil(result.height*factor)),Image.Resampling.LANCZOS)
    # Helligkeit, Kontrast und Sättigung leicht ändern
    result=ImageEnhance.Brightness(result).enhance(p['brightness'])
    result=ImageEnhance.Contrast(result).enhance(p['contrast'])
    result=ImageEnhance.Color(result).enhance(p['saturation'])
    if p['blur_radius']: result=result.filter(ImageFilter.GaussianBlur(p['blur_radius']))
    return result

# Bild als JPEG in den Speicher schreiben
def encode(image):
    buffer=io.BytesIO(); image.save(buffer,'JPEG',quality=JPEG_QUALITY,subsampling=0); return buffer.getvalue()

def plan_dir(dataset): return PLAN_ROOT/f"{dataset['name']}_{TARGET}"

# Planen, welche Varianten entstehen, noch ohne Bilder zu schreiben
def plan(name):
    dataset=resolve_dataset(name); train=read(Path(dataset['train'])); sizes=[len(read(Path(dataset[s]))) for s in ('val','test')]
    assert len(train)+sizes[0]+sizes[1]+EXTRA==TARGET,(len(train),sizes)
    assert {r['breed'] for r in train}==set(CLASS_NAMES)
    licences={r['path']:r['license'] for r in read(Path(dataset['folder'])/'attribution.csv')}
    counts=collections.Counter(r['breed'] for r in train); extra=allocate(counts); rows=[]; summary=[]
    for breed in CLASS_NAMES:
        mine=[dict(r,sha256=sha((ROOT/r['path']).resolve())) for r in train if r['breed']==breed]
        assert len({r['sha256'] for r in mine})==len(mine),breed
        # Feste Reihenfolge, unabhängig von der Liste
        mine.sort(key=lambda r:hashlib.sha256(f"{BASE_SEED}|{r['sha256']}".encode()).hexdigest())
        used=collections.Counter()
        for k in range(extra[breed]):
            r=mine[k%len(mine)]; variant=used[r['sha256']]; used[r['sha256']]+=1
            seed=int(hashlib.sha256(f"{BASE_SEED}|{r['sha256']}|{variant}".encode()).hexdigest()[:16],16)
            aug_id=f"aug_{CLASS_SLUGS[breed]}_{r['sha256'][:16]}_v{variant}"
            origin=r['source'].split(' / ')[0].split(';')[0]
            rows.append(dict(aug_id=aug_id,breed=breed,split='train',source_path=r['path'],source_sha256=r['sha256'],source_origin=origin,
                source_license=licences.get(r['path'],'Datensatzebene oder akademische Nutzung, siehe Quelle'),variant=variant,seed=seed,
                params=json.dumps(parameters(seed),sort_keys=True),relative_path=f"images/{CLASS_SLUGS[breed]}/{aug_id}.jpg"))
        summary.append(dict(klasse=breed,train_original=counts[breed],zusaetzlich=extra[breed],train_gesamt=counts[breed]+extra[breed],faktor_je_original=round(extra[breed]/counts[breed],2),maximal_varianten_je_original=max(used.values()),val=sum(r['breed']==breed for r in read(Path(dataset['val']))),test=sum(r['breed']==breed for r in read(Path(dataset['test'])))))
    assert len(rows)==EXTRA and len({r['aug_id'] for r in rows})==EXTRA
    assert sum(s['train_gesamt'] for s in summary)+sizes[0]+sizes[1]==TARGET
    out=plan_dir(dataset); out.mkdir(parents=True,exist_ok=True)
    write(out/'augmentation_plan.csv',rows,FIELDS); write(out/'allocation.csv',summary,list(summary[0]))
    config=dict(dataset=dataset['name'],train_sha256=sha(Path(dataset['train'])),val_sha256=sha(Path(dataset['val'])),test_sha256=sha(Path(dataset['test'])),base_seed=BASE_SEED,extra_train_images=EXTRA,total_examples=TARGET,ranges=RANGES,
        jpeg_quality=JPEG_QUALITY,subsampling='4:4:4',min_side=MIN_SIDE,excluded_operations=['hue','grayscale','vertical flip','erasing','mixup','cutmix'],python=platform.python_version(),pillow=PIL.__version__,planned_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    (out/'augmentation_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PLAN',out,'zeilen',len(rows),'plan sha256',sha(out/'augmentation_plan.csv')[:16],flush=True)
    for s in summary: print(s,flush=True)

# Stichprobe prüfen, ohne Dateien zu schreiben
def self_test(name):
    dataset=resolve_dataset(name); rows=read(plan_dir(dataset)/'augmentation_plan.csv')
    sample=random.Random(BASE_SEED).sample(rows,60)+[next(r for r in rows if r['breed']==b) for b in CLASS_NAMES]
    extremes=[dict(rotation=d,flip=False,crop_area=0.8,crop_ratio=q,crop_x=0.0,crop_y=1.0,brightness=1.0,contrast=1.0,saturation=1.0,blur_radius=0.0) for d,q in ((10,0.9),(-10,1.1))]
    checked=0
    for r in sample:
        p=json.loads(r['params']); assert p==json.loads(json.dumps(parameters(int(r['seed'])),sort_keys=True)),'parameters differ from seed'
        source=(ROOT/r['source_path']).resolve(); assert sha(source)==r['source_sha256']
        with Image.open(source) as image:
            first=encode(augment(image,p)); second=encode(augment(image,p))
            assert first==second,'not repeatable'; assert hashlib.sha256(first).hexdigest()!=r['source_sha256']
            with Image.open(io.BytesIO(first)) as out: assert out.mode=='RGB' and min(out.size)>=MIN_SIDE,out.size
            for q in extremes:
                result=augment(image,q)
                assert min(result.size)>=MIN_SIDE
                # Mit einem weißen Testbild prüfen, ob schwarze Ecken entstehen
                blank=augment(Image.new('RGB',image.size,'white'),q); assert min(blank.getextrema()[c][0] for c in range(3))>=250,'rotation corner inside crop'
        checked+=1
    print('SELF-TEST PASS',checked,'Stichproben: Parameter aus Seed, Bytes wiederholbar, RGB, mindestens',MIN_SIDE,'px, keine leeren Rotationsecken; keine Datei geschrieben',flush=True)

# Suchbaum mit allen Validierungs- und Testbildern
def val_test_tree(dataset):
    tree=BKTree()
    for split in ('val','test'):
        for r in read(Path(dataset[split])):
            q=inspect((ROOT/r['path']).resolve()); assert not q['error'],r['path']
            tree.add(int(q['phash'],16),dict(path=r['path'],split=split,breed=r['breed']))
    return tree

# Varianten finden, die einem Validierungs- oder Testbild ähneln
def links(tree,path,phash,row):
    """pHash in 8 orientations against val and test (<= 10)."""
    found={}
    for h in {int(phash,16),*variants(path).values()}:
        for dist,o in tree.query(h,10):
            key=o['path']; found[key]=min(found.get(key,99),dist)
    return [dict(aug_id=row['aug_id'],breed=row['breed'],other=o,split=split,other_breed=breed,distance=d) for o,d,split,breed in ((o,d,*tree_info[o]) for o,d in found.items())]

# Zu ähnlich zum Test oder zur gleichen Klasse in der Validierung
def blocked(link): return link['split']=='test' or link['breed']==link['other_breed']

# Bilder erzeugen und neuen Datensatz schreiben
def generate(name,confirm):
    assert confirm,'generate needs --confirm; the project owner must approve the generation first'
    dataset=resolve_dataset(name); folder=plan_dir(dataset); config=json.loads((folder/'augmentation_config.json').read_text(encoding='utf-8'))
    # Daten dürfen sich seit der Planung nicht geändert haben
    for s in ('train','val','test'): assert sha(Path(dataset[s]))==config[f'{s}_sha256'],s+' changed since planning'
    assert PIL.__version__==config['pillow'],'other Pillow version than planned'
    out=OUT_ROOT/f"{dataset['name']}_{TARGET}"; assert not out.exists(),'target exists: '+str(out); out.mkdir(parents=True)
    rows=read(folder/'augmentation_plan.csv'); done=[]; remaining=[]; replaced=[]
    tree=val_test_tree(dataset); tree_info.clear()
    for s in ('val','test'): tree_info.update({r['path']:(s,r['breed']) for r in read(Path(dataset[s]))})
    for r in rows:
        path=out/r['relative_path']; path.parent.mkdir(parents=True,exist_ok=True); seed,params=int(r['seed']),json.loads(r['params'])
        # Zu ähnliche Varianten mit einem neuen Seed noch einmal erzeugen
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                seed=int(hashlib.sha256(f"{BASE_SEED}|{r['source_sha256']}|{r['variant']}|attempt{attempt}".encode()).hexdigest()[:16],16); params=parameters(seed)
            with Image.open((ROOT/r['source_path']).resolve()) as image: raw=encode(augment(image,params))
            path.write_bytes(raw); q=inspect(path); assert not q['error']
            found=links(tree,path,q['phash'],r); hard=[f for f in found if blocked(f)]
            if not hard: break
            replaced+=[dict(f,attempt=attempt,seed=seed) for f in hard]
        assert not hard,'no clean variant for '+r['aug_id']
        remaining+=found
        done.append(dict(r,seed=seed,params=json.dumps(params,sort_keys=True),attempt=attempt,local_path=str(path),output_sha256=q['sha256'],output_phash=q['phash'],width=q['width'],height=q['height']))
    write(out/'augmentation_manifest.csv',done,list(done[0]))
    if replaced: write(out/'augmentation_replacements.csv',replaced,list(replaced[0]))
    if remaining: write(out/'augmentation_leakage_check.csv',remaining,list(remaining[0]))
    print('LEAKAGE CHECK',len(done),'Bilder; ersetzt',len({f['aug_id'] for f in replaced}),'; verbleibende Treffer andere Klasse in Val',len(remaining),flush=True)
    # Listen schreiben, Validierung und Test unverändert kopieren
    originals=read(Path(dataset['train'])); augmented=[dict(path=d['local_path'],breed=d['breed'],source=f"Augmentiert aus {d['source_path']} / {d['aug_id']}") for d in done]
    shutil.copy2(dataset['train'],out/'train_original.csv'); write(out/'train_augmented.csv',augmented,['path','breed','source'])
    write(out/'train.csv',originals+augmented,['path','breed','source'])
    for s in ('val','test'): shutil.copy2(dataset[s],out/f'{s}.csv'); assert sha(out/f'{s}.csv')==sha(Path(dataset[s]))
    for f in ('augmentation_config.json','allocation.csv'): shutil.copy2(folder/f,out/f)
    print('GENERATED',out,len(done),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('phase',choices=['plan','self-test','generate']); ap.add_argument('--dataset',default='active'); ap.add_argument('--confirm',action='store_true')
    a=ap.parse_args()
    {'plan':lambda:plan(a.dataset),'self-test':lambda:self_test(a.dataset),'generate':lambda:generate(a.dataset,a.confirm)}[a.phase]()
