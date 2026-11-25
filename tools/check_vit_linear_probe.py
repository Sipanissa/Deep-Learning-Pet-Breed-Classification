"""Function test for the ViT-B/16 linear probe. No training, no model saved, no test image loaded, no dataset file changed.

Writes only an environment and result summary to the file given by --report.
"""
import argparse,collections,copy,hashlib,json,platform,sys,time,warnings
from pathlib import Path
from common import ROOT,BASE,sha
import PIL,torch,torch.nn as nn,torch_directml,torchvision
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataset_paths import resolve_dataset
from pet_dataset import PetDataset,CLASS_NAMES,CLASS_TO_IDX
from vit_linear_probe import WEIGHTS,build_vit_linear_probe,set_probe_mode

VARIANT=BASE/'augmented_versions/20261106T110038Z_15000'; ACTIVE=BASE/'combined_versions/20261106T110038Z'
ORDER=['British Shorthair','Russian Blue','Persian','Ragdoll','Bengal','Samoyed','Pomeranian','Keeshond','Great Pyrenees','Newfoundland']
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225); SEED=42

# Ein Bild je Klasse auswählen
def one_per_class(dataset):
    first={}
    for i,s in enumerate(dataset.samples): first.setdefault(s['breed'],i)
    return [first[b] for b in CLASS_NAMES]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--report',required=True); a=ap.parse_args()
    torch.manual_seed(SEED); report={}
    device=torch_directml.device(); assert torch_directml.is_available()
    report['umgebung']=dict(python=sys.version.split()[0],betriebssystem=platform.platform(),torch=torch.__version__,torchvision=torchvision.__version__,pillow=PIL.__version__,directml_geraet=torch_directml.device_name(0),directml_anzahl=torch_directml.device_count(),geraet=str(device))
    print('PASS Umgebung',report['umgebung'],flush=True)

    # Gewichte und Modell prüfen
    assert 'vit_b_16' in torchvision.models.list_models()
    start=time.time(); model,preprocess=build_vit_linear_probe(); load_seconds=time.time()-start
    checkpoint=Path(torch.hub.get_dir())/'checkpoints'/Path(WEIGHTS.url).name
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest(); assert digest.startswith('c867db91')
    # Der Checkpoint nutzt ältere Namen, torchvision benennt sie beim Laden um
    original={k.replace('mlp.linear_1','mlp.0').replace('mlp.linear_2','mlp.3'):v for k,v in WEIGHTS.get_state_dict(progress=False,check_hash=True).items()}; current=model.state_dict()
    assert set(original)==set(current) and not [k for k,v in original.items() if not k.startswith('heads.') and not torch.equal(v,current[k])]
    assert not torch.equal(original['heads.head.bias'][:10],current['heads.head.bias']) and current['heads.head.weight'].shape==(10,768)
    print('PASS Gewichte',WEIGHTS.url,'sha256',digest[:16],'Backbone identisch zum Checkpoint',len(original)-2,'Tensoren',flush=True)

    # Nur der Kopf darf trainierbar sein
    trainable={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}; total=sum(p.numel() for p in model.parameters())
    assert set(trainable)=={'heads.head.weight','heads.head.bias'} and model.heads.head.out_features==10 and model.heads.head.in_features==768
    assert sum(trainable.values())==7690 and all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('heads.head'))
    print('PASS Kopf 768->10; trainierbar',trainable,sum(trainable.values()),'von',total,flush=True)

    # Datensatz und Klassenreihenfolge prüfen
    dataset=resolve_dataset(str(VARIANT)); assert sha(VARIANT/'val.csv')==sha(ACTIVE/'val.csv') and sha(VARIANT/'test.csv')==sha(ROOT/'data/final_splits/test.csv')
    train=PetDataset(dataset['train'],transform=preprocess); val=PetDataset(dataset['val'],transform=preprocess)
    assert (len(train),len(val))==(12768,1233)
    print('PASS Datensatz',dataset['name'],'train',len(train),'val',len(val),'| val.csv bytegleich zur aktiven Version; Testbilder nicht geladen',flush=True)
    assert CLASS_NAMES==ORDER and list(CLASS_TO_IDX)==ORDER and list(CLASS_TO_IDX.values())==list(range(10))
    per=lambda d:collections.Counter(s['breed'] for s in d.samples)
    for d in (train,val):
        assert set(per(d))==set(ORDER)
        for i in one_per_class(d): _,label=d[i]; assert CLASS_NAMES[label]==d.samples[i]['breed']
    report['klassen']=dict(reihenfolge=ORDER,train={b:per(train)[b] for b in ORDER},val={b:per(val)[b] for b in ORDER})
    print('PASS Klassenreihenfolge und Labels',report['klassen'],flush=True)

    # Vorverarbeitung mit einer selbst gebauten vergleichen
    assert list(preprocess.crop_size)==[224] and list(preprocess.resize_size)==[256] and preprocess.interpolation==InterpolationMode.BILINEAR and tuple(round(x,3) for x in preprocess.mean)==MEAN and tuple(round(x,3) for x in preprocess.std)==STD
    manual=transforms.Compose([transforms.Resize(256,InterpolationMode.BILINEAR,antialias=True),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    probe=PetDataset(dataset['train'],transform=manual); x,_=train[0]; y,_=probe[0]
    assert x.shape==(3,224,224) and torch.allclose(x,y,atol=2e-2)
    report['vorverarbeitung']=dict(resize=256,center_crop=224,interpolation='bilinear',mean=MEAN,std=STD,max_abweichung_manuell=float((x-y).abs().max()))
    print('PASS Vorverarbeitung',report['vorverarbeitung'],'Wertebereich',round(float(x.min()),3),round(float(x.max()),3),flush=True)

    # Kleinen Batch auf CPU und GPU rechnen und vergleichen
    train_batch=next(iter(DataLoader(Subset(train,one_per_class(train)),batch_size=10))); val_batch=next(iter(DataLoader(Subset(val,one_per_class(val)),batch_size=10)))
    images,labels=train_batch; assert sorted(labels.tolist())==list(range(10))
    criterion=nn.CrossEntropyLoss(); set_probe_mode(model); features={}
    hook=model.heads.register_forward_hook(lambda _m,i,_o:features.__setitem__('x',i[0].detach().cpu()))
    with torch.no_grad(): cpu_out=model(images); cpu_features=features['x']
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=model.to(device); set_probe_mode(model)
        assert all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('heads.head'))
        outputs=model(images.to(device)); assert tuple(outputs.shape)==(10,10); dml_features=features['x']
        loss=criterion(outputs,labels.to(device)); assert loss.requires_grad; loss.backward(); loss_value=float(loss.item())
        assert 1.0<loss_value<5.0
        with_grad={n:float(p.grad.abs().sum().item()) for n,p in model.named_parameters() if p.grad is not None}
        assert set(with_grad)=={'heads.head.weight','heads.head.bias'} and all(v>0 for v in with_grad.values())
        unchanged=all(torch.equal(p.detach().cpu(),before[n]) for n,p in model.named_parameters()); assert unchanged
        with torch.no_grad(): val_out=model(val_batch[0].to(device)); assert tuple(val_out.shape)==(10,10); assert torch.isfinite(val_out.cpu()).all()
        difference=float((dml_features-cpu_features).abs().max())
        print('PASS Forward und Backward auf DirectML; Loss',round(loss_value,4),'| Gradienten nur',with_grad,'| kein Parameter veraendert',flush=True)
        print('INFO Merkmale DirectML gegen CPU: maximale Abweichung',difference,flush=True); assert difference<1e-2
        # DirectML warnt nur einmal je Prozess, deshalb vor dem Optimizer prüfen
        model_fallbacks=[str(w.message)[:120] for w in caught if 'fall back' in str(w.message)]; assert not model_fallbacks,model_fallbacks
        print('PASS Forward und Backward ohne Rueckfall auf die CPU',flush=True)
        # AdamW nur an einer Kopie des Kopfes testen
        copy_head=copy.deepcopy(model.heads).to(device); optimizer=torch.optim.AdamW(copy_head.parameters(),lr=1e-3)
        copy_loss=criterion(copy_head(dml_features.to(device)),labels.to(device)); copy_loss.backward(); optimizer.step()
        print('PASS AdamW-Schritt auf einer Wegwerf-Kopie des Kopfes laeuft',flush=True)
        # Geschwindigkeit grob messen
        model.zero_grad(set_to_none=True); batch=32; start=time.time(); big=next(iter(DataLoader(train,batch_size=batch,shuffle=False))); load=time.time()-start
        timing=[]
        for _ in range(4):
            start=time.time()
            with torch.no_grad(): float(model(big[0].to(device)).sum().item())
            timing.append(time.time()-start)
        forward=sum(timing[1:])/3
    fallbacks=sorted({str(w.message)[:220] for w in caught})
    report['funktionstest']=dict(batch=10,loss=loss_value,gradienten_nur=sorted(with_grad),merkmalsabweichung_directml_cpu=difference,warnungen=fallbacks)
    per_epoch=(forward+load)*(len(train)+len(val))/batch
    report['geschwindigkeit']=dict(batch=batch,bild_laden_sekunden=round(load,2),forward_sekunden=round(forward,2),grobe_schaetzung_epoche_minuten=round(per_epoch/60,1))
    report['modell']=dict(name='ViT-B/16',gewichte=str(WEIGHTS),url=WEIGHTS.url,sha256=digest,top1_imagenet=WEIGHTS.meta['_metrics']['ImageNet-1K']['acc@1'],parameter_gesamt=total,parameter_trainierbar=sum(trainable.values()),parameter_eingefroren=total-sum(trainable.values()),encoder_schichten=len(model.encoder.layers),patch=model.patch_size,bildgroesse=model.image_size,kopf=str(model.heads.head),ladezeit_sekunden=round(load_seconds,1))
    print('INFO Warnungen',fallbacks,flush=True); print('INFO Geschwindigkeit',report['geschwindigkeit'],flush=True)
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
