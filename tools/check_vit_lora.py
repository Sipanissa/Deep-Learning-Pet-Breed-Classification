"""Technical test of vit_lora.py: correctness, DirectML compatibility, memory and stability. No full training.

Uses a few stratified training and validation images; never opens a test image; changes no dataset file; writes only
into --out (configurations, a test checkpoint and the results).
"""
import argparse,copy,hashlib,json,math,os,re,subprocess,time,warnings
from pathlib import Path
from common import ROOT
import PIL.Image,torch,torch.nn as nn,torch_directml
from torch.utils.data import DataLoader,Subset
from torchvision.models import vit_b_16
from pet_dataset import CLASS_NAMES,PetDataset
from train_vit_linear_probe import DEFAULT_DATASET,macro_f1,stratified_subset
from vit_lora import CONFIGS,WEIGHTS,AdamWNoLerp,LoRAAttention,build_vit_lora,load_lora_checkpoint,save_lora_checkpoint

opened=[]; _open=PIL.Image.open
# Geöffnete Bilder mitschreiben, um Testbilder zu erkennen
def recording_open(fp,*a,**k): opened.append(str(fp)); return _open(fp,*a,**k)

class GpuMemory:
    """Samples the dedicated GPU memory of this process from the Windows performance counter."""
    def __enter__(self):
        pattern=f"\\GPU Process Memory(pid_{os.getpid()}_*)\\Dedicated Usage"
        script=f"while($true){{try{{$s=(Get-Counter '{pattern}' -ErrorAction Stop).CounterSamples|Measure-Object CookedValue -Sum;[Console]::WriteLine($s.Sum)}}catch{{[Console]::WriteLine('x')}}}}"
        self.proc=subprocess.Popen(['powershell.exe','-NoProfile','-Command',script],stdout=subprocess.PIPE,text=True); time.sleep(2); return self
    def __exit__(self,*a):
        time.sleep(2); self.proc.kill(); values=[float(v) for v in self.proc.stdout.read().split() if re.fullmatch(r'[0-9.]+',v)]
        self.peak_gb=round(max(values)/1e9,2) if values else None; self.samples=len(values)

# Prüfsumme der eingefrorenen Gewichte
def fingerprint(model):
    h=hashlib.sha256()
    for n,p in sorted(model.named_parameters()):
        if not p.requires_grad: h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    for n,b in sorted(model.named_buffers()): h.update(n.encode()); h.update(b.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

# Warnungen sammeln, wenn DirectML auf die CPU ausweicht
def fallbacks(caught): return sorted({str(w.message)[:110] for w in caught if 'fall back' in str(w.message)})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    PIL.Image.open=recording_open; R={}; device=torch_directml.device()

    # Namen der Schichten in torchvision nachsehen
    plain=vit_b_16(weights=None); att=plain.encoder.layers[0].self_attention
    R['modulnamen']=dict(klasse=type(att).__name__,parameter=[n for n,_ in att.named_parameters()],in_proj_weight=list(att.in_proj_weight.shape),
        aufteilung='Zeilen 0-767 Query, 768-1535 Key, 1536-2303 Value',beispiel='encoder.layers.encoder_layer_0.self_attention')
    print('INFO Modulnamen',R['modulnamen'],flush=True)

    # Modelle mit den drei Rängen aufbauen
    torch.manual_seed(42); model,pre,config=build_vit_lora(8)
    for rank in (4,8,16):
        _,_,c=build_vit_lora(rank); (out/f'lora_r{rank}_config.json').write_text(json.dumps(c,indent=2,ensure_ascii=False),encoding='utf-8')
        R.setdefault('konfigurationen',{})[rank]={k:c[k] for k in ('rank','alpha','scaling','dropout','targets','lora_parameters','trainable_parameters')}
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert all(('lora_' in n) or n.startswith('heads.head.') for n in names) and len(names)==12*4+2
    assert config['lora_parameters']==12*2*2*8*768 and config['trainable_parameters']==config['lora_parameters']+7690
    print('PASS trainierbar nur LoRA und Kopf',R['konfigurationen'],flush=True)

    # Mit B = 0 muss das Modell wie das Original rechnen.
    # LoRA muss dasselbe ergeben wie direkt geänderte Gewichte.
    base=vit_b_16(weights=WEIGHTS); base.heads.head=copy.deepcopy(model.heads.head); base.eval(); model.eval()
    x=torch.randn(2,3,224,224)
    with torch.no_grad(): d0=float((model(x)-base(x)).abs().max())
    lora=copy.deepcopy(model.encoder.layers[0].self_attention); torch.nn.init.normal_(lora.lora_B_query,std=0.02); torch.nn.init.normal_(lora.lora_B_value,std=0.02)
    merged=copy.deepcopy(base.encoder.layers[0].self_attention)
    with torch.no_grad():
        merged.in_proj_weight[:768]+=lora.scaling*lora.lora_B_query@lora.lora_A_query; merged.in_proj_weight[1536:]+=lora.scaling*lora.lora_B_value@lora.lora_A_value
        t=torch.randn(2,197,768); dm=float((lora(t,t,t)[0]-merged(t,t,t,need_weights=False)[0]).abs().max())
        key_same=torch.equal(merged.in_proj_weight[768:1536],base.encoder.layers[0].self_attention.in_proj_weight[768:1536])
    assert d0<1e-4 and dm<1e-4 and key_same
    R['korrektheit']=dict(abweichung_B0_gegen_original=d0,abweichung_gegen_zusammengefuehrte_gewichte=dm,key_unveraendert=key_same)
    print('PASS Korrektheit',R['korrektheit'],flush=True)

    # Eigener Optimizer muss dasselbe rechnen wie AdamW
    torch.manual_seed(0); w1=nn.Parameter(torch.randn(10,768)); w2=nn.Parameter(w1.detach().clone())
    o1=torch.optim.AdamW([w1],lr=1e-3,weight_decay=0.01); o2=AdamWNoLerp([w2],lr=1e-3,weight_decay=0.01)
    for _ in range(5):
        g=torch.randn(10,768); w1.grad=g.clone(); w2.grad=g.clone(); o1.step(); o2.step()
    dopt=float((w1-w2).abs().max()); assert dopt<1e-6; R['optimizer_abweichung_zu_torch_adamw']=dopt
    print('PASS AdamWNoLerp gleich torch.optim.AdamW, Abweichung',dopt,flush=True)

    # Datensatz laden
    train=PetDataset(DEFAULT_DATASET+'/train.csv',transform=pre); val=PetDataset(DEFAULT_DATASET+'/val.csv',transform=pre)
    assert (len(train),len(val))==(12768,1233)
    ti=stratified_subset(train,32,42); vi=stratified_subset(val,64,42)
    images,labels=next(iter(DataLoader(Subset(train,ti[:8]),batch_size=8)))
    vimg,vlab=next(iter(DataLoader(Subset(val,vi),batch_size=64)))

    # Gradienten auf GPU und CPU vergleichen
    torch.manual_seed(42); safe,_,_=build_vit_lora(8,dropout=0.0); torch.manual_seed(42); ref,_,_=build_vit_lora(8,dropout=0.0,directml_safe_layernorm=False)
    with torch.no_grad():
        for (n,p),(m,q) in zip(safe.named_parameters(),ref.named_parameters()):
            assert n==m
            if 'lora_B' in n: p.normal_(0,0.02)
            q.copy_(p)
    def grads(m,dev):
        m=m.to(dev); m.train(); l=nn.CrossEntropyLoss()(m(images.to(dev)),labels.to(dev)); l.backward()
        return {n:p.grad.detach().cpu() for n,p in m.named_parameters() if p.requires_grad}
    gc=grads(ref,'cpu')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); gd=grads(safe,device)
    rel=max(float((gd[k]-gc[k]).abs().max()/(gc[k].abs().max()+1e-12)) for k in gc)
    assert rel<1e-3 and not fallbacks(caught) and len(gc)==50
    R['gradientenvergleich_directml_cpu']=dict(tensoren=len(gc),max_relative_abweichung=rel,referenz='CPU mit eingebauter LayerNorm, LoRA-B zufaellig, Dropout 0')
    print('PASS Gradienten des ganzen Modells DirectML gegen CPU, max. relative Abweichung',rel,flush=True)
    del safe,ref

    # Drei Trainingsschritte auf der GPU testen
    model=model.to(device); frozen_before=fingerprint(model); start={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=AdamWNoLerp([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0.01); criterion=nn.CrossEntropyLoss(); phases={}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model.train()
        outputs=model(images.to(device)); loss=criterion(outputs,labels.to(device)); phases['forward']=fallbacks(caught)
        loss.backward(); phases['backward']=fallbacks(caught)
        grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.grad is not None}
        assert set(grads)==set(start)
        assert all(v>0 for n,v in grads.items() if 'lora_B' in n or 'heads' in n) and all(v==0 for n,v in grads.items() if 'lora_A' in n)
        optimizer.step(); phases['optimizer']=fallbacks(caught); losses=[float(loss.item())]
        for _ in range(2):
            optimizer.zero_grad(); loss=criterion(model(images.to(device)),labels.to(device)); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad(): vout=model(vimg.to(device)).cpu()
        phases['validierung']=fallbacks(caught)
    assert not any(phases.values()),phases
    assert fingerprint(model)==frozen_before
    changed={n:float((p.detach().cpu()-start[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
    assert all(v>0 for v in changed.values())
    conf=torch.bincount(vlab*10+vout.argmax(1),minlength=100).view(10,10); f1,_=macro_f1(conf)
    vloss=float(criterion(vout,vlab)); vacc=100*float(conf.diag().sum())/len(vlab)
    R['funktionstest']=dict(trainbatch=8,schritte=3,loss=losses,cpu_rueckfaelle=phases,gradienten='nur LoRA und Kopf; LoRA-A im ersten Schritt exakt 0, weil B mit 0 startet',
        geaenderte_trainierbare_tensoren=len(changed),eingefrorene_gewichte_unveraendert=True,validierung=dict(bilder=64,loss=vloss,accuracy=vacc,macro_f1=f1))
    print('PASS Forward, Backward, 3 Schritte ohne CPU-Rueckfall; Loss',[round(l,4) for l in losses],'| eingefroren unveraendert | alle',len(changed),'LoRA/Kopf-Tensoren veraendert',flush=True)
    print('PASS Validierung 64 Bilder: Loss',round(vloss,4),'Acc',vacc,'Macro-F1',round(f1,4),flush=True)

    # Checkpoint speichern und wieder laden
    path=out/'technischer_test_checkpoint.pt'; save_lora_checkpoint(path,model.cpu(),config,epoch=0,note='technischer Test, kein Training')
    loaded,_,lc,_=load_lora_checkpoint(path); loaded.eval(); model.eval()
    with torch.no_grad(): dl=float((loaded(vimg[:4])-model(vimg[:4])).abs().max())
    assert dl<1e-5 and lc==json.loads(json.dumps(config,sort_keys=True)); R['checkpoint']=dict(datei=path.name,bytes=path.stat().st_size,abweichung_nach_laden=dl)
    print('PASS Checkpoint speichern und laden, Abweichung',dl,'Groesse',path.stat().st_size,flush=True)

    # Gleicher Seed muss gleiche Ergebnisse liefern
    def short_run():
        torch.manual_seed(42); m,_,_=build_vit_lora(8); m=m.to(device); m.train(); o=AdamWNoLerp([p for p in m.parameters() if p.requires_grad],lr=1e-3)
        ls=[]
        for _ in range(2):
            o.zero_grad(); l=criterion(m(images.to(device)),labels.to(device)); l.backward(); o.step(); ls.append(float(l.item()))
        return ls
    r1,r2=short_run(),short_run(); R['reproduzierbarkeit']=dict(lauf1=r1,lauf2=r2,identisch=r1==r2)
    print('INFO Reproduzierbarkeit DirectML',r1,r2,'identisch' if r1==r2 else 'NICHT identisch',flush=True)

    # Speicher und Tempo je Batchgröße messen
    R['speicher_und_zeit']={}
    for batch in (8,16,32):
        torch.manual_seed(42); m,_,_=build_vit_lora(8); m=m.to(device); m.train(); o=AdamWNoLerp([p for p in m.parameters() if p.requires_grad],lr=1e-3)
        xb,yb=next(iter(DataLoader(Subset(train,ti[:batch]),batch_size=batch)))
        try:
            with GpuMemory() as mem:
                times=[]
                for _ in range(4):
                    s=time.time(); o.zero_grad(); l=criterion(m(xb.to(device)),yb.to(device)); l.backward(); o.step(); lv=float(l.item()); times.append(time.time()-s)
                    assert math.isfinite(lv)
            step=sum(times[1:])/3
            R['speicher_und_zeit'][batch]=dict(gpu_speicher_spitze_gb=mem.peak_gb,messpunkte=mem.samples,sekunden_pro_schritt=round(step,2),bilder_pro_sekunde=round(batch/step,1),
                geschaetzte_epoche_minuten=round(len(train)/batch*step/60,1),status='stabil')
        except RuntimeError as e:
            R['speicher_und_zeit'][batch]=dict(status='Fehler',meldung=str(e)[:300])
        print('INFO Batch',batch,R['speicher_und_zeit'][batch],flush=True)
        del m,o

    # Es darf kein Testbild geöffnet worden sein
    with open(ROOT/'data/final_splits/test.csv',encoding='utf-8-sig',newline='') as h:
        import csv; test={str(Path(r['path'])).lower() for r in csv.DictReader(h)}
    seen={str(Path(p)).lower() for p in opened}; assert not seen&test; R['geoeffnete_bilder']=len(seen); R['testbilder_geoeffnet']=0
    print('PASS kein Oxford-Testbild geoeffnet; geoeffnete Bilddateien',len(seen),flush=True)
    R['umgebung']=dict(torch=torch.__version__,directml=torch_directml.device_name(0).strip('\x00'))
    (out/'technischer_test.json').write_text(json.dumps(R,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALLES BESTANDEN',flush=True)

if __name__=='__main__': main()
